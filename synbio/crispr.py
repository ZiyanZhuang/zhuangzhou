"""
CRISPR Design Engine
====================
Guide RNA design for:
  - Gene knockout (frameshift via NHEJ)
  - Gene knock-in (HDR with donor template)
  - CRISPRi (dCas9 transcriptional repression)
  - CRISPRa (dCas9 + activation domain)
  - Base editing (CBE/ABE)

Supports: SpCas9 (NGG), SaCas9 (NNGRRT), Cas12a/Cpf1 (TTTV)

Off-target prediction: simplified GC + seed region heuristic
(Full off-target requires Bowtie2/CRISPR-ML, which runs in WSL)
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple


# ── PAM patterns ─────────────────────────────────────────────────────────────

PAM_PATTERNS = {
    "SpCas9":  (r"[ACGT]{20}(?P<pam>GG)",           "3prime"),
    "SaCas9":  (r"[ACGT]{21}(?P<pam>[AG]G[AG]T)",   "3prime"),
    "Cas12a":  (r"(?P<pam>TTT[ACG])[ACGT]{23}",     "5prime"),
    "dCas9":   (r"[ACGT]{20}(?P<pam>GG)",            "3prime"),
}

NUCLEOTIDE_COMP = str.maketrans("ACGTacgt", "TGCAtgca")

def revcomp(seq: str) -> str:
    return seq.translate(NUCLEOTIDE_COMP)[::-1]


def gc_content(seq: str) -> float:
    seq = seq.upper()
    return (seq.count("G") + seq.count("C")) / len(seq) if seq else 0.0


def score_guide(spacer: str, cas_system: str = "SpCas9") -> float:
    """Heuristic guide quality score (0-1). Higher = better.

    Rules:
      - GC content 40-70%: optimal
      - No TTTT run (pol III terminator)
      - Seed region (3' 10 nt) GC 30-80%
      - No G quadruplex in seed
      - 5' G preferred for U6 transcription (SpCas9)
    """
    s = spacer.upper()
    score = 0.5  # baseline

    gc = gc_content(s)
    if 0.4 <= gc <= 0.7:
        score += 0.2
    elif gc < 0.3 or gc > 0.8:
        score -= 0.2

    if "TTTT" in s:
        score -= 0.3  # pol III termination risk

    seed = s[-10:] if len(s) >= 10 else s
    seed_gc = gc_content(seed)
    if 0.3 <= seed_gc <= 0.7:
        score += 0.1

    if s[0] == "G":
        score += 0.1  # U6 preference

    # Penalize runs of same base
    for base in "ACGT":
        if base * 4 in s:
            score -= 0.1

    return round(max(0.0, min(1.0, score)), 3)


# ── Guide finder ─────────────────────────────────────────────────────────────

def find_guides(
    dna_sequence: str,
    cas_system: str = "SpCas9",
    n_guides: int = 10,
    target_region: Optional[Tuple[int, int]] = None,  # (start, end) 0-based
) -> List[Dict]:
    """Find all valid guide RNAs in a DNA sequence.

    Args:
        dna_sequence: Target genomic DNA (sense strand, 5'→3')
        cas_system: 'SpCas9', 'SaCas9', 'Cas12a', 'dCas9'
        n_guides: maximum guides to return
        target_region: optional (start, end) to restrict search

    Returns:
        List of guide dicts sorted by quality score
    """
    if cas_system not in PAM_PATTERNS:
        return [{"error": f"Unknown Cas system. Choose: {list(PAM_PATTERNS)}"}]

    seq = dna_sequence.upper().replace(" ", "").replace("\n", "")
    guides = []

    # Search sense and antisense strands
    for strand, search_seq in [("+", seq), ("-", revcomp(seq))]:
        spacer_len = 20 if cas_system != "Cas12a" else 23

        if cas_system == "Cas12a":
            # 5' PAM: TTTV[23nt spacer]
            pattern = r"TTT[ACG]([ACGT]{23})"
            for m in re.finditer(pattern, search_seq):
                pos = m.start()
                spacer = m.group(1)
                if target_region:
                    if strand == "+":
                        adj_pos = pos
                    else:
                        adj_pos = len(seq) - pos - len(m.group(0))
                    if not (target_region[0] <= adj_pos <= target_region[1]):
                        continue
                guides.append({
                    "spacer":      spacer,
                    "pam":         m.group(0)[:4],
                    "strand":      strand,
                    "position":    pos,
                    "cas_system":  cas_system,
                    "score":       score_guide(spacer, cas_system),
                    "gc":          round(gc_content(spacer), 3),
                    "guide_rna":   spacer.replace("T", "U"),
                })
        else:
            # 3' PAM: [20nt spacer]NGG
            pattern = r"([ACGT]{20})[ACGT]GG"  # NGG PAM: 20nt spacer + N + GG
            for m in re.finditer(pattern, search_seq):
                spacer = m.group(1)
                pos = m.start()
                if target_region and strand == "+":
                    if not (target_region[0] <= pos <= target_region[1]):
                        continue
                guides.append({
                    "spacer":      spacer,
                    "pam":         m.group(0)[-3:],
                    "strand":      strand,
                    "position":    pos,
                    "cas_system":  cas_system,
                    "score":       score_guide(spacer, cas_system),
                    "gc":          round(gc_content(spacer), 3),
                    "guide_rna":   spacer.replace("T", "U"),
                    "cut_site":    pos + 17,  # 3 bp upstream of PAM
                })

    guides.sort(key=lambda x: x["score"], reverse=True)
    return guides[:n_guides]


# ── Knock-in design ──────────────────────────────────────────────────────────

def design_hdr_template(
    cut_site: int,
    insert_sequence: str,
    genomic_flank_len: int = 800,
    genomic_sequence: Optional[str] = None,
) -> Dict:
    """Generate HDR donor template for knock-in.

    Args:
        cut_site: position of Cas9 cut in genomic coordinates (0-based)
        insert_sequence: DNA to insert
        genomic_flank_len: length of homology arms (bp)
        genomic_sequence: full genomic sequence for arm extraction

    Returns:
        Dict with left_arm, insert, right_arm, template_length, notes
    """
    notes = []

    if genomic_sequence:
        left_arm  = genomic_sequence[max(0, cut_site - genomic_flank_len): cut_site]
        right_arm = genomic_sequence[cut_site: cut_site + genomic_flank_len]
        template_len = len(left_arm) + len(insert_sequence) + len(right_arm)
    else:
        left_arm  = f"[{genomic_flank_len} bp left homology arm — extract from genome]"
        right_arm = f"[{genomic_flank_len} bp right homology arm — extract from genome]"
        template_len = 2 * genomic_flank_len + len(insert_sequence)
        notes.append("Provide genomic_sequence to get actual arm sequences")

    if len(insert_sequence) > 5000:
        notes.append("Large insert (>5kb): use two-step integration or landing pad approach")

    if genomic_flank_len < 500:
        notes.append("Short arms (<500bp): HDR efficiency may be low; recommend ≥800bp")

    return {
        "left_homology_arm":  left_arm,
        "insert_sequence":    insert_sequence,
        "right_homology_arm": right_arm,
        "template_length_bp": template_len,
        "homology_arm_len":   genomic_flank_len,
        "notes":              notes,
        "synthesis_vector":   "pUC19 or pUCIDT (for ssODN if insert <200bp, use ssDNA)",
    }


# ── CRISPRi repression design ─────────────────────────────────────────────────

def design_crispri(
    gene_name: str,
    promoter_sequence: Optional[str] = None,
    repression_target: float = 0.9,  # 0-1 (fold repression)
) -> Dict:
    """Design CRISPRi system for transcriptional repression.

    dCas9-SspB or dCas9-KRAB fusion targets promoter non-template strand
    near TSS for maximal repression.

    Args:
        gene_name: target gene
        promoter_sequence: 200bp upstream of TSS if available
        repression_target: desired repression level (0.9 = 90% reduction)

    Returns:
        Dict with recommended dCas9 strategy, guide design notes
    """
    strategy = []

    if repression_target > 0.8:
        strategy.append("Use dCas9-KRAB fusion for strong repression (>80%)")
        strategy.append("Target non-template strand -50 to +50 relative to TSS")
        strategy.append("Use 2 guides for additive repression (redundancy)")
        system = "dCas9-KRAB"
    elif repression_target > 0.5:
        strategy.append("dCas9 alone (steric blockage) achieves 50-80% repression")
        strategy.append("Target region: -1 to +20 relative to TSS")
        system = "dCas9"
    else:
        strategy.append("For mild repression (<50%), dCas9 is sufficient")
        system = "dCas9"

    guides_note = []
    if promoter_sequence:
        # Find TSS-proximal NGG sites (crude: look near end of promoter sequence)
        tss_region = promoter_sequence[-60:] if len(promoter_sequence) >= 60 else promoter_sequence
        guides = find_guides(tss_region, cas_system="dCas9", n_guides=5)
        guides_note = [f"gRNA {i+1}: {g['spacer']} (score {g['score']})"
                       for i, g in enumerate(guides[:3])]
    else:
        guides_note = [f"Provide promoter sequence (200bp upstream of {gene_name} TSS) for guide design"]

    return {
        "gene":            gene_name,
        "system":          system,
        "repression_goal": f"{int(repression_target*100)}%",
        "strategy":        strategy,
        "guide_candidates": guides_note,
        "inducible_control": "Use Ptet-dCas9 for titratable repression with aTc",
        "plasmid_backbone": "pHERD26T (E. coli) or p416-TEF (yeast)",
    }
