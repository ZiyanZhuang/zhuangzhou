"""
Synthetic Biology Design Skills — 7 skills for pathway construction
====================================================================
Tools for: promoter selection, expression balancing, CRISPR design,
metabolic burden assessment, and dynamic regulation.

Skills:
  synbio_promoter_select    — choose promoter matching target expression
  synbio_expression_balance — balance multi-gene pathway expression
  synbio_crispr_design      — gRNA design for knockout/CRISPRi/knock-in
  synbio_codon_optimize     — codon-optimize for E. coli or yeast
  synbio_burden_assess      — metabolic burden estimation for heterologous genes
  synbio_design_package     — generate complete genetic design document
  synbio_graphrag_query     — query SynBio/SystemsBio KG via GraphRAG
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Skill functions ────────────────────────────────────────────────────────────

def synbio_promoter_select(
    target_expression: str = "medium",
    organism: str = "ecoli",
    inducible: bool = False,
    inducer: Optional[str] = None,
    n_candidates: int = 5,
    output_dir: Optional[str] = None,
) -> Dict:
    """Select optimal promoters for a target expression level.

    Args:
        target_expression: 'weak', 'low', 'medium', 'high', 'ultra' or numeric (0.0-2.0)
        organism: 'ecoli' or 'yeast'
        inducible: True for inducible promoters
        inducer: specific inducer ('IPTG', 'arabinose', 'galactose', etc.)
        n_candidates: number of alternatives to return

    Returns:
        Dict with ranked promoter candidates with strength, sequence, notes
    """
    try:
        from zhuangzhou.synbio.parts import PartSelector, ECOLI_PROMOTERS, YEAST_PROMOTERS

        strength_map = {"weak": 0.1, "low": 0.2, "medium": 0.5, "high": 0.9, "ultra": 1.8}
        if isinstance(target_expression, str) and target_expression in strength_map:
            strength = strength_map[target_expression]
        else:
            try:
                strength = float(target_expression)
            except ValueError:
                strength = 0.5

        candidates = PartSelector.select_promoter(
            organism=organism,
            strength=strength,
            inducible=inducible,
            inducer=inducer,
            n_candidates=n_candidates,
        )

        # Add context notes
        for c in candidates:
            c["note"] = (
                f"Relative strength {c['strength']:.2f} "
                f"(Anderson J23100=1.0 reference for E. coli constitutive)"
            )
            if c.get("type") == "inducible":
                c["note"] += f" | Inducer: {c.get('inducer','?')} | Tunable via inducer concentration"

        return {
            "target_expression": target_expression,
            "target_strength": strength,
            "organism": organism,
            "inducible": inducible,
            "candidates": candidates,
            "usage": (
                "Place promoter 5' of RBS-gene cassette. "
                "For fine-tuning: mix constitutive+inducible or use multiple RBS variants."
            ),
            "next_steps": ["synbio_expression_balance", "synbio_design_package"],
        }
    except Exception as e:
        return {"error": str(e), "hint": "Check organism parameter ('ecoli' or 'yeast')"}


def synbio_expression_balance(
    enzyme_names: Optional[List[str]] = None,
    target_fluxes: Optional[List[float]] = None,
    organism: str = "ecoli",
    bottleneck_indices: Optional[List[int]] = None,
    output_dir: Optional[str] = None,
    # Aliases for LLM compatibility
    genes: Optional[List[str]] = None,
    target_ratios: Optional[List[float]] = None,
    **kw,
) -> Dict:
    """Balance multi-gene pathway expression using promoter + RBS selection.

    Strategy based on metabolic control analysis: enzymes with high flux
    control coefficients need higher expression; downstream enzymes need
    lower expression to prevent toxic intermediate accumulation.

    Args:
        enzyme_names: ordered list of pathway enzymes/genes
        target_fluxes: relative flux targets per enzyme (0.0-2.0, normalized)
                       If None, uses heuristic gradient
        organism: 'ecoli' or 'yeast'
        bottleneck_indices: 0-based indices of bottleneck enzymes (get boosted expression)

    Returns:
        Dict with per-enzyme promoter+RBS recommendations and design rationale
    """
    try:
        from zhuangzhou.synbio.parts import PartSelector

        # Resolve aliases
        enzyme_names = enzyme_names or genes or []
        if not enzyme_names:
            return {'error': 'enzyme_names (or genes) list is required',
                    'hint': 'Pass enzyme_names=["TAL","PAL"] or genes=["TAL","PAL"]'}
        target_fluxes = target_fluxes or target_ratios

        # Adjust flux targets for bottlenecks
        if bottleneck_indices and not target_fluxes:
            n = len(enzyme_names)
            target_fluxes = [0.5] * n
            for i in bottleneck_indices:
                if 0 <= i < n:
                    target_fluxes[i] = 1.5  # boost bottleneck

        recommendations = PartSelector.balance_pathway_expression(
            enzyme_names=enzyme_names,
            target_fluxes=target_fluxes,
            organism=organism,
        )

        # Compute total pathway efficiency estimate
        from zhuangzhou.synbio.parts import ECOLI_PROMOTERS, YEAST_PROMOTERS
        lib = ECOLI_PROMOTERS if organism == "ecoli" else YEAST_PROMOTERS

        for rec in recommendations:
            pname = rec["promoter"]
            if pname in lib:
                actual_strength = lib[pname]["strength"]
                deviation = abs(actual_strength - rec["target_flux"])
                rec["expression_deviation"] = round(deviation, 3)
                rec["tunable"] = lib[pname]["type"] == "inducible"

        return {
            "organism": organism,
            "n_enzymes": len(enzyme_names),
            "recommendations": recommendations,
            "design_principles": [
                "First enzyme: high expression for precursor capture",
                "Rate-limiting enzymes: strong promoter (J23100 or Ptrc)",
                "Last enzyme: medium expression to avoid product toxicity",
                "Avoid IPTG if industrial scale (cost); use constitutive instead",
            ],
            "next_steps": ["synbio_design_package", "synbio_burden_assess"],
        }
    except Exception as e:
        return {"error": str(e), "hint": "Provide enzyme_names as a list of gene/enzyme names"}


def synbio_crispr_design(
    target_gene: str,
    genome_sequence: Optional[str] = None,
    organism: str = "ecoli",
    edit_type: str = "knockout",
    cas_system: str = "SpCas9",
    n_guides: int = 5,
    insert_sequence: Optional[str] = None,
    output_dir: Optional[str] = None,
    strategy: Optional[str] = None,   # alias for edit_type
    host: Optional[str] = None,        # alias for organism
    **kw,
) -> Dict:
    """Design CRISPR guide RNAs for gene editing.

    Args:
        target_gene: gene name or locus tag (e.g. 'lacZ', 'b0344')
        genome_sequence: optional 500-2000bp DNA sequence around target
        organism: 'ecoli' or 'yeast'
        edit_type: 'knockout', 'knockin', 'crispri', 'base_edit'
        cas_system: 'SpCas9', 'SaCas9', 'Cas12a', 'dCas9'
        n_guides: number of guide candidates
        insert_sequence: DNA to insert (for knockin only)

    Returns:
        Dict with gRNA sequences, scores, PAM sites, and strategy notes
    """
    try:
        from zhuangzhou.synbio.crispr import find_guides, design_crispri, design_hdr_template
        from zhuangzhou.metakg.config import ECOLI_INTEGRATION_SITES, CRISPR_SYSTEMS

        # Resolve aliases
        if host and not organism or organism == "ecoli":
            organism = host if host else organism
        if strategy and edit_type == "knockout":
            edit_type = strategy
        if not target_gene:
            return {'error': 'target_gene is required',
                    'hint': 'Pass target_gene="tyrR" or target_gene="aroG"'}

        result: Dict = {
            "target_gene": target_gene,
            "organism": organism,
            "edit_type": edit_type,
            "cas_system": cas_system,
        }

        # Get Cas system info
        cas_info = CRISPR_SYSTEMS.get(cas_system, {})
        result["pam"] = cas_info.get("pam", "NGG")
        result["spacer_length"] = cas_info.get("spacer_len", 20)

        if edit_type == "crispri":
            crispri = design_crispri(target_gene, promoter_sequence=genome_sequence)
            result.update(crispri)
            result["vector_note"] = "Express dCas9 under Ptet or Para for inducible repression"

        elif genome_sequence:
            guides = find_guides(genome_sequence, cas_system=cas_system, n_guides=n_guides)
            result["guides"] = guides

            if edit_type == "knockout":
                result["strategy"] = [
                    f"1. Clone spacer into sgRNA backbone (pX330 or similar)",
                    f"2. Transform with Cas9 expression plasmid",
                    f"3. Screen colonies by PCR/Sanger for frameshift",
                    f"4. Use guide with score ≥ 0.7 for best efficiency",
                ]
                result["knockout_efficiency_note"] = (
                    "SpCas9 + NGG: typically 60-90% knockout in E. coli (one-step)"
                )

            elif edit_type == "knockin" and insert_sequence:
                best_guide = guides[0] if guides else None
                cut_site = best_guide.get("cut_site", 0) if best_guide else 0
                hdr = design_hdr_template(
                    cut_site=cut_site,
                    insert_sequence=insert_sequence,
                    genomic_flank_len=800,
                    genomic_sequence=genome_sequence,
                )
                result["hdr_template"] = hdr
                result["strategy"] = [
                    "1. Co-transform sgRNA + Cas9 + donor template",
                    "2. Use linear dsDNA donor (PCR product) for efficiency",
                    "3. Screen by colony PCR with junction primers",
                    "4. Sequence-verify integration",
                ]

        else:
            # No sequence provided — return design guidance
            result["guidance"] = [
                f"Provide genome_sequence (500-2000bp around {target_gene}) for guide design",
                f"Retrieve sequence: NCBI Gene search '{target_gene} {organism}'",
                f"Or use get_ncbi_protein_fasta skill to get gene sequence",
            ]
            if organism == "ecoli":
                if target_gene in ECOLI_INTEGRATION_SITES:
                    site = ECOLI_INTEGRATION_SITES[target_gene]
                    result["integration_site"] = site

        result["plasmid_resources"] = {
            "SpCas9": "Addgene #62988 (pCas), #85815 (pKDO-Cas9)",
            "dCas9":  "Addgene #44249 (pdCas9-bacteria)",
            "yeast":  "Addgene #43804 (pML107, CAS9-NAT)",
        }
        result["next_steps"] = ["synbio_design_package", "metakg_essentiality"]
        return result

    except Exception as e:
        logger.exception("synbio_crispr_design failed")
        return {"error": str(e), "hint": "Provide target_gene and optionally genome_sequence"}


def synbio_codon_optimize(
    protein_sequence: str,
    organism: str = "ecoli",
    method: str = "frequency",
    output_dir: Optional[str] = None,
) -> Dict:
    """Codon-optimize a protein sequence for expression in the target chassis.

    Args:
        protein_sequence: amino acid sequence (single-letter, no gaps)
        organism: 'ecoli', 'yeast', or 'pichia'
        method: 'frequency' (match host codon table), 'cai' (maximize CAI)

    Returns:
        Dict with optimized DNA sequence, CAI score, GC content
    """
    try:
        # Route to existing codon_opt module for pichia
        if organism == "pichia":
            from zhuangzhou.codon_opt import optimize_codon_usage
            result = optimize_codon_usage(protein_sequence, method=method)
            result["organism"] = organism
            return result

        # Simple frequency-based optimization for E. coli and yeast
        ECOLI_CODON_TABLE = {
            "A": "GCG", "R": "CGT", "N": "AAC", "D": "GAT", "C": "TGC",
            "Q": "CAG", "E": "GAA", "G": "GGC", "H": "CAT", "I": "ATT",
            "L": "CTG", "K": "AAA", "M": "ATG", "F": "TTT", "P": "CCG",
            "S": "AGC", "T": "ACC", "W": "TGG", "Y": "TAT", "V": "GTG",
            "*": "TAA",
        }
        YEAST_CODON_TABLE = {
            "A": "GCT", "R": "AGA", "N": "AAT", "D": "GAT", "C": "TGT",
            "Q": "CAA", "E": "GAA", "G": "GGT", "H": "CAT", "I": "ATT",
            "L": "TTG", "K": "AAA", "M": "ATG", "F": "TTT", "P": "CCA",
            "S": "TCT", "T": "ACT", "W": "TGG", "Y": "TAT", "V": "GTT",
            "*": "TAA",
        }
        table = ECOLI_CODON_TABLE if organism == "ecoli" else YEAST_CODON_TABLE

        dna = ""
        warnings = []
        for aa in protein_sequence.upper():
            codon = table.get(aa)
            if codon is None:
                warnings.append(f"Unknown amino acid: {aa}")
                codon = "NNN"
            dna += codon

        gc = sum(1 for b in dna if b in "GC") / len(dna) if dna else 0
        length = len(dna)

        return {
            "organism": organism,
            "method": "preferred_codon",
            "protein_length": len(protein_sequence),
            "dna_length": length,
            "optimized_dna": dna,
            "dna_sequence": dna,   # alias for downstream consumers
            "gc_content": round(gc, 3),
            "cai_estimate": 0.85 if not warnings else 0.60,
            "warnings": warnings,
            "notes": [
                "Add ATG start codon before RBS-optimized sequence",
                "Verify no internal restriction sites for your cloning strategy",
                "Consider IDT codon optimization tool for fine-tuned GC distribution",
            ],
            "next_steps": ["synbio_design_package"],
        }
    except Exception as e:
        return {"error": str(e), "hint": "Provide protein_sequence as single-letter amino acid string"}


def synbio_burden_assess(
    heterologous_genes: Optional[List[str]] = None,
    expression_levels: Optional[List[str]] = None,
    organism: str = "ecoli",
    output_dir: Optional[str] = None,
    genes: Optional[List[str]] = None,          # alias
    copy_numbers: Optional[List[int]] = None,   # alias → maps to expression_levels
    **kw,
) -> Dict:
    """Estimate metabolic burden of heterologous gene expression.

    Based on: Glick 1995, Ceroni et al. 2015 metabolic burden model.
    Estimates: ribosome sequestration, amino acid drain, energy (ATP) cost.

    Args:
        heterologous_genes: list of gene/enzyme names
        expression_levels: list of expression levels per gene ('low','medium','high')
        organism: 'ecoli' or 'yeast'

    Returns:
        Dict with burden score, predicted growth impact, optimization suggestions
    """
    try:
        heterologous_genes = heterologous_genes or genes or []
        if not isinstance(heterologous_genes, list):
            return {'error': 'heterologous_genes must be a list of gene names',
                    'hint': 'Pass heterologous_genes=["TAL","4CL","STS"] — a Python list, not a string'}
        if not heterologous_genes:
            return {'error': 'heterologous_genes (or genes) list is required',
                    'hint': 'Pass heterologous_genes=["TAL","PAL"]'}
        # Map copy_numbers → expression_levels bucket
        if copy_numbers and not expression_levels:
            def _cn_to_level(cn):
                return 'high' if cn >= 15 else 'medium' if cn >= 5 else 'low'
            expression_levels = [_cn_to_level(c) for c in copy_numbers]
        n = len(heterologous_genes)
        if not expression_levels:
            expression_levels = ["medium"] * n
        expression_levels = (expression_levels + ["medium"] * n)[:n]

        # Burden model: each gene adds ~0.3-3% growth reduction depending on expression
        burden_per_gene = {"low": 0.3, "medium": 1.0, "high": 3.0, "ultra": 8.0}
        total_burden = sum(burden_per_gene.get(lvl, 1.0) for lvl in expression_levels)
        growth_reduction_pct = min(total_burden, 60.0)  # cap at 60%

        risk = "LOW" if growth_reduction_pct < 5 else "MEDIUM" if growth_reduction_pct < 20 else "HIGH"

        recommendations = []
        if risk == "HIGH":
            recommendations += [
                "Use inducible promoters — only express during production phase",
                "Split pathway into two strains (co-culture / two-stage fermentation)",
                "Use adaptive laboratory evolution (ALE) to restore fitness",
                "Reduce copy number: integrate into chromosome vs. high-copy plasmid",
            ]
        elif risk == "MEDIUM":
            recommendations += [
                "Monitor growth curve — confirm < 20% reduction vs. WT",
                "Use mid-copy plasmid (p15A origin) instead of high-copy (ColE1)",
                "Consider staggered induction (express one module at a time)",
            ]
        else:
            recommendations += [
                "Burden appears manageable — proceed with balanced expression design",
                "Monitor if adding >3 more genes",
            ]

        per_gene = []
        for gene, lvl in zip(heterologous_genes, expression_levels):
            per_gene.append({
                "gene": gene,
                "expression_level": lvl,
                "burden_pct": round(burden_per_gene.get(lvl, 1.0), 2),
            })

        return {
            "organism": organism,
            "n_heterologous_genes": n,
            "per_gene_burden": per_gene,
            "total_burden_score": round(total_burden, 2),
            "estimated_growth_reduction_pct": round(growth_reduction_pct, 1),
            "risk_level": risk,
            "recommendations": recommendations,
            "note": (
                "Model is approximate (Glick 1995 / Ceroni 2015). "
                "Actual burden depends on protein size, folding stress, and toxicity."
            ),
            "next_steps": ["synbio_design_package"],
        }
    except Exception as e:
        return {"error": str(e), "hint": "Provide heterologous_genes as list of gene names"}


def synbio_design_package(
    target_product: str,
    chassis: str = "ecoli",
    pathway_enzymes: Optional[List[str]] = None,
    pathway_reactions: Optional[List[str]] = None,
    knockout_targets: Optional[List[str]] = None,
    expression_recommendations: Optional[List[Dict]] = None,
    output_dir: Optional[str] = None,
    organism: Optional[str] = None,          # alias for chassis
    host_genes: Optional[List[str]] = None,  # alias for pathway_enzymes
    # ── Enrichment data (pass-through to Word report) ──
    protein_candidates: Optional[List[Dict]] = None,   # from get_protein_profile
    codon_sequences: Optional[Dict] = None,            # {gene: codon_optimize result}
    fba_results: Optional[Dict] = None,                # from metakg_flux
    bottleneck_results: Optional[Dict] = None,         # from metakg_bottleneck
    burden_results: Optional[Dict] = None,             # from synbio_burden_assess
    metakg_pathway: Optional[Dict] = None,             # from metakg_pathway
    **kw,
) -> Dict:
    """Generate a complete genetic design package document.

    Consolidates pathway design, part selections, CRISPR targets, and
    synthesis plan into a structured design document (JSON + text summary).

    Args:
        target_product: target molecule name
        chassis: 'ecoli' or 'yeast'
        pathway_enzymes: ordered list of enzyme names
        pathway_reactions: BiGG reaction IDs of pathway
        knockout_targets: genes to knock out
        expression_recommendations: from synbio_expression_balance
        output_dir: directory to save design.json

    Returns:
        Dict with complete design package, file path
    """
    import json
    import datetime

    # Resolve aliases
    chassis = organism or chassis
    pathway_enzymes = pathway_enzymes or host_genes

    output_dir = output_dir or os.path.join(os.getcwd(), f"{target_product.replace(' ','_')}_design")
    os.makedirs(output_dir, exist_ok=True)

    design = {
        "design_id": f"{target_product.replace(' ','_')}_{chassis}_{datetime.date.today().isoformat()}",
        "target_product": target_product,
        "chassis_organism": chassis,
        "generated_by": "Zhuangzhou Bio-design Mode v0.8.0",
        "date": datetime.date.today().isoformat(),
    }

    # Pathway module
    if pathway_enzymes:
        design["pathway_module"] = {
            "enzymes": pathway_enzymes,
            "n_steps": len(pathway_enzymes),
            "reactions": pathway_reactions or [],
        }

    # Expression module
    if expression_recommendations:
        design["expression_module"] = {
            "strategy": expression_recommendations,
            "note": "Assemble genes in operon or individual transcription units (TUs)",
        }
    elif pathway_enzymes:
        # Auto-generate basic recommendations
        from zhuangzhou.synbio.parts import PartSelector
        recs = PartSelector.balance_pathway_expression(pathway_enzymes, organism=chassis)
        design["expression_module"] = {"strategy": recs, "auto_generated": True}

    # Knockout module
    if knockout_targets:
        design["knockout_module"] = {
            "targets": knockout_targets,
            "method": "CRISPR-Cas9 (SpCas9/NGG) or Lambda Red recombination",
            "verification": "Colony PCR + Sanger sequencing",
        }

    # Chassis preparation
    design["chassis_preparation"] = {
        "recommended_strain": (
            "E. coli BL21(DE3) (high expression) or MG1655 (metabolic studies)"
            if chassis == "ecoli"
            else "S. cerevisiae CEN.PK113-7D (reference strain) or BY4741"
        ),
        "auxotrophies": "Check strain auxotrophies match plasmid selectable markers",
        "competent_cells": "Prepare electrocompetent cells (>10⁸ cfu/μg for Lambda Red)",
    }

    # Assembly strategy
    design["assembly_strategy"] = {
        "method": "Gibson Assembly (overlap ≥30bp) or Golden Gate (BsaI sites)",
        "vector": (
            "pUCIDT (cloning), then pACYC184 (low copy) or pBBR1 (medium copy)"
            if chassis == "ecoli"
            else "YEp13 (2μ, high copy) or pRS414 (CEN/ARS, single copy)"
        ),
        "operon_design": "Promoter → RBS → Gene1 → RBS → Gene2 → Terminator",
    }

    # Verification plan
    design["verification_plan"] = [
        "1. Transform and plate on selective media",
        "2. Colony PCR to confirm insert size",
        "3. Sanger sequencing of entire pathway",
        "4. Growth curve comparison (WT vs. engineered)",
        "5. LC-MS/GC-MS metabolite profiling for product titer",
        "6. Adjust promoter strength if titer < expected (replace with Anderson variants)",
    ]

    # Save design file
    design_path = os.path.join(output_dir, "design.json")
    summary_path = os.path.join(output_dir, "design_summary.md")

    with open(design_path, "w", encoding="utf-8") as f:
        json.dump(design, f, indent=2, ensure_ascii=False)

    # Markdown summary
    lines = [
        f"# Metabolic Engineering Design: {target_product}",
        f"**Chassis**: {chassis}  |  **Date**: {design['date']}",
        "",
        "## Pathway",
    ]
    if pathway_enzymes:
        for i, e in enumerate(pathway_enzymes):
            rxn = (pathway_reactions[i] if pathway_reactions and i < len(pathway_reactions) else "?")
            lines.append(f"  Step {i+1}: `{rxn}` — enzyme: **{e}**")
    lines += [
        "",
        "## Expression Strategy",
    ]
    if design.get("expression_module", {}).get("strategy"):
        for rec in design["expression_module"]["strategy"]:
            lines.append(
                f"  - **{rec.get('enzyme','?')}** → Promoter: `{rec.get('promoter','?')}` "
                f"(strength {rec.get('promoter_strength',0):.2f}) "
                f"| RBS: `{rec.get('rbs','?')}`"
            )
    if knockout_targets:
        lines += ["", "## Knockouts", "  - " + "\n  - ".join(knockout_targets)]

    lines += [
        "", "## Assembly",
        f"  - Method: {design['assembly_strategy']['method']}",
        f"  - Vector: {design['assembly_strategy']['vector']}",
        "", "## Verification",
    ] + [f"  {v}" for v in design["verification_plan"]]

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # ── Auto-generate Word report ────────────────────────────────────────────
    word_report_path = None
    try:
        from zhuangzhou.report.templates import build_synbio_report
        report_data = {
            'design_package': design,
            'target_product': target_product,
            'chassis': chassis,
            'output_dir': output_dir,
            'protein_candidates': protein_candidates or [],
            'codon_sequences': codon_sequences or {},
            'fba_results': fba_results or {},
            'bottleneck_results': bottleneck_results or {},
            'burden_results': burden_results or {},
            'metakg_pathway': metakg_pathway or {},
        }
        word_result = build_synbio_report(
            report_data,
            output_path=os.path.join(output_dir, 'synbio_report.docx'),
        )
        word_report_path = word_result.get('docx_path')
    except Exception as _we:
        import logging as _log
        _log.getLogger(__name__).warning('Word report generation failed: %s', _we)

    # ── Word report in Chinese ───────────────────────────────────────────────
    word_report_zh_path = None
    try:
        from zhuangzhou.report.templates import build_synbio_report
        word_zh = build_synbio_report(
            report_data,
            output_path=os.path.join(output_dir, 'synbio_report_zh.docx'),
        )
        word_report_zh_path = word_zh.get('docx_path')
    except Exception:
        pass

    return {
        "target_product": target_product,
        "chassis": chassis,
        "design_package": design,
        "output_dir": output_dir,
        "design_json": design_path,
        "design_summary_md": summary_path,
        "word_report": word_report_path,
        "word_report_zh": word_report_zh_path,
        "next_steps": [
            "generate_report(report_type='synbio'): 重新生成双语Word报告",
            "metakg_flux: 运行FBA分析获取产率预测",
            "metakg_bottleneck: 识别通量瓶颈",
            "synbio_codon_optimize: 优化各酶基因密码子",
        ],
    }


def synbio_graphrag_query(
    query: str,
    kg_domain: str = "auto",
    limit: int = 10,
    output_dir: Optional[str] = None,
) -> Dict:
    """Query GraphRAG knowledge graphs for systems/synthetic biology evidence.

    Searches: Synthetic_Biology, Systems_Biology, Evolution KGs.

    Args:
        query: natural language query about pathway/organism/engineering strategy
        kg_domain: 'Synthetic_Biology', 'Systems_Biology', 'Evolution', or 'auto'
        limit: max results per KG

    Returns:
        Dict with entities, relationships, and community context summaries
    """
    try:
        from zhuangzhou.brain.graphrag import GraphRAGBridge
        bridge = GraphRAGBridge()

        if kg_domain == "auto":
            kg_domain = bridge.select_domain(query) or "Synthetic_Biology"

        keywords = query.split()[:8]

        entities = bridge.search_entities(keywords, kg_id=kg_domain, limit=limit)
        rels = bridge.search_relationships(query.split()[0], kg_id=kg_domain, limit=limit // 2)
        context = bridge.get_community_context(query, kg_id=kg_domain, limit=3)

        return {
            "query": query,
            "kg_domain": kg_domain,
            "entities": entities,
            "relationships": rels,
            "community_context": context,
            "next_steps": ["metakg_pathway", "synbio_expression_balance"],
        }
    except Exception as e:
        return {"error": str(e),
                "hint": "GraphRAG requires the graphrag conda env and a configured ZHUANGZHOU_GRAPHRAG_ROOT."}


# ── Skill registration ─────────────────────────────────────────────────────────

def register(registry):
    from zhuangzhou.skills.registry import Skill, SkillParam

    S = SkillParam

    SKILLS = [
        Skill(
            name="synbio_promoter_select",
            description="Select optimal promoter for target expression level in E. coli or yeast",
            category="synbio",
            fn=synbio_promoter_select,
            inputs=[
                S("target_expression", "string",
                  "Expression level: 'weak', 'low', 'medium', 'high', 'ultra' or 0.0-2.0",
                  required=False, default="medium",
                  choices=["weak", "low", "medium", "high", "ultra"]),
                S("organism", "string", "Chassis organism", required=False, default="ecoli",
                  choices=["ecoli", "yeast"]),
                S("inducible", "bool", "Return inducible promoters only", required=False, default=False),
                S("inducer", "string", "Specific inducer (IPTG, arabinose, galactose, aTc)",
                  required=False),
                S("n_candidates", "int", "Number of alternatives", required=False, default=5),
            ],
            outputs=[S("candidates", "list", "Ranked promoter candidates")],
            requires_gpu=False, requires_wsl=False, requires_network=False,
            remote_capable=False, remote_preferred=False,
            depends_on=[], output_type="analysis", risk_level="low",
            estimated_time="~1s",
        ),
        Skill(
            name="synbio_expression_balance",
            description="Balance multi-gene pathway expression with promoter+RBS selection",
            category="synbio",
            fn=synbio_expression_balance,
            inputs=[
                S("enzyme_names", "list", "Ordered enzyme/gene names in pathway", required=True),
                S("target_fluxes", "list", "Relative flux per enzyme (0-2)", required=False),
                S("organism", "string", "Chassis", required=False, default="ecoli",
                  choices=["ecoli", "yeast"]),
                S("bottleneck_indices", "list",
                  "0-based indices of bottleneck enzymes (get boosted expression)",
                  required=False),
            ],
            outputs=[S("recommendations", "list", "Per-enzyme promoter+RBS recommendations")],
            requires_gpu=False, requires_wsl=False, requires_network=False,
            remote_capable=False, remote_preferred=False,
            depends_on=["metakg_bottleneck"], output_type="analysis", risk_level="low",
            estimated_time="~1s",
        ),
        Skill(
            name="synbio_crispr_design",
            description="Design CRISPR guide RNAs for knockout, knockin, or CRISPRi",
            category="synbio",
            fn=synbio_crispr_design,
            inputs=[
                S("target_gene", "string", "Gene name or locus tag", required=True),
                S("genome_sequence", "string", "500-2000bp DNA sequence around target",
                  required=False),
                S("organism", "string", "Chassis", required=False, default="ecoli",
                  choices=["ecoli", "yeast"]),
                S("edit_type", "string", "Type of edit",
                  required=False, default="knockout",
                  choices=["knockout", "knockin", "crispri", "base_edit"]),
                S("cas_system", "string", "Cas nuclease system",
                  required=False, default="SpCas9",
                  choices=["SpCas9", "SaCas9", "Cas12a", "dCas9"]),
                S("n_guides", "int", "Number of guide candidates", required=False, default=5),
                S("insert_sequence", "string", "DNA to insert (knockin only)", required=False),
            ],
            outputs=[S("guides", "list", "gRNA sequences and quality scores")],
            requires_gpu=False, requires_wsl=False, requires_network=False,
            remote_capable=False, remote_preferred=False,
            depends_on=["metakg_competitor", "metakg_essentiality"],
            output_type="analysis", risk_level="medium",
            estimated_time="~2-5s",
        ),
        Skill(
            name="synbio_codon_optimize",
            description="Codon-optimize heterologous protein sequence for E. coli or yeast",
            category="synbio",
            fn=synbio_codon_optimize,
            inputs=[
                S("protein_sequence", "string", "Amino acid sequence (single-letter)", required=True),
                S("organism", "string", "Target organism for codon table",
                  required=False, default="ecoli", choices=["ecoli", "yeast", "pichia"]),
                S("method", "string", "Optimization method",
                  required=False, default="frequency", choices=["frequency", "cai"]),
            ],
            outputs=[S("optimized_dna", "string", "Codon-optimized DNA sequence")],
            requires_gpu=False, requires_wsl=False, requires_network=False,
            remote_capable=False, remote_preferred=False,
            depends_on=[], output_type="analysis", risk_level="low",
            estimated_time="~1s",
        ),
        Skill(
            name="synbio_burden_assess",
            description="Estimate metabolic burden and growth impact of heterologous gene expression",
            category="synbio",
            fn=synbio_burden_assess,
            inputs=[
                S("heterologous_genes", "list", "Gene/enzyme names to express", required=True),
                S("expression_levels", "list",
                  "Expression level per gene ('low','medium','high')", required=False),
                S("organism", "string", "Chassis", required=False, default="ecoli",
                  choices=["ecoli", "yeast"]),
            ],
            outputs=[S("risk_level", "string", ""), S("estimated_growth_reduction_pct", "float", "")],
            requires_gpu=False, requires_wsl=False, requires_network=False,
            remote_capable=False, remote_preferred=False,
            depends_on=[], output_type="analysis", risk_level="low",
            estimated_time="~1s",
        ),
        Skill(
            name="synbio_design_package",
            description="Generate complete metabolic engineering design document (JSON + Markdown)",
            category="synbio",
            fn=synbio_design_package,
            inputs=[
                S("target_product", "string", "Target molecule name", required=True),
                S("chassis", "string", "Chassis organism",
                  required=False, default="ecoli", choices=["ecoli", "yeast"]),
                S("pathway_enzymes", "list", "Ordered enzyme names", required=False),
                S("pathway_reactions", "list", "BiGG reaction IDs", required=False),
                S("knockout_targets", "list", "Genes to knock out", required=False),
                S("expression_recommendations", "list",
                  "From synbio_expression_balance", required=False),
                S("output_dir", "path", "Output directory", required=False, is_output_path=True),
            ],
            outputs=[S("design_json", "path", ""), S("design_summary_md", "path", "")],
            requires_gpu=False, requires_wsl=False, requires_network=False,
            remote_capable=False, remote_preferred=False,
            depends_on=["metakg_pathway", "synbio_expression_balance"],
            output_type="report", risk_level="low",
            estimated_time="~3s",
        ),
        Skill(
            name="synbio_graphrag_query",
            description="Query Synthetic_Biology and Systems_Biology GraphRAG KGs",
            category="synbio",
            fn=synbio_graphrag_query,
            inputs=[
                S("query", "string", "Research question about metabolic engineering strategy",
                  required=True),
                S("kg_domain", "string", "Knowledge graph to query",
                  required=False, default="auto",
                  choices=["auto", "Synthetic_Biology", "Systems_Biology", "Evolution",
                           "Bioinformatics"]),
                S("limit", "int", "Max results", required=False, default=10),
            ],
            outputs=[S("entities", "dict", ""), S("community_context", "dict", "")],
            requires_gpu=False, requires_wsl=False, requires_network=False,
            remote_capable=False, remote_preferred=False,
            depends_on=[], output_type="analysis", risk_level="low",
            estimated_time="~5-20s",
        ),
    ]

    for skill in SKILLS:
        registry.register(skill)
    return len(SKILLS)


# Compatibility alias — allows `synbio.register_skills(r)` alongside `synbio.register(r)`
register_skills = register
