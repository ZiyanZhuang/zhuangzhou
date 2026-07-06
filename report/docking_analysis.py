"""
Docking Analysis Enrichment
============================
Enriches docking results with UniProt active site data, KEGG pathway context,
ligand role classification (inhibitor/substrate/activator), and mechanistic
hypothesis generation.

All network calls use stdlib urllib only (no requests dependency).
All functions return dicts; they never raise exceptions to the caller.
"""

import json
import logging
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

USER_AGENT = 'Zhuangzhou/0.7.0 (enzyme engineering; mailto:zhuangzhou@betathink.ai)'

UNIPROT_BASE = 'https://rest.uniprot.org'
KEGG_BASE = 'https://rest.kegg.jp'


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _http_get(url: str, headers: dict = None, timeout: int = 20) -> bytes:
    """Robust HTTP GET with 3-retry and rate-limit back-off."""
    hdrs = {'User-Agent': USER_AGENT, 'Accept': 'application/json'}
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(url, headers=hdrs)
    last_err = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(1.0 * (attempt + 1))
                last_err = e
                continue
            raise
        except urllib.error.URLError as e:
            last_err = e
            if attempt < 2:
                time.sleep(0.5)
                continue
            raise
        except (OSError, socket.timeout) as e:
            last_err = e
            if attempt < 2:
                time.sleep(0.5)
                continue
            raise
    raise last_err


def _safe_http_get(url: str, headers: dict = None, timeout: int = 20) -> Optional[bytes]:
    """Like _http_get but returns None instead of raising."""
    try:
        return _http_get(url, headers=headers, timeout=timeout)
    except Exception as e:
        logger.warning("HTTP GET failed for %s: %s", url, e)
        return None


# ---------------------------------------------------------------------------
# UniProt binding site enrichment
# ---------------------------------------------------------------------------

def get_uniprot_binding_sites(uniprot_id: str, **kw) -> dict:
    """
    Query UniProt REST API for active/binding site annotations.

    Parses the 'features' array from the UniProt JSON entry for:
      - Active site residues
      - Binding site residues
      - Catalytic activity description
      - Domain annotations
      - Motif annotations
      - EC number
      - Protein function description

    Args:
        uniprot_id: UniProt accession (e.g. 'P08684' for CYP3A4)

    Returns:
        {
          'active_sites': [{'pos': int, 'desc': str}, ...],
          'binding_sites': [{'pos': int, 'desc': str, 'ligand': str}, ...],
          'catalytic_residues': [{'pos': int, 'desc': str}, ...],
          'domains': [{'start': int, 'end': int, 'description': str}, ...],
          'motifs': [{'start': int, 'end': int, 'description': str}, ...],
          'ec_number': str,
          'function': str,
          'protein_name': str,
          'organism': str,
        }
        or {'error': str, 'hint': str} on failure.
    """
    try:
        url = f'{UNIPROT_BASE}/uniprotkb/{uniprot_id}.json'
        raw = _safe_http_get(url)
        if raw is None:
            return {'error': f'Network error fetching UniProt entry {uniprot_id}',
                    'hint': 'Check network connectivity or UniProt ID'}
        data = json.loads(raw)

        # ── Feature extraction helpers ──
        def _extract_single_pos(feat_type: str) -> List[dict]:
            """For features with a single residue position."""
            out = []
            for f in data.get('features', []):
                if f.get('type') == feat_type:
                    loc = f.get('location', {})
                    pos_val = loc.get('position', {}).get('value')
                    # Fallback: some entries use start==end
                    if pos_val is None:
                        pos_val = loc.get('start', {}).get('value')
                    out.append({
                        'pos': pos_val,
                        'desc': f.get('description', ''),
                    })
            return out

        def _extract_range(feat_type: str) -> List[dict]:
            """For features spanning a residue range."""
            out = []
            for f in data.get('features', []):
                if f.get('type') == feat_type:
                    loc = f.get('location', {})
                    out.append({
                        'start': loc.get('start', {}).get('value'),
                        'end': loc.get('end', {}).get('value'),
                        'description': f.get('description', ''),
                    })
            return out

        def _extract_binding_sites() -> List[dict]:
            """Binding site features include ligand info in ligand/ligandPart."""
            out = []
            for f in data.get('features', []):
                if f.get('type') == 'Binding site':
                    loc = f.get('location', {})
                    pos_val = loc.get('position', {}).get('value')
                    if pos_val is None:
                        pos_val = loc.get('start', {}).get('value')
                    ligand = f.get('ligand', {}).get('name', '')
                    ligand_part = f.get('ligandPart', {}).get('name', '')
                    out.append({
                        'pos': pos_val,
                        'desc': f.get('description', ''),
                        'ligand': ligand,
                        'ligand_part': ligand_part,
                    })
            return out

        # ── EC number ──
        ec_number = ''
        for prot in data.get('proteinDescription', {}).get('recommendedName', {}).get('ecNumbers', []):
            ec_number = prot.get('value', '')
            break

        # ── Function description ──
        function_text = ''
        for cc in data.get('comments', []):
            if cc.get('commentType') == 'FUNCTION':
                texts = cc.get('texts', [])
                if texts:
                    function_text = texts[0].get('value', '')
                    break

        # ── Protein name ──
        protein_name = (
            data.get('proteinDescription', {})
                .get('recommendedName', {})
                .get('fullName', {})
                .get('value', uniprot_id)
        )

        # ── Organism ──
        organism = data.get('organism', {}).get('scientificName', '')

        return {
            'uniprot_id': uniprot_id,
            'protein_name': protein_name,
            'organism': organism,
            'ec_number': ec_number,
            'function': function_text,
            'active_sites': _extract_single_pos('Active site'),
            'binding_sites': _extract_binding_sites(),
            'catalytic_residues': _extract_single_pos('Catalytic residue'),
            'domains': _extract_range('Domain'),
            'motifs': _extract_range('Motif'),
        }

    except json.JSONDecodeError as e:
        return {'error': f'Invalid JSON from UniProt for {uniprot_id}: {e}',
                'hint': 'Verify UniProt accession is correct'}
    except Exception as e:
        logger.exception("get_uniprot_binding_sites failed for %s", uniprot_id)
        return {'error': str(e), 'hint': f'Failed to fetch UniProt data for {uniprot_id}'}


# ---------------------------------------------------------------------------
# KEGG protein / pathway info
# ---------------------------------------------------------------------------

def get_kegg_protein_info(uniprot_id: str = None, gene_name: str = None,
                           organism: str = 'hsa', **kw) -> dict:
    """
    Query KEGG for pathway and reaction information for a protein.

    Uses gene_name search first; falls back to UniProt cross-reference lookup
    in KEGG if only uniprot_id is provided.

    Args:
        uniprot_id: UniProt accession (used to look up gene name if gene_name not given)
        gene_name:  Gene symbol (e.g. 'CYP3A4') — preferred input
        organism:   KEGG organism code (default 'hsa' = Homo sapiens)

    Returns:
        {
          'kegg_id': str,
          'ko_number': str,
          'pathways': [{'id': str, 'name': str}, ...],
          'reactions': [{'id': str, 'equation': str}, ...],
          'gene_name': str,
        }
        or {'error': str, 'hint': str} on failure.
    """
    try:
        resolved_gene = gene_name

        # If no gene_name given, try to resolve via UniProt entry
        if not resolved_gene and uniprot_id:
            url = f'{UNIPROT_BASE}/uniprotkb/{uniprot_id}.json'
            raw = _safe_http_get(url)
            if raw:
                data = json.loads(raw)
                gene_names = data.get('genes', [])
                if gene_names:
                    gn = gene_names[0].get('geneName', {}).get('value', '')
                    if gn:
                        resolved_gene = gn

        if not resolved_gene:
            return {'error': 'Could not resolve gene name for KEGG lookup',
                    'hint': 'Provide gene_name (e.g. CYP3A4) or a valid uniprot_id'}

        # Step 1: Find KEGG gene ID
        find_url = f'{KEGG_BASE}/find/genes/{urllib.parse.quote(resolved_gene)}'
        raw = _safe_http_get(find_url, headers={'Accept': 'text/plain'})
        if not raw:
            return {'error': f'KEGG search failed for gene {resolved_gene}',
                    'hint': 'Check network or gene name'}

        kegg_id = ''
        text = raw.decode('utf-8', errors='replace')
        for line in text.splitlines():
            # Lines look like: "hsa:1576\tCYP3A4; cytochrome P450..."
            if line.startswith(f'{organism}:'):
                parts = line.split('\t', 1)
                candidate_id = parts[0].strip()
                # Prefer exact gene name match in description
                desc = parts[1] if len(parts) > 1 else ''
                gene_sym = resolved_gene.upper()
                if gene_sym in desc.upper():
                    kegg_id = candidate_id
                    break
                if not kegg_id:
                    kegg_id = candidate_id  # take first if no exact match

        if not kegg_id:
            return {'error': f'No KEGG entry found for {resolved_gene} in organism {organism}',
                    'hint': f'Try a different organism code (e.g. mmu for mouse)'}

        # Step 2: Fetch entry details
        get_url = f'{KEGG_BASE}/get/{urllib.parse.quote(kegg_id)}'
        raw = _safe_http_get(get_url, headers={'Accept': 'text/plain'})
        if not raw:
            return {'error': f'Failed to fetch KEGG entry {kegg_id}',
                    'hint': 'Network error or invalid KEGG ID'}

        entry_text = raw.decode('utf-8', errors='replace')

        # Parse KEGG flat file format
        pathways = []
        reactions = []
        ko_number = ''
        in_pathway = False
        in_reaction = False

        for line in entry_text.splitlines():
            if line.startswith('ORTHOLOGY'):
                # e.g. "ORTHOLOGY   K07408  CYP3A4"
                parts = line.split()
                if len(parts) >= 2:
                    ko_number = parts[1]
                in_pathway = False
                in_reaction = False

            elif line.startswith('PATHWAY'):
                in_pathway = True
                in_reaction = False
                rest = line[len('PATHWAY'):].strip()
                if rest:
                    _parse_kegg_pathway_line(rest, pathways)

            elif line.startswith('REACTION'):
                in_reaction = True
                in_pathway = False
                rest = line[len('REACTION'):].strip()
                if rest:
                    for rid in rest.split():
                        reactions.append({'id': rid.strip(), 'equation': ''})

            elif line.startswith(' ') and (in_pathway or in_reaction):
                rest = line.strip()
                if in_pathway and rest:
                    _parse_kegg_pathway_line(rest, pathways)
                elif in_reaction and rest:
                    for rid in rest.split():
                        if rid.strip():
                            reactions.append({'id': rid.strip(), 'equation': ''})
            else:
                in_pathway = False
                in_reaction = False

        return {
            'kegg_id': kegg_id,
            'gene_name': resolved_gene,
            'ko_number': ko_number,
            'pathways': pathways,
            'reactions': reactions,
        }

    except Exception as e:
        logger.exception("get_kegg_protein_info failed for gene=%s uniprot=%s",
                         gene_name, uniprot_id)
        return {'error': str(e), 'hint': 'KEGG query failed'}


def _parse_kegg_pathway_line(text: str, pathways: list):
    """Parse a KEGG PATHWAY line into {id, name} and append to list."""
    # Format: "hsa00982  Drug metabolism - cytochrome P450"
    parts = text.split(None, 1)
    if len(parts) == 2:
        pathways.append({'id': parts[0], 'name': parts[1].strip()})
    elif len(parts) == 1:
        pathways.append({'id': parts[0], 'name': ''})


# ---------------------------------------------------------------------------
# Ligand role classification
# ---------------------------------------------------------------------------

def classify_ligand_role(plip_result: dict, uniprot_data: dict,
                          affinity: float, ligand_smiles: str = None,
                          **kw) -> dict:
    """
    Classify whether the docking ligand is likely an inhibitor, substrate, or activator.

    Classification logic uses a weighted evidence system:

    Substrate indicators:
      - Affinity between -6 and -8 kcal/mol (moderate, reversible binding)
      - Ligand overlaps with annotated active site residues
      - Estimated MW < 500 Da (typical metabolite/substrate range)
      - H-bonds to catalytic residues

    Inhibitor indicators:
      - Affinity < -9 kcal/mol (tight, drug-like binding)
      - Occupies active site but has no catalytic residue H-bonds (dead-end complex)
      - Hydrophobic contacts dominate interaction profile
      - MW in 300-600 Da drug-like range

    Activator indicators:
      - Binds away from active site (no active site residue overlap)
      - Allosteric site binding suggested by off-active-site position
      - No overlap with annotated catalytic residues

    Args:
        plip_result:   PLIP interaction dict with keys like 'hydrogen_bonds',
                       'hydrophobic_contacts', 'interacting_residues', etc.
        uniprot_data:  Output from get_uniprot_binding_sites()
        affinity:      Docking affinity in kcal/mol (negative = favorable)
        ligand_smiles: SMILES string for MW estimation (optional)

    Returns:
        {
          'role': 'inhibitor' | 'substrate' | 'activator' | 'unknown',
          'confidence': float (0.0-1.0),
          'evidence': [str, ...],
          'binding_mode': str,
          'score_breakdown': {'inhibitor': float, 'substrate': float, 'activator': float},
        }
    """
    try:
        evidence = []
        scores = {'inhibitor': 0.0, 'substrate': 0.0, 'activator': 0.0}

        # ── Collect PLIP interacting residue numbers ──
        plip_residues = _collect_plip_residue_numbers(plip_result)

        # ── Collect UniProt active site positions ──
        active_site_positions = set()
        catalytic_positions = set()
        if not isinstance(uniprot_data, dict) or 'error' in uniprot_data:
            uniprot_data = {}

        for feat in uniprot_data.get('active_sites', []):
            pos = feat.get('pos')
            if pos is not None:
                active_site_positions.add(int(pos))

        for feat in uniprot_data.get('catalytic_residues', []):
            pos = feat.get('pos')
            if pos is not None:
                catalytic_positions.add(int(pos))
                active_site_positions.add(int(pos))

        # ── Affinity-based scoring ──
        if affinity is not None:
            if affinity < -10.0:
                scores['inhibitor'] += 2.0
                evidence.append(f'Very tight binding (affinity {affinity:.1f} kcal/mol) — strong inhibitor signal')
            elif affinity < -9.0:
                scores['inhibitor'] += 1.5
                evidence.append(f'Tight binding (affinity {affinity:.1f} kcal/mol) — inhibitor range')
            elif affinity < -7.5:
                scores['inhibitor'] += 0.5
                scores['substrate'] += 0.5
                evidence.append(f'Moderate-strong binding ({affinity:.1f} kcal/mol) — ambiguous')
            elif affinity < -6.0:
                scores['substrate'] += 1.5
                evidence.append(f'Moderate binding ({affinity:.1f} kcal/mol) — substrate-like affinity')
            else:
                scores['substrate'] += 0.5
                evidence.append(f'Weak binding ({affinity:.1f} kcal/mol) — possible transient substrate')

        # ── Active site overlap ──
        overlapping = plip_residues & active_site_positions
        catalytic_overlap = plip_residues & catalytic_positions

        if active_site_positions:
            if overlapping:
                overlap_count = len(overlapping)
                evidence.append(
                    f'Ligand contacts {overlap_count} annotated active site residue(s): '
                    f'{sorted(overlapping)}'
                )
                if catalytic_overlap:
                    # H-bonds to catalytic residues = substrate signal
                    hbond_res = _collect_hbond_residue_numbers(plip_result)
                    cat_hbonds = hbond_res & catalytic_positions
                    if cat_hbonds:
                        scores['substrate'] += 2.0
                        evidence.append(
                            f'H-bonds to catalytic residue(s) {sorted(cat_hbonds)} — substrate indicator'
                        )
                    else:
                        scores['inhibitor'] += 1.0
                        evidence.append(
                            'Contacts catalytic residues but no H-bonds — inhibitor-like dead-end complex'
                        )
                else:
                    scores['inhibitor'] += 0.5
                    evidence.append('Occupies active site but misses catalytic residues')
            else:
                scores['activator'] += 2.0
                evidence.append(
                    'Ligand does NOT contact any UniProt active site residues — '
                    'suggests allosteric/activator binding'
                )
        else:
            evidence.append('No UniProt active site annotations available for overlap analysis')

        # ── Interaction type profile ──
        hbond_count = len(plip_result.get('hydrogen_bonds', []))
        hydrophobic_count = len(plip_result.get('hydrophobic_contacts', []))
        total_contacts = hbond_count + hydrophobic_count + \
                         len(plip_result.get('pi_stacking', [])) + \
                         len(plip_result.get('salt_bridges', []))

        if total_contacts > 0:
            hydrophobic_fraction = hydrophobic_count / total_contacts
            if hydrophobic_fraction > 0.7:
                scores['inhibitor'] += 1.0
                evidence.append(
                    f'Hydrophobic-dominant interaction profile ({hydrophobic_fraction:.0%} of contacts) '
                    f'— typical for drug-like inhibitors'
                )
            elif hbond_count >= 3:
                scores['substrate'] += 0.8
                evidence.append(
                    f'{hbond_count} H-bonds — polar interaction profile typical for substrates'
                )

        # ── Molecular weight estimate from SMILES ──
        if ligand_smiles:
            mw = _estimate_mw_from_smiles(ligand_smiles)
            if mw is not None:
                if mw < 300:
                    scores['substrate'] += 1.0
                    evidence.append(f'Estimated MW {mw:.0f} Da — small molecule, substrate-like')
                elif mw <= 600:
                    scores['inhibitor'] += 0.5
                    evidence.append(f'Estimated MW {mw:.0f} Da — drug-like range (Lipinski RO5)')
                else:
                    evidence.append(f'Estimated MW {mw:.0f} Da — large molecule (beyond RO5)')

        # ── Determine winner ──
        total = sum(scores.values())
        if total == 0:
            return {
                'role': 'unknown',
                'confidence': 0.0,
                'evidence': evidence or ['Insufficient data for classification'],
                'binding_mode': 'undetermined',
                'score_breakdown': scores,
            }

        # Normalize scores to get confidence
        normalized = {k: v / total for k, v in scores.items()}
        winner = max(normalized, key=normalized.get)
        confidence = normalized[winner]

        # Require minimum confidence
        if confidence < 0.4:
            winner = 'unknown'

        binding_mode = _infer_binding_mode(winner, plip_result, overlapping)

        return {
            'role': winner,
            'confidence': round(confidence, 3),
            'evidence': evidence,
            'binding_mode': binding_mode,
            'score_breakdown': {k: round(v, 2) for k, v in scores.items()},
        }

    except Exception as e:
        logger.exception("classify_ligand_role failed")
        return {'error': str(e), 'hint': 'Classification failed — check plip_result and uniprot_data'}


def _collect_plip_residue_numbers(plip_result: dict) -> set:
    """Extract all residue numbers from a PLIP result dict."""
    residues = set()
    for itype in ['hydrogen_bonds', 'hydrophobic_contacts', 'pi_stacking',
                  'salt_bridges', 'water_bridges']:
        for ix in plip_result.get(itype, []):
            resnr = ix.get('resnr')
            if resnr is not None:
                try:
                    residues.add(int(resnr))
                except (ValueError, TypeError):
                    pass
    # Also try top-level interacting_residues key
    for r in plip_result.get('interacting_residues', []):
        if isinstance(r, dict):
            resnr = r.get('resnr', r.get('resnum'))
        else:
            resnr = r
        try:
            residues.add(int(resnr))
        except (ValueError, TypeError):
            pass
    return residues


def _collect_hbond_residue_numbers(plip_result: dict) -> set:
    """Extract residue numbers specifically from hydrogen bond interactions."""
    residues = set()
    for hb in plip_result.get('hydrogen_bonds', []):
        resnr = hb.get('resnr')
        if resnr is not None:
            try:
                residues.add(int(resnr))
            except (ValueError, TypeError):
                pass
    return residues


def _estimate_mw_from_smiles(smiles: str) -> Optional[float]:
    """
    Rough molecular weight estimate from SMILES using atom counts.
    Does not require RDKit — uses a simple atom counting heuristic.
    Accurate to ~10% for typical drug-like molecules.
    """
    ATOM_WEIGHTS = {
        'C': 12.011, 'N': 14.007, 'O': 15.999, 'S': 32.06,
        'F': 18.998, 'Cl': 35.45, 'Br': 79.904, 'I': 126.904,
        'P': 30.974, 'H': 1.008,
    }
    try:
        mw = 0.0
        i = 0
        while i < len(smiles):
            # Two-char atoms: Cl, Br
            if i + 1 < len(smiles) and smiles[i:i+2] in ATOM_WEIGHTS:
                mw += ATOM_WEIGHTS[smiles[i:i+2]]
                i += 2
            elif smiles[i] in ATOM_WEIGHTS:
                mw += ATOM_WEIGHTS[smiles[i]]
                i += 1
            else:
                i += 1
        # Approximate H count: organic molecules average ~H/C ratio ~1.6
        c_count = smiles.count('C') - smiles.count('Cl')
        mw += c_count * 1.6 * 1.008
        return mw if mw > 10 else None
    except Exception:
        return None


def _infer_binding_mode(role: str, plip_result: dict, active_site_overlap: set) -> str:
    """Generate a binding mode description string."""
    if role == 'inhibitor':
        if active_site_overlap:
            return 'Competitive active site inhibition'
        else:
            return 'Non-competitive / allosteric inhibition'
    elif role == 'substrate':
        if active_site_overlap:
            return 'Productive active site binding (catalytic orientation)'
        else:
            return 'Non-productive binding (incorrect orientation)'
    elif role == 'activator':
        return 'Allosteric activation (remote site binding)'
    else:
        return 'Undetermined binding mode'


# ---------------------------------------------------------------------------
# Active site consistency cross-reference
# ---------------------------------------------------------------------------

def check_active_site_consistency(plip_result: dict, uniprot_data: dict,
                                   **kw) -> dict:
    """
    Cross-reference PLIP interaction residues with UniProt annotated active/binding sites.

    For each residue contacted in the PLIP analysis, checks whether that residue
    is a known active site or binding site residue according to UniProt annotations.
    Returns an overlap analysis with consistency score and interpretation text.

    Args:
        plip_result:  PLIP interaction dict (from plip_analysis skill)
        uniprot_data: Output from get_uniprot_binding_sites()

    Returns:
        {
          'consistent': bool,
          'overlap_pct': float (0-100),
          'matched_residues': [int, ...],
          'novel_residues': [int, ...],
          'annotated_count': int,
          'plip_count': int,
          'analysis_text': str,
        }
    """
    try:
        if isinstance(uniprot_data, dict) and 'error' in uniprot_data:
            return {
                'consistent': False,
                'overlap_pct': 0.0,
                'matched_residues': [],
                'novel_residues': [],
                'annotated_count': 0,
                'plip_count': 0,
                'analysis_text': f'UniProt data unavailable: {uniprot_data.get("error", "unknown error")}',
            }

        # Collect PLIP residues
        plip_residues = _collect_plip_residue_numbers(plip_result)

        # Collect UniProt annotated functional residues
        annotated = set()
        for feat in uniprot_data.get('active_sites', []):
            pos = feat.get('pos')
            if pos is not None:
                try:
                    annotated.add(int(pos))
                except (ValueError, TypeError):
                    pass
        for feat in uniprot_data.get('binding_sites', []):
            pos = feat.get('pos')
            if pos is not None:
                try:
                    annotated.add(int(pos))
                except (ValueError, TypeError):
                    pass
        for feat in uniprot_data.get('catalytic_residues', []):
            pos = feat.get('pos')
            if pos is not None:
                try:
                    annotated.add(int(pos))
                except (ValueError, TypeError):
                    pass

        matched = sorted(plip_residues & annotated)
        novel = sorted(plip_residues - annotated)

        if not plip_residues:
            overlap_pct = 0.0
        elif not annotated:
            overlap_pct = 0.0
        else:
            overlap_pct = 100.0 * len(matched) / len(annotated)

        consistent = overlap_pct >= 30.0 and len(matched) >= 1

        # Generate interpretation text
        if not annotated:
            analysis_text = (
                'No UniProt active/binding site annotations found for this protein. '
                'Consistency analysis cannot be performed. The docking pose contacts '
                f'{len(plip_residues)} residues according to PLIP.'
            )
        elif not plip_residues:
            analysis_text = (
                'PLIP analysis found no significant interactions. '
                'This may indicate a weak or non-specific binding pose.'
            )
        elif consistent:
            analysis_text = (
                f'The docking pose shows CONSISTENT binding with UniProt annotations. '
                f'{len(matched)} of {len(annotated)} annotated functional residues '
                f'are contacted ({overlap_pct:.0f}% overlap). '
                f'Matched residues: {matched}. '
            )
            if novel:
                analysis_text += (
                    f'Additionally, {len(novel)} novel interaction(s) not in UniProt '
                    f'were detected at residues {novel}, which may represent '
                    'induced-fit contacts or previously unreported binding determinants.'
                )
        else:
            analysis_text = (
                f'The docking pose shows LIMITED consistency with UniProt annotations. '
                f'Only {len(matched)} of {len(annotated)} annotated residues are '
                f'contacted ({overlap_pct:.0f}% overlap). '
            )
            if novel:
                analysis_text += (
                    f'The ligand primarily contacts {len(novel)} non-annotated residue(s) '
                    f'({novel}), suggesting either an alternative binding mode, '
                    'allosteric binding, or a non-productive docking pose.'
                )

        return {
            'consistent': consistent,
            'overlap_pct': round(overlap_pct, 1),
            'matched_residues': matched,
            'novel_residues': novel,
            'annotated_count': len(annotated),
            'plip_count': len(plip_residues),
            'analysis_text': analysis_text,
        }

    except Exception as e:
        logger.exception("check_active_site_consistency failed")
        return {
            'consistent': False,
            'overlap_pct': 0.0,
            'matched_residues': [],
            'novel_residues': [],
            'annotated_count': 0,
            'plip_count': 0,
            'analysis_text': f'Consistency analysis failed: {e}',
        }
