"""
Biological Database API Skills — Online protein/gene/structure retrieval.

Provides comprehensive access to major bioinformatics databases:
  - UniProt (enhanced): sequence, domains, GO terms, subcellular location, pathway
  - NCBI Protein/Gene (Entrez E-utilities): protein records, gene info, FASTA
  - EMBL-EBI InterPro: protein domains, families, signatures
  - RCSB PDB: 3D structure search, metadata, ligands
  - AlphaFold DB: predicted structure availability

These skills feed downstream pipeline steps (docking, MD, mutagenesis)
with precise protein context — sequence, structure, domains, catalytic info.
"""

import json
import logging
import os
import re
import time
import socket
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

from .registry import Skill, SkillParam, SkillRegistry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# API Base URLs
# ---------------------------------------------------------------------------
UNIPROT_BASE = "https://rest.uniprot.org"
NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
INTERPRO_BASE = "https://www.ebi.ac.uk/interpro/api"
PDB_SEARCH_BASE = "https://search.rcsb.org/rcsbsearch/v2/query"
PDB_DATA_BASE = "https://data.rcsb.org/rest/v1/core"
ALPHAFOLD_BASE = "https://alphafold.ebi.ac.uk/api"

# NCBI rate limit: 3 req/s without API key
_last_ncbi_call = 0.0
NCBI_API_KEY = os.environ.get('NCBI_API_KEY', '')

USER_AGENT = 'Zhuangzhou/0.7.0 (enzyme engineering; mailto:zhuangzhou@betathink.ai)'


def _http_get(url: str, headers: dict = None, timeout: int = 20) -> bytes:
    """Robust HTTP GET with retry and error-classified backoff.

    Uses utils.retry error classification to decide retry eligibility and
    appropriate backoff delays (longer for rate limits, shorter for transient).
    """
    from zhuangzhou.utils.retry import (
        classify_error, exponential_backoff, rate_limit_backoff, ErrorKind,
    )
    from zhuangzhou.utils.robustness import sanitize_error_html

    hdrs = {'User-Agent': USER_AGENT, 'Accept': 'application/json'}
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(url, headers=hdrs)
    last_err = None
    max_attempts = 3

    for attempt in range(max_attempts):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as e:
            last_err = e
            kind = classify_error(e)

            # Don't retry auth or permanent errors
            if not kind.retryable:
                # Sanitize HTML error pages from API responses
                if isinstance(e, urllib.error.HTTPError):
                    try:
                        body = e.read().decode('utf-8', errors='replace')[:2000]
                        clean = sanitize_error_html(body)
                        if clean != body:
                            logger.debug("HTTP %d: %s", e.code, clean)
                    except Exception:
                        pass
                raise

            if attempt >= max_attempts - 1:
                raise

            # Error-classified backoff
            if kind == ErrorKind.RATE_LIMIT:
                delay = rate_limit_backoff(attempt)
            else:
                delay = exponential_backoff(attempt, base_delay=0.5)

            logger.debug("HTTP %s error on attempt %d/%d, retry in %.1fs: %s",
                         kind.value, attempt + 1, max_attempts, delay, e)
            time.sleep(delay)

    raise last_err


_ncbi_lock = __import__('threading').Lock()


def _ncbi_throttle():
    """Enforce NCBI 3 req/s rate limit (thread-safe)."""
    global _last_ncbi_call
    with _ncbi_lock:
        elapsed = time.time() - _last_ncbi_call
        if elapsed < 0.34:
            time.sleep(0.34 - elapsed)
        _last_ncbi_call = time.time()


# ===========================================================================
# UniProt Enhanced API
# ===========================================================================

def get_uniprot_sequence(accession: str, **kw) -> dict:
    """
    Get protein FASTA sequence from UniProt by accession.

    Returns sequence string, length, molecular weight, and checksum.
    """
    try:
        url = f"{UNIPROT_BASE}/uniprotkb/{accession}.json"
        data = json.loads(_http_get(url))
        seq_info = data.get('sequence', {})
        return {
            'accession': data.get('primaryAccession', accession),
            'sequence': seq_info.get('value', ''),
            'length': seq_info.get('length', 0),
            'mol_weight': seq_info.get('molWeight', 0),
            'crc64': seq_info.get('crc64', ''),
        }
    except Exception as e:
        return {'error': str(e), 'hint': f'Failed to fetch sequence for {accession}'}


def get_uniprot_domains(accession: str, **kw) -> dict:
    """
    Get protein domain annotations from UniProt.

    Returns domains, regions, motifs, repeats, zinc fingers, signal peptides,
    transmembrane regions, and topology information.
    """
    try:
        url = f"{UNIPROT_BASE}/uniprotkb/{accession}.json"
        params = urllib.parse.urlencode({
            'fields': ','.join([
                'accession', 'protein_name', 'ft_domain', 'ft_region',
                'ft_motif', 'ft_repeat', 'ft_zn_fing', 'ft_signal',
                'ft_transmem', 'ft_topo_dom', 'ft_chain', 'ft_propep',
                'cc_domain', 'xref_pfam', 'xref_interpro',
            ]),
        })
        data = json.loads(_http_get(f"{url}?{params}"))

        def extract_ft(feat_type):
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

        # Extract cross-references for Pfam and InterPro
        pfam_refs = []
        interpro_refs = []
        for xref in data.get('uniProtKBCrossReferences', []):
            db = xref.get('database', '')
            if db == 'Pfam':
                props = {p.get('key', ''): p.get('value', '') for p in xref.get('properties', [])}
                pfam_refs.append({
                    'id': xref.get('id', ''),
                    'name': props.get('EntryName', ''),
                })
            elif db == 'InterPro':
                props = {p.get('key', ''): p.get('value', '') for p in xref.get('properties', [])}
                interpro_refs.append({
                    'id': xref.get('id', ''),
                    'name': props.get('EntryName', ''),
                })

        return {
            'accession': data.get('primaryAccession', accession),
            'domains': extract_ft('Domain'),
            'regions': extract_ft('Region'),
            'motifs': extract_ft('Motif'),
            'repeats': extract_ft('Repeat'),
            'zinc_fingers': extract_ft('Zinc finger'),
            'signal_peptide': extract_ft('Signal peptide'),
            'transmembrane': extract_ft('Transmembrane'),
            'topological_domain': extract_ft('Topological domain'),
            'chain': extract_ft('Chain'),
            'propeptide': extract_ft('Propeptide'),
            'pfam': pfam_refs,
            'interpro': interpro_refs,
        }
    except Exception as e:
        return {'error': str(e), 'hint': f'Failed to fetch domains for {accession}'}


def get_uniprot_full(accession: str, **kw) -> dict:
    """
    Get comprehensive UniProt entry: sequence, function, catalytic activity,
    domains, GO terms, subcellular location, pathway, cofactors, kinetics,
    PDB cross-refs, EC number, gene names, and taxonomy.

    This is the primary protein info retrieval tool — use it to gather
    all context needed for downstream analysis.
    """
    try:
        url = f"{UNIPROT_BASE}/uniprotkb/{accession}.json"
        data = json.loads(_http_get(url))

        # Basic info
        seq = data.get('sequence', {})
        org = data.get('organism', {})
        genes = [g.get('geneName', {}).get('value', '')
                 for g in data.get('genes', []) if g.get('geneName')]

        # Extract protein name
        pn = data.get('proteinDescription', {})
        rec = pn.get('recommendedName', {})
        name = rec.get('fullName', {}).get('value', '') if rec else ''
        if not name:
            subs = pn.get('submissionNames', [])
            name = subs[0].get('fullName', {}).get('value', '') if subs else ''
        ec_numbers = [ec.get('value', '') for ec in rec.get('ecNumbers', [])] if rec else []

        # Comments extraction
        comments = data.get('comments', [])
        result = {
            'accession': data.get('primaryAccession', accession),
            'name': name,
            'gene_names': genes,
            'ec_numbers': ec_numbers,
            'organism': org.get('scientificName', ''),
            'taxonomy_id': org.get('taxonId', 0),
            'lineage': [t if isinstance(t, str) else t.get('scientificName', '')
                       for t in org.get('lineage', [])][:5],
            'sequence': seq.get('value', ''),
            'length': seq.get('length', 0),
            'mol_weight': seq.get('molWeight', 0),
        }

        # Parse comments by type
        for c in comments:
            ct = c.get('commentType', '')
            if ct == 'FUNCTION':
                texts = c.get('texts', [])
                result['function'] = texts[0].get('value', '')[:800] if texts else ''
            elif ct == 'CATALYTIC ACTIVITY':
                if 'catalytic_activity' not in result:
                    result['catalytic_activity'] = []
                rxn = c.get('reaction', {})
                result['catalytic_activity'].append({
                    'name': rxn.get('name', ''),
                    'ec': rxn.get('ecNumber', ''),
                    'evidence': [e.get('evidenceCode', '') for e in rxn.get('evidences', [])][:3],
                })
            elif ct == 'COFACTOR':
                if 'cofactors' not in result:
                    result['cofactors'] = []
                for cof in c.get('cofactors', []):
                    result['cofactors'].append({
                        'name': cof.get('name', ''),
                        'db_id': cof.get('cofactorCrossReference', {}).get('id', ''),
                    })
            elif ct == 'SUBCELLULAR LOCATION':
                locs = []
                for sl in c.get('subcellularLocations', []):
                    loc = sl.get('location', {})
                    locs.append(loc.get('value', ''))
                result['subcellular_location'] = locs
            elif ct == 'PATHWAY':
                texts = c.get('texts', [])
                if 'pathway' not in result:
                    result['pathway'] = []
                for t in texts:
                    result['pathway'].append(t.get('value', ''))
            elif ct == 'BIOPHYSICOCHEMICAL PROPERTIES':
                bp = {}
                ki = c.get('kineticParameters', {})
                if ki:
                    bp['km'] = [{'value': m.get('constant', ''),
                                 'substrate': m.get('substrate', '')}
                                for m in ki.get('michaelisConstants', [])]
                    bp['vmax'] = [v.get('constant', '') for v in ki.get('maximumVelocities', [])]
                ph = c.get('phDependence', {})
                if ph:
                    bp['ph_dependence'] = ph.get('texts', [{}])[0].get('value', '') if ph.get('texts') else ''
                temp = c.get('temperatureDependence', {})
                if temp:
                    bp['temp_dependence'] = temp.get('texts', [{}])[0].get('value', '') if temp.get('texts') else ''
                if bp:
                    result['kinetics'] = bp

        # Features
        features = data.get('features', [])
        feat_map = {}
        for f in features:
            ft = f.get('type', '')
            if ft in ('Active site', 'Binding site', 'Metal binding',
                       'Disulfide bond', 'Modified residue', 'Mutagenesis',
                       'Site', 'Glycosylation'):
                if ft not in feat_map:
                    feat_map[ft] = []
                loc = f.get('location', {})
                feat_map[ft].append({
                    'start': loc.get('start', {}).get('value'),
                    'end': loc.get('end', {}).get('value'),
                    'description': f.get('description', ''),
                })
        result['features'] = feat_map

        # GO annotations
        go_terms = []
        for xref in data.get('uniProtKBCrossReferences', []):
            if xref.get('database') == 'GO':
                props = {p.get('key', ''): p.get('value', '') for p in xref.get('properties', [])}
                go_terms.append({
                    'id': xref.get('id', ''),
                    'term': props.get('GoTerm', ''),
                    'source': props.get('GoEvidenceType', ''),
                })
        result['go_terms'] = go_terms[:30]

        # PDB cross-refs
        pdb_refs = []
        for xref in data.get('uniProtKBCrossReferences', []):
            if xref.get('database') == 'PDB':
                props = {p.get('key', ''): p.get('value', '') for p in xref.get('properties', [])}
                pdb_refs.append({
                    'id': xref.get('id', ''),
                    'method': props.get('Method', ''),
                    'resolution': props.get('Resolution', ''),
                    'chains': props.get('Chains', ''),
                })
        result['pdb_structures'] = pdb_refs[:20]

        # Pfam / InterPro
        pfam = []
        for xref in data.get('uniProtKBCrossReferences', []):
            if xref.get('database') == 'Pfam':
                props = {p.get('key', ''): p.get('value', '') for p in xref.get('properties', [])}
                pfam.append({'id': xref.get('id', ''), 'name': props.get('EntryName', '')})
        result['pfam'] = pfam

        return result
    except Exception as e:
        return {'error': str(e), 'hint': f'Failed to fetch full entry for {accession}'}


# ===========================================================================
# NCBI Entrez API (Protein + Gene)
# ===========================================================================

def search_ncbi_protein(query: str, organism: str = None,
                        max_results: int = 5, **kw) -> dict:
    """
    Search NCBI Protein database. Returns protein records with
    accession, description, organism, and length.
    """
    try:
        search_query = query
        if organism:
            search_query += f" AND {organism}[Organism]"

        # Phase 1: esearch
        _ncbi_throttle()
        params = urllib.parse.urlencode({
            'db': 'protein',
            'term': search_query,
            'retmax': max_results,
            'retmode': 'json',
            'usehistory': 'y',
            **(({'api_key': NCBI_API_KEY} if NCBI_API_KEY else {})),
        })
        data = json.loads(_http_get(f"{NCBI_BASE}/esearch.fcgi?{params}"))
        result = data.get('esearchresult', {})
        ids = result.get('idlist', [])
        if not ids:
            return {'query': query, 'results': [], 'count': 0}

        # Phase 2: esummary
        _ncbi_throttle()
        params2 = urllib.parse.urlencode({
            'db': 'protein',
            'id': ','.join(ids),
            'retmode': 'json',
            **(({'api_key': NCBI_API_KEY} if NCBI_API_KEY else {})),
        })
        summ = json.loads(_http_get(f"{NCBI_BASE}/esummary.fcgi?{params2}"))
        summ_result = summ.get('result', {})

        proteins = []
        for uid in ids:
            entry = summ_result.get(uid, {})
            if not isinstance(entry, dict):
                continue
            proteins.append({
                'uid': uid,
                'accession': entry.get('caption', ''),
                'title': entry.get('title', ''),
                'organism': entry.get('organism', ''),
                'length': entry.get('slen', 0),
                'mol_type': entry.get('moltype', ''),
                'source_db': entry.get('sourcedb', ''),
            })

        return {
            'query': query,
            'results': proteins,
            'count': int(result.get('count', len(proteins))),
        }
    except Exception as e:
        return {'error': str(e), 'hint': 'NCBI Protein search failed'}


def get_ncbi_protein_fasta(accession: str, **kw) -> dict:
    """
    Fetch protein FASTA sequence from NCBI by accession/GI.

    Returns header, sequence, and length.
    """
    try:
        _ncbi_throttle()
        params = urllib.parse.urlencode({
            'db': 'protein',
            'id': accession,
            'rettype': 'fasta',
            'retmode': 'text',
            **(({'api_key': NCBI_API_KEY} if NCBI_API_KEY else {})),
        })
        raw = _http_get(f"{NCBI_BASE}/efetch.fcgi?{params}").decode('utf-8')
        lines = raw.strip().split('\n')
        if not lines:
            return {'error': 'Empty FASTA response'}
        if not lines[0].startswith('>'):
            return {'error': 'Invalid FASTA response', 'hint': f'Check accession: {accession}'}
        header = lines[0][1:].strip()
        sequence = ''.join(l.strip() for l in lines[1:])
        return {
            'accession': accession,
            'header': header,
            'sequence': sequence,
            'length': len(sequence),
        }
    except Exception as e:
        return {'error': str(e), 'hint': f'Failed to fetch FASTA for {accession}'}


def search_ncbi_gene(query: str, organism: str = None,
                     max_results: int = 5, **kw) -> dict:
    """
    Search NCBI Gene database. Returns gene records with
    gene ID, symbol, description, chromosome location, and organism.
    """
    try:
        search_query = query
        if organism:
            search_query += f" AND {organism}[Organism]"

        _ncbi_throttle()
        params = urllib.parse.urlencode({
            'db': 'gene',
            'term': search_query,
            'retmax': max_results,
            'retmode': 'json',
            **(({'api_key': NCBI_API_KEY} if NCBI_API_KEY else {})),
        })
        data = json.loads(_http_get(f"{NCBI_BASE}/esearch.fcgi?{params}"))
        result = data.get('esearchresult', {})
        ids = result.get('idlist', [])
        if not ids:
            return {'query': query, 'results': [], 'count': 0}

        _ncbi_throttle()
        params2 = urllib.parse.urlencode({
            'db': 'gene',
            'id': ','.join(ids),
            'retmode': 'json',
            **(({'api_key': NCBI_API_KEY} if NCBI_API_KEY else {})),
        })
        summ = json.loads(_http_get(f"{NCBI_BASE}/esummary.fcgi?{params2}"))
        summ_result = summ.get('result', {})

        genes = []
        for uid in ids:
            entry = summ_result.get(uid, {})
            if not isinstance(entry, dict):
                continue
            genes.append({
                'gene_id': uid,
                'symbol': entry.get('name', ''),
                'description': entry.get('description', ''),
                'organism': entry.get('organism', {}).get('scientificname', '')
                    if isinstance(entry.get('organism'), dict)
                    else entry.get('organism', ''),
                'chromosome': entry.get('chromosome', ''),
                'map_location': entry.get('maplocation', ''),
                'aliases': entry.get('otheraliases', ''),
                'summary': entry.get('summary', '')[:500],
            })

        return {
            'query': query,
            'results': genes,
            'count': int(result.get('count', len(genes))),
        }
    except Exception as e:
        return {'error': str(e), 'hint': 'NCBI Gene search failed'}


# ===========================================================================
# EMBL-EBI InterPro API (Protein Domains & Families)
# ===========================================================================

def get_interpro_domains(accession: str, **kw) -> dict:
    """
    Get InterPro domain annotations for a UniProt accession.

    Returns domain/family/homologous superfamily entries with
    member databases (Pfam, PROSITE, PANTHER, CDD, etc.), positions,
    and GO terms.
    """
    try:
        url = f"{INTERPRO_BASE}/entry/interpro/protein/uniprot/{accession}"
        params = urllib.parse.urlencode({'page_size': 50})
        data = json.loads(_http_get(f"{url}?{params}", timeout=25))

        domains = []
        for item in data.get('results', []):
            meta = item.get('metadata', {})
            entry = {
                'interpro_id': meta.get('accession', ''),
                'name': meta.get('name', ''),
                'type': meta.get('type', ''),  # domain, family, homologous_superfamily, etc.
                'description': '',
                'go_terms': [],
                'member_databases': [],
                'locations': [],
            }

            # GO terms from InterPro
            for go in (meta.get('go_terms') or []):
                cat = go.get('category')
                entry['go_terms'].append({
                    'id': go.get('identifier', ''),
                    'name': go.get('name', ''),
                    'category': cat.get('name', '') if isinstance(cat, dict) else str(cat or ''),
                })

            # Member databases
            for db_name, db_entries in (meta.get('member_databases') or {}).items():
                if not isinstance(db_entries, dict):
                    continue
                for db_acc, db_info in db_entries.items():
                    entry['member_databases'].append({
                        'database': db_name,
                        'accession': db_acc,
                        'name': db_info if isinstance(db_info, str) else str(db_info),
                    })

            # Protein locations (where this domain appears on target protein)
            proteins = item.get('proteins') or []
            for prot in proteins:
                for loc_frag in prot.get('entry_protein_locations', []):
                    for frag in loc_frag.get('fragments', []):
                        entry['locations'].append({
                            'start': frag.get('start'),
                            'end': frag.get('end'),
                        })

            domains.append(entry)

        return {
            'accession': accession,
            'domains': domains,
            'count': len(domains),
        }
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return {'accession': accession, 'domains': [], 'count': 0,
                    'note': 'No InterPro entries found for this accession'}
        return {'error': str(e), 'hint': f'InterPro query failed for {accession}'}
    except Exception as e:
        return {'error': str(e), 'hint': f'InterPro query failed for {accession}'}


# ===========================================================================
# RCSB PDB API (3D Structures)
# ===========================================================================

def search_pdb_structure(query: str, entity_type: str = 'protein',
                         max_results: int = 5, **kw) -> dict:
    """
    Search RCSB PDB for 3D structures by text query.

    Returns PDB IDs, titles, experimental method, resolution, organism.
    """
    try:
        # RCSB Search API v2 (JSON body POST)
        search_body = {
            "query": {
                "type": "terminal",
                "service": "full_text",
                "parameters": {"value": query}
            },
            "return_type": "entry",
            "request_options": {
                "paginate": {"start": 0, "rows": max_results},
                "scoring_strategy": "combined",
                "sort": [{"sort_by": "score", "direction": "desc"}]
            }
        }
        req = urllib.request.Request(
            PDB_SEARCH_BASE,
            data=json.dumps(search_body).encode('utf-8'),
            headers={
                'User-Agent': USER_AGENT,
                'Content-Type': 'application/json',
                'Accept': 'application/json',
            },
            method='POST',
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw = resp.read()
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return {'error': 'PDB search returned non-JSON response',
                    'hint': 'RCSB API may be temporarily unavailable. Try again later.'}

        pdb_ids = [hit.get('identifier', '') for hit in data.get('result_set', [])]
        if not pdb_ids:
            return {'query': query, 'results': [], 'count': 0}

        # Fetch metadata for each PDB
        structures = []
        for pdb_id in pdb_ids[:max_results]:
            try:
                meta = json.loads(_http_get(f"{PDB_DATA_BASE}/entry/{pdb_id}"))
                struct = meta.get('struct', {})
                cell = meta.get('cell', {})
                exptl = meta.get('exptl', [{}])[0] if meta.get('exptl') else {}
                rcsb = meta.get('rcsb_entry_info', {})
                structures.append({
                    'pdb_id': pdb_id,
                    'title': struct.get('title', ''),
                    'method': exptl.get('method', ''),
                    'resolution': rcsb.get('resolution_combined', [None])[0]
                        if rcsb.get('resolution_combined') else None,
                    'deposit_date': meta.get('rcsb_accession_info', {}).get('deposit_date', ''),
                    'polymer_count': rcsb.get('polymer_entity_count', 0),
                    'nonpolymer_count': rcsb.get('nonpolymer_entity_count', 0),
                })
            except Exception:
                structures.append({'pdb_id': pdb_id, 'error': 'metadata fetch failed'})

        return {
            'query': query,
            'results': structures,
            'count': data.get('total_count', len(structures)),
        }
    except Exception as e:
        return {'error': str(e), 'hint': 'PDB structure search failed'}


def get_pdb_entry(pdb_id: str, **kw) -> dict:
    """
    Get detailed PDB entry metadata: title, method, resolution,
    polymer entities (chains, sequence), ligands, source organism.
    """
    try:
        pdb_id = pdb_id.upper().strip()
        meta = json.loads(_http_get(f"{PDB_DATA_BASE}/entry/{pdb_id}"))
        struct = meta.get('struct', {})
        exptl = meta.get('exptl', [{}])[0] if meta.get('exptl') else {}
        rcsb = meta.get('rcsb_entry_info', {})

        result = {
            'pdb_id': pdb_id,
            'title': struct.get('title', ''),
            'method': exptl.get('method', ''),
            'resolution': rcsb.get('resolution_combined', [None])[0]
                if rcsb.get('resolution_combined') else None,
            'deposit_date': meta.get('rcsb_accession_info', {}).get('deposit_date', ''),
            'release_date': meta.get('rcsb_accession_info', {}).get('initial_release_date', ''),
        }

        # Fetch polymer entities (protein chains)
        try:
            entities_data = json.loads(_http_get(
                f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/1"))
            entity_poly = entities_data.get('entity_poly', {})
            rcsb_poly = entities_data.get('rcsb_polymer_entity', {})
            src = entities_data.get('rcsb_entity_source_organism', [{}])
            result['entity'] = {
                'type': entity_poly.get('type', ''),
                'sequence': entity_poly.get('pdbx_seq_one_letter_code_can', ''),
                'length': len(entity_poly.get('pdbx_seq_one_letter_code_can', '')),
                'organism': src[0].get('scientific_name', '') if src else '',
                'uniprot_ids': [],
            }
            # Get UniProt mapping
            container = entities_data.get('rcsb_polymer_entity_container_identifiers', {})
            uniprot_ids = container.get('uniprot_ids', [])
            result['entity']['uniprot_ids'] = uniprot_ids or []
        except Exception:
            pass

        # Fetch nonpolymer entities (ligands)
        try:
            ligands = []
            np_count = rcsb.get('nonpolymer_entity_count', 0)
            for i in range(1, min(np_count + 1, 11)):  # max 10 ligands
                try:
                    lig_data = json.loads(_http_get(
                        f"https://data.rcsb.org/rest/v1/core/nonpolymer_entity/{pdb_id}/{i}"))
                    np_comp = lig_data.get('pdbx_entity_nonpoly', {})
                    rcsb_np = lig_data.get('rcsb_nonpolymer_entity', {})
                    ligands.append({
                        'comp_id': np_comp.get('comp_id', ''),
                        'name': np_comp.get('name', ''),
                        'formula': rcsb_np.get('formula_weight', ''),
                    })
                except Exception:
                    break
            result['ligands'] = ligands
        except Exception:
            pass

        return result
    except Exception as e:
        return {'error': str(e), 'hint': f'Failed to fetch PDB entry {pdb_id}'}


# ===========================================================================
# AlphaFold DB API
# ===========================================================================

def get_alphafold_prediction(uniprot_id: str, **kw) -> dict:
    """
    Check AlphaFold DB for predicted structure by UniProt accession.

    Returns model URL, pLDDT confidence, and CIF/PDB download links.
    """
    try:
        url = f"{ALPHAFOLD_BASE}/prediction/{uniprot_id}"
        data = json.loads(_http_get(url))
        if isinstance(data, list) and data:
            data = data[0]

        return {
            'uniprot_id': uniprot_id,
            'model_available': True,
            'entry_id': data.get('entryId', ''),
            'gene': data.get('gene', ''),
            'organism': data.get('organismScientificName', ''),
            'model_url': data.get('modelUrl', ''),  # mmCIF
            'pdb_url': data.get('pdbUrl', ''),       # PDB format
            'pae_image_url': data.get('paeImageUrl', ''),
            'confidence_url': data.get('cifUrl', ''),
            'model_version': data.get('latestVersion', ''),
            'global_plddt': data.get('globalMetricValue', None),
        }
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return {
                'uniprot_id': uniprot_id,
                'model_available': False,
                'note': 'No AlphaFold prediction available for this UniProt ID',
            }
        return {'error': str(e), 'hint': f'AlphaFold query failed for {uniprot_id}'}
    except Exception as e:
        return {'error': str(e), 'hint': f'AlphaFold query failed for {uniprot_id}'}


# ===========================================================================
# Composite: Protein Profile (one-shot comprehensive retrieval)
# ===========================================================================

def get_protein_profile(identifier: str = '', **kw) -> dict:
    """
    One-shot comprehensive protein profile retrieval.

    Given a UniProt accession, retrieves ALL available information:
    - UniProt full annotation (sequence, function, domains, catalytic activity)
    - InterPro domain analysis
    - PDB structures (experimental)
    - AlphaFold prediction availability

    This is the recommended entry point for gathering protein context
    before running docking, MD, mutagenesis, or other computational pipelines.
    """
    # Accept uniprot_id= or accession= as keyword aliases
    identifier = identifier or kw.pop('uniprot_id', '') or kw.pop('accession', '')
    if not identifier:
        return {'error': 'identifier (UniProt accession) is required',
                'hint': 'Pass identifier="Q3IWM5" or uniprot_id="Q3IWM5"'}

    profile = {'identifier': identifier, 'sources': {}}

    # 1. UniProt full entry
    uniprot = get_uniprot_full(identifier)
    if not uniprot.get('error'):
        profile['sources']['uniprot'] = 'ok'
        profile['name'] = uniprot.get('name', '')
        profile['organism'] = uniprot.get('organism', '')
        profile['gene_names'] = uniprot.get('gene_names', [])
        profile['ec_numbers'] = uniprot.get('ec_numbers', [])
        profile['sequence'] = uniprot.get('sequence', '')
        profile['length'] = uniprot.get('length', 0)
        profile['function'] = uniprot.get('function', '')
        profile['catalytic_activity'] = uniprot.get('catalytic_activity', [])
        profile['cofactors'] = uniprot.get('cofactors', [])
        profile['kinetics'] = uniprot.get('kinetics', {})
        profile['subcellular_location'] = uniprot.get('subcellular_location', [])
        profile['pathway'] = uniprot.get('pathway', [])
        profile['features'] = uniprot.get('features', {})
        profile['go_terms'] = uniprot.get('go_terms', [])
        profile['pdb_structures'] = uniprot.get('pdb_structures', [])
        profile['pfam'] = uniprot.get('pfam', [])
    else:
        profile['sources']['uniprot'] = uniprot.get('error', 'failed')

    # 2. InterPro domains
    interpro = get_interpro_domains(identifier)
    if not interpro.get('error'):
        profile['sources']['interpro'] = 'ok'
        profile['interpro_domains'] = interpro.get('domains', [])
    else:
        profile['sources']['interpro'] = interpro.get('error', 'failed')

    # 3. AlphaFold
    af = get_alphafold_prediction(identifier)
    if not af.get('error'):
        profile['sources']['alphafold'] = 'ok'
        profile['alphafold'] = {
            'available': af.get('model_available', False),
            'pdb_url': af.get('pdb_url', ''),
            'global_plddt': af.get('global_plddt'),
        }
    else:
        profile['sources']['alphafold'] = af.get('error', 'failed')

    # 4. Best PDB structure (if available from UniProt xrefs)
    pdb_list = profile.get('pdb_structures', [])
    if pdb_list:
        best = pdb_list[0]
        pdb_detail = get_pdb_entry(best['id'])
        if not pdb_detail.get('error'):
            profile['sources']['pdb'] = 'ok'
            profile['best_pdb'] = pdb_detail
        else:
            profile['sources']['pdb'] = pdb_detail.get('error', 'failed')

    return profile


# ===========================================================================
# PubChem Compound API — Chemical Molecule Lookup
# ===========================================================================

PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"


def search_pubchem_compound(name: str = None, **kw) -> dict:
    """
    Search PubChem for a compound by name and return authoritative SMILES,
    molecular formula, molecular weight, InChI, synonyms, and 3D SDF URL.

    This is the ONLY reliable way to resolve a compound name (e.g. "asiatic acid",
    "ursolic acid", "camphor") to a correct SMILES string. Never let the LLM
    guess SMILES — always call this function first.

    Args:
        name: Compound name in English (e.g. "asiatic acid", "ursolic acid", "caffeine")
              Also accepts keyword argument `query` as an alias for `name`.

    Returns:
        dict with: cid, name, canonical_smiles, isomeric_smiles, molecular_formula,
                   molecular_weight, inchi, inchikey, synonyms, sdf_3d_url
    """
    # Accept `query` as alias for `name` (agents may call with query='nicotine')
    if not name:
        name = kw.get('query', '') or kw.get('compound', '') or ''
    if not name or not name.strip():
        return {'error': 'Compound name is empty', 'hint': 'Provide a chemical compound name'}

    name_clean = name.strip()

    # Step 1: Resolve name → CID
    try:
        url = (f"{PUBCHEM_BASE}/compound/name/"
               f"{urllib.parse.quote(name_clean, safe='')}/cids/JSON")
        data = json.loads(_http_get(url, timeout=15))
        cids = data.get('IdentifierList', {}).get('CID', [])
        if not cids:
            return {'error': f'Compound "{name_clean}" not found in PubChem',
                    'hint': 'Try the English chemical name, IUPAC name, or common name'}
        cid = cids[0]
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return {'error': f'Compound "{name_clean}" not found in PubChem',
                    'hint': 'Try the English chemical name, IUPAC name, or common name. '
                            'Examples: "asiatic acid", "ursolic acid", "camphor"'}
        return {'error': f'PubChem API error: {e}'}
    except Exception as e:
        return {'error': f'PubChem lookup failed: {e}',
                'hint': 'Check network connectivity'}

    # Step 2: Get compound properties
    try:
        props = ('CanonicalSMILES,IsomericSMILES,MolecularFormula,'
                 'MolecularWeight,InChI,InChIKey,IUPACName')
        url2 = (f"{PUBCHEM_BASE}/compound/cid/{cid}/property/"
                f"{props}/JSON")
        data2 = json.loads(_http_get(url2, timeout=15))
        prop_list = data2.get('PropertyTable', {}).get('Properties', [{}])
        prop = prop_list[0] if prop_list else {}
    except Exception as e:
        logger.warning("PubChem property fetch failed: %s", e)
        prop = {}

    # Step 3: Get synonyms (first 10)
    synonyms = []
    try:
        url3 = f"{PUBCHEM_BASE}/compound/cid/{cid}/synonyms/JSON"
        data3 = json.loads(_http_get(url3, timeout=10))
        syn_list = data3.get('InformationList', {}).get('Information', [{}])
        synonyms = (syn_list[0].get('Synonym', []) if syn_list else [])[:10]
    except Exception:
        pass

    # PubChem API returns 'CanonicalSMILES' as 'ConnectivitySMILES' or 'SMILES'
    # and 'IsomericSMILES' as 'SMILES' in the JSON response
    canonical = (prop.get('CanonicalSMILES') or prop.get('ConnectivitySMILES')
                 or prop.get('SMILES', ''))
    isomeric = (prop.get('IsomericSMILES') or prop.get('SMILES', ''))

    result = {
        'cid': cid,
        'name': name_clean,
        'iupac_name': prop.get('IUPACName', ''),
        'canonical_smiles': canonical,
        'isomeric_smiles': isomeric,
        'molecular_formula': prop.get('MolecularFormula', ''),
        'molecular_weight': prop.get('MolecularWeight', ''),
        'inchi': prop.get('InChI', ''),
        'inchikey': prop.get('InChIKey', ''),
        'synonyms': synonyms,
        'pubchem_url': f'https://pubchem.ncbi.nlm.nih.gov/compound/{cid}',
        'sdf_3d_url': f'{PUBCHEM_BASE}/compound/cid/{cid}/SDF?record_type=3d',
        'sdf_2d_url': f'{PUBCHEM_BASE}/compound/cid/{cid}/SDF',
    }

    return result


def download_pubchem_sdf(name: str = None, cid: int = None, smiles: str = None,
                          output_dir: str = None, prefer_3d: bool = True, **kw) -> dict:
    """
    Download a compound's SDF structure file from PubChem.

    This provides AUTHORITATIVE 3D structures — more reliable than LLM-generated
    SMILES + RDKit conformer generation for well-known compounds.

    Args:
        name: Compound name (e.g. "asiatic acid") — auto-resolves via PubChem
        cid: PubChem CID (alternative to name)
        smiles: SMILES string (alternative — searches PubChem for matching compound)
        output_dir: Output directory (default: CWD)
        prefer_3d: Try 3D SDF first, fall back to 2D if unavailable

    Returns:
        dict with: sdf_path, cid, canonical_smiles, molecular_formula, molecular_weight
    """
    # Resolve to CID
    if not cid:
        if name:
            lookup = search_pubchem_compound(name)
            if 'error' in lookup:
                return lookup
            cid = lookup['cid']
        elif smiles:
            try:
                url = (f"{PUBCHEM_BASE}/compound/smiles/"
                       f"{urllib.parse.quote(smiles, safe='')}/cids/JSON")
                data = json.loads(_http_get(url, timeout=15))
                cids = data.get('IdentifierList', {}).get('CID', [])
                if not cids:
                    return {'error': f'SMILES not found in PubChem: {smiles}',
                            'hint': 'The SMILES may be invalid or not in PubChem database'}
                cid = cids[0]
            except Exception as e:
                return {'error': f'PubChem SMILES lookup failed: {e}'}
        else:
            return {'error': 'Provide name, cid, or smiles',
                    'hint': 'e.g. name="asiatic acid" or cid=119034'}

    if output_dir is None:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    # Get properties for metadata
    props = {}
    try:
        purl = (f"{PUBCHEM_BASE}/compound/cid/{cid}/property/"
                f"CanonicalSMILES,IsomericSMILES,MolecularFormula,MolecularWeight/JSON")
        pdata = json.loads(_http_get(purl, timeout=10))
        plist = pdata.get('PropertyTable', {}).get('Properties', [{}])
        props = plist[0] if plist else {}
    except Exception:
        pass

    safe_name = re.sub(r'[^\w\-]', '_', name or f'CID{cid}')
    sdf_path = os.path.join(output_dir, f'{safe_name}.sdf')

    # Try 3D first, then 2D
    downloaded_3d = False
    if prefer_3d:
        try:
            url_3d = f"{PUBCHEM_BASE}/compound/cid/{cid}/SDF?record_type=3d"
            sdf_data = _http_get(url_3d, timeout=20)
            if len(sdf_data) > 100:
                with open(sdf_path, 'wb') as f:
                    f.write(sdf_data)
                downloaded_3d = True
                logger.info("Downloaded 3D SDF from PubChem: CID %d → %s", cid, sdf_path)
        except Exception as e:
            logger.debug("3D SDF not available for CID %d: %s", cid, e)

    if not downloaded_3d:
        try:
            url_2d = f"{PUBCHEM_BASE}/compound/cid/{cid}/SDF"
            sdf_data = _http_get(url_2d, timeout=20)
            if len(sdf_data) > 100:
                with open(sdf_path, 'wb') as f:
                    f.write(sdf_data)
                logger.info("Downloaded 2D SDF from PubChem: CID %d → %s", cid, sdf_path)
            else:
                return {'error': f'SDF download returned empty data for CID {cid}'}
        except Exception as e:
            return {'error': f'Failed to download SDF for CID {cid}: {e}'}

    return {
        'sdf_path': sdf_path,
        'cid': cid,
        'name': name or f'CID{cid}',
        'is_3d': downloaded_3d,
        'canonical_smiles': (props.get('CanonicalSMILES') or props.get('ConnectivitySMILES')
                             or props.get('SMILES', '')),
        'isomeric_smiles': props.get('IsomericSMILES') or props.get('SMILES', ''),
        'molecular_formula': props.get('MolecularFormula', ''),
        'molecular_weight': props.get('MolecularWeight', ''),
        'output_dir': output_dir,
        'next_steps': ['smart_docking (use sdf_path as ligand)', 'plip_analysis', 'conformer_generate'],
    }


def validate_compound_smiles(name: str, smiles: str, **kw) -> dict:
    """
    Validate a SMILES string against PubChem's authoritative data.

    Cross-checks the provided SMILES against PubChem to detect LLM hallucinations.
    Returns the correct canonical SMILES from PubChem if the input is wrong.

    Args:
        name: Expected compound name (e.g. "asiatic acid")
        smiles: SMILES string to validate

    Returns:
        dict with: valid (bool), canonical_smiles, molecular_formula, discrepancy (if any)
    """
    # Step 1: Validate SMILES syntax locally
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {'valid': False, 'error': f'SMILES syntax invalid: {smiles}',
                    'hint': 'Use search_pubchem_compound to get the correct SMILES'}
        local_formula = rdMolDescriptors.CalcMolFormula(mol)
        local_mw = round(Descriptors.MolWt(mol), 2)
        local_canonical = Chem.MolToSmiles(mol)
    except ImportError:
        local_formula = None
        local_mw = None
        local_canonical = smiles

    # Step 2: Look up PubChem authoritative data
    pubchem = search_pubchem_compound(name)
    if 'error' in pubchem:
        # Can't validate against PubChem, return local validation only
        return {
            'valid': True,
            'validated_against': 'local_rdkit_only',
            'canonical_smiles': local_canonical,
            'molecular_formula': local_formula,
            'molecular_weight': local_mw,
            'warning': f'Cannot cross-validate with PubChem: {pubchem["error"]}',
        }

    pubchem_smiles = pubchem.get('canonical_smiles', '')
    pubchem_isomeric = pubchem.get('isomeric_smiles', '')
    pubchem_formula = pubchem.get('molecular_formula', '')
    pubchem_mw = pubchem.get('molecular_weight', '')

    # Step 3: Compare — check formula first (catches major errors),
    # then compare canonical SMILES (catches stereochemistry errors)
    formula_match = (local_formula == pubchem_formula) if local_formula else None

    # Canonicalize PubChem SMILES through RDKit for fair comparison
    # (PubChem canonical != RDKit canonical notation)
    pubchem_canonical_rdkit = pubchem_smiles
    try:
        from rdkit import Chem as _Chem
        _pm = _Chem.MolFromSmiles(pubchem_smiles)
        if _pm:
            pubchem_canonical_rdkit = _Chem.MolToSmiles(_pm)
    except Exception:
        pass

    is_match = formula_match is True and (local_canonical == pubchem_canonical_rdkit)

    result = {
        'valid': is_match,
        'input_smiles': smiles,
        'input_canonical': local_canonical,
        'input_formula': local_formula,
        'input_mw': local_mw,
        'pubchem_smiles': pubchem_smiles,
        'pubchem_formula': pubchem_formula,
        'pubchem_mw': pubchem_mw,
        'pubchem_cid': pubchem.get('cid'),
        'formula_match': formula_match,
    }

    if not is_match:
        # Use isomeric SMILES (with stereochemistry) as the correct one
        correct = pubchem_isomeric or pubchem_smiles
        result['discrepancy'] = (
            f'SMILES MISMATCH for "{name}": '
            f'input={local_canonical} (MW={local_mw}, {local_formula}) vs '
            f'PubChem={pubchem_smiles} (MW={pubchem_mw}, {pubchem_formula}). '
            f'USE THE PubChem SMILES.'
        )
        result['correct_smiles'] = correct
        result['hint'] = 'The provided SMILES is WRONG. Use correct_smiles instead.'

    return result


# ===========================================================================
# Registration
# ===========================================================================

def download_pdb_file(pdb_id: str, output_dir: str = None, **kw) -> dict:
    """Download PDB structure file from RCSB to local disk."""
    pdb_id = pdb_id.upper().strip()
    if not re.match(r'^[A-Z0-9]{4}$', pdb_id):
        return {'error': f'Invalid PDB ID format: "{pdb_id}"',
                'hint': 'PDB ID must be exactly 4 alphanumeric characters (e.g. 1Z10, 2CPP)'}

    if output_dir is None:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f'{pdb_id}.pdb')

    # Skip if already downloaded
    if os.path.exists(output_path) and os.path.getsize(output_path) > 500:
        logger.info("PDB already exists: %s", output_path)
        return {'pdb_path': output_path, 'pdb_id': pdb_id, 'cached': True,
                'size_bytes': os.path.getsize(output_path)}

    url = f'https://files.rcsb.org/download/{pdb_id}.pdb'
    try:
        urllib.request.urlretrieve(url, output_path)
        if os.path.exists(output_path) and os.path.getsize(output_path) > 100:
            size = os.path.getsize(output_path)
            logger.info("Downloaded PDB %s: %s (%d bytes)", pdb_id, output_path, size)
            return {'pdb_path': output_path, 'pdb_id': pdb_id, 'cached': False,
                    'size_bytes': size}
        else:
            return {'error': f'Downloaded file is too small or empty: {output_path}',
                    'hint': f'PDB ID "{pdb_id}" may not exist in RCSB'}
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return {'error': f'PDB ID "{pdb_id}" not found in RCSB',
                    'hint': 'Check the PDB ID or search with search_pdb_structure'}
        return {'error': f'HTTP error downloading PDB: {e}',
                'hint': 'Check network connectivity'}
    except Exception as e:
        return {'error': f'Failed to download PDB {pdb_id}: {e}',
                'hint': 'Check network connectivity'}


def register_skills(registry: SkillRegistry):

    # --- UniProt Enhanced ---
    registry.register(Skill(
        name='get_uniprot_sequence',
        description='Get protein FASTA sequence from UniProt by accession. '
                    'Returns sequence, length, molecular weight.',
        category='research',
        fn=get_uniprot_sequence,
        requires_network=True,
        estimated_time='~2s',
        inputs=[
            SkillParam('accession', 'string', 'UniProt accession (e.g. P00183, Q9Y6K9)'),
        ],
        outputs=[SkillParam('result', 'dict', 'Sequence and metadata')],
        source='UniProt REST API',
    ))

    registry.register(Skill(
        name='get_uniprot_domains',
        description='Get protein domain/region/motif annotations from UniProt. '
                    'Returns domains, Pfam/InterPro cross-refs, signal peptides, '
                    'transmembrane regions.',
        category='research',
        fn=get_uniprot_domains,
        requires_network=True,
        estimated_time='~2s',
        inputs=[
            SkillParam('accession', 'string', 'UniProt accession'),
        ],
        outputs=[SkillParam('result', 'dict', 'Domain annotations')],
        source='UniProt REST API',
    ))

    registry.register(Skill(
        name='get_uniprot_full',
        description='Get COMPREHENSIVE UniProt entry: sequence, function, '
                    'catalytic activity, EC number, domains, GO terms, '
                    'subcellular location, pathway, cofactors, kinetics, '
                    'PDB cross-refs, gene names, taxonomy. '
                    'PRIMARY protein info retrieval tool.',
        category='research',
        fn=get_uniprot_full,
        requires_network=True,
        estimated_time='~3s',
        inputs=[
            SkillParam('accession', 'string', 'UniProt accession (e.g. P00183)'),
        ],
        outputs=[SkillParam('result', 'dict', 'Full protein annotation')],
        source='UniProt REST API',
    ))

    # --- NCBI ---
    registry.register(Skill(
        name='search_ncbi_protein',
        description='Search NCBI Protein database by keyword. Returns accessions, '
                    'descriptions, organisms. Use to find protein records when '
                    'UniProt accession is unknown.',
        category='research',
        fn=search_ncbi_protein,
        requires_network=True,
        estimated_time='~3s',
        inputs=[
            SkillParam('query', 'string', 'Search query (gene name, protein, organism)'),
            SkillParam('organism', 'string', 'Organism filter', required=False),
            SkillParam('max_results', 'int', 'Max results', required=False, default=5),
        ],
        outputs=[SkillParam('results', 'list', 'Protein records')],
        source='NCBI Entrez E-utilities',
    ))

    registry.register(Skill(
        name='get_ncbi_protein_fasta',
        description='Fetch protein FASTA sequence from NCBI by accession or GI number. '
                    'Use when you need raw sequence from RefSeq/GenBank.',
        category='research',
        fn=get_ncbi_protein_fasta,
        requires_network=True,
        estimated_time='~2s',
        inputs=[
            SkillParam('accession', 'string', 'NCBI protein accession or GI'),
        ],
        outputs=[SkillParam('result', 'dict', 'FASTA header and sequence')],
        source='NCBI Entrez E-utilities',
    ))

    registry.register(Skill(
        name='search_ncbi_gene',
        description='Search NCBI Gene database. Returns gene ID, symbol, description, '
                    'chromosome location, summary. Use for gene-level context.',
        category='research',
        fn=search_ncbi_gene,
        requires_network=True,
        estimated_time='~3s',
        inputs=[
            SkillParam('query', 'string', 'Gene name or keyword'),
            SkillParam('organism', 'string', 'Organism filter', required=False),
            SkillParam('max_results', 'int', 'Max results', required=False, default=5),
        ],
        outputs=[SkillParam('results', 'list', 'Gene records')],
        source='NCBI Entrez E-utilities',
    ))

    # --- EMBL-EBI InterPro ---
    registry.register(Skill(
        name='get_interpro_domains',
        description='Get InterPro domain/family annotations for a protein. '
                    'Returns domain boundaries, member databases (Pfam, PROSITE, '
                    'PANTHER, CDD), GO terms. Use for domain architecture analysis.',
        category='research',
        fn=get_interpro_domains,
        requires_network=True,
        estimated_time='~3s',
        inputs=[
            SkillParam('accession', 'string', 'UniProt accession'),
        ],
        outputs=[SkillParam('result', 'dict', 'InterPro domain annotations')],
        source='EMBL-EBI InterPro REST API',
    ))

    # --- RCSB PDB ---
    registry.register(Skill(
        name='search_pdb_structure',
        description='Search RCSB PDB for 3D protein structures by text query. '
                    'Returns PDB IDs, titles, method, resolution. Use to find '
                    'experimental structures for docking/MD.',
        category='research',
        fn=search_pdb_structure,
        requires_network=True,
        estimated_time='~5s',
        inputs=[
            SkillParam('query', 'string', 'Search text (protein name, organism, etc.)'),
            SkillParam('max_results', 'int', 'Max results', required=False, default=5),
        ],
        outputs=[SkillParam('results', 'list', 'PDB structure entries')],
        source='RCSB PDB Search API v2',
    ))

    registry.register(Skill(
        name='get_pdb_entry',
        description='Get detailed PDB entry metadata: title, method, resolution, '
                    'polymer entities (chains, sequence, UniProt mapping), ligands. '
                    'Use after finding PDB ID to get structure details.',
        category='research',
        fn=get_pdb_entry,
        requires_network=True,
        estimated_time='~5s',
        inputs=[
            SkillParam('pdb_id', 'string', 'PDB ID (e.g. 2CPP, 1W0F)'),
        ],
        outputs=[SkillParam('result', 'dict', 'PDB entry metadata')],
        source='RCSB PDB Data API',
    ))

    # --- PDB File Download ---
    registry.register(Skill(
        name='download_pdb',
        description='Download PDB 3D structure file from RCSB to local disk. '
                    'REQUIRED before docking/MD — other PDB skills only return metadata. '
                    'Returns local file path for use with smart_docking, gromacs_md, etc.',
        category='structure',
        fn=download_pdb_file,
        requires_network=True,
        estimated_time='~3s',
        inputs=[
            SkillParam('pdb_id', 'string', 'PDB ID (e.g. 1Z10, 2CPP)'),
            SkillParam('output_dir', 'string', 'Output directory (default: CWD)',
                       required=False),
        ],
        outputs=[SkillParam('pdb_path', 'string', 'Local path to downloaded PDB file')],
        source='RCSB PDB',
        output_type='pdb',
    ))

    # --- AlphaFold DB ---
    registry.register(Skill(
        name='get_alphafold_prediction',
        description='Check if AlphaFold predicted structure exists for a UniProt '
                    'accession. Returns model URL, pLDDT confidence, download links. '
                    'Use when no experimental structure available.',
        category='research',
        fn=get_alphafold_prediction,
        requires_network=True,
        estimated_time='~2s',
        inputs=[
            SkillParam('uniprot_id', 'string', 'UniProt accession'),
        ],
        outputs=[SkillParam('result', 'dict', 'AlphaFold model info')],
        source='AlphaFold DB (EMBL-EBI)',
    ))

    # --- Composite ---
    registry.register(Skill(
        name='get_protein_profile',
        description='ONE-SHOT comprehensive protein profile: UniProt full annotation '
                    '+ InterPro domains + PDB structures + AlphaFold prediction. '
                    'RECOMMENDED first step before any computational pipeline. '
                    'Returns sequence, function, catalytic activity, domains, '
                    'cofactors, kinetics, GO terms, structures.',
        category='research',
        fn=get_protein_profile,
        requires_network=True,
        estimated_time='~10s',
        inputs=[
            SkillParam('identifier', 'string',
                       'UniProt accession (e.g. P00183 for CYP101/P450cam)'),
        ],
        outputs=[SkillParam('profile', 'dict', 'Comprehensive protein profile')],
        source='UniProt + InterPro + PDB + AlphaFold',
    ))

    # --- PubChem Compound Lookup ---
    registry.register(Skill(
        name='search_pubchem_compound',
        description='Search PubChem for a compound by name. Returns AUTHORITATIVE '
                    'canonical SMILES, molecular formula, weight, InChI, synonyms, '
                    'and 3D SDF download URL. ALWAYS use this instead of guessing SMILES. '
                    'Examples: "asiatic acid", "ursolic acid", "camphor", "caffeine".',
        category='chemistry',
        fn=search_pubchem_compound,
        requires_network=True,
        estimated_time='~3s',
        inputs=[
            SkillParam('name', 'string',
                       'Compound name in English (e.g. "asiatic acid", "caffeine", "camphor")'),
        ],
        outputs=[
            SkillParam('canonical_smiles', 'string', 'Authoritative SMILES from PubChem'),
            SkillParam('molecular_formula', 'string', 'Molecular formula'),
            SkillParam('molecular_weight', 'string', 'Molecular weight'),
            SkillParam('cid', 'int', 'PubChem compound ID'),
        ],
        source='PubChem PUG-REST API',
    ))

    registry.register(Skill(
        name='download_pubchem_sdf',
        description='Download compound 3D structure (SDF) from PubChem by name, CID, '
                    'or SMILES. Provides AUTHORITATIVE 3D coordinates — more reliable '
                    'than RDKit conformer generation for known compounds. '
                    'Use this for docking ligand preparation.',
        category='chemistry',
        fn=download_pubchem_sdf,
        requires_network=True,
        estimated_time='~5s',
        inputs=[
            SkillParam('name', 'string', 'Compound name (e.g. "asiatic acid")', required=False),
            SkillParam('cid', 'int', 'PubChem CID', required=False),
            SkillParam('smiles', 'string', 'SMILES string to look up', required=False),
            SkillParam('output_dir', 'path', 'Output directory (default: CWD)',
                       required=False, is_output_path=True),
            SkillParam('prefer_3d', 'bool', 'Try 3D structure first', required=False, default=True),
        ],
        outputs=[
            SkillParam('sdf_path', 'path', 'Downloaded SDF file path'),
            SkillParam('canonical_smiles', 'string', 'Authoritative SMILES'),
            SkillParam('is_3d', 'bool', 'Whether 3D coordinates are available'),
        ],
        source='PubChem PUG-REST API',
        output_type='sdf',
    ))

    registry.register(Skill(
        name='validate_compound_smiles',
        description='Cross-validate a SMILES string against PubChem authoritative data. '
                    'Detects LLM-hallucinated or incorrect SMILES by comparing molecular '
                    'formula and canonical form with PubChem records. '
                    'Returns the CORRECT SMILES if input is wrong.',
        category='chemistry',
        fn=validate_compound_smiles,
        requires_network=True,
        estimated_time='~3s',
        inputs=[
            SkillParam('name', 'string', 'Expected compound name'),
            SkillParam('smiles', 'string', 'SMILES string to validate'),
        ],
        outputs=[
            SkillParam('valid', 'bool', 'True if SMILES matches PubChem'),
            SkillParam('correct_smiles', 'string', 'PubChem canonical SMILES (if different)'),
        ],
        source='PubChem + RDKit',
    ))
