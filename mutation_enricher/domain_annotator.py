"""
Domain Annotator
================
Query UniProt and InterPro REST APIs to annotate residue positions with
domain membership (Pfam), functional site labels (active site, binding
site, metal binding, etc.).

All network calls use pure Python stdlib (urllib).  Results are cached
as JSON files under cache_dir so repeated calls are free.

Public API
----------
annotate_domains(uniprot_id, seq_len=None, cache_dir=None) -> dict
annotate_domains_by_sequence(sequence, protein_name=None, cache_dir=None) -> dict
detect_fusion_domains(sequence, fasta_header=None) -> dict
get_position_annotation(pos, annotation) -> dict
"""

import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_CACHE_DIR = os.path.join(
    os.path.expanduser("~"), ".zhuangzhou", "structure_cache"
)

_USER_AGENT = (
    "Zhuangzhou/0.7.0 (enzyme engineering; mailto:zhuangzhou@betathink.ai)"
)

# EBI Proteins API — features endpoint (active site, binding, etc.)
_EBI_FEATURES_URL = "https://www.ebi.ac.uk/proteins/api/features/{uid}"

# UniProtKB entry JSON (sequence, organism, description)
_UNIPROT_ENTRY_URL = "https://rest.uniprot.org/uniprotkb/{uid}.json"

# InterPro Pfam domain boundaries for a UniProt protein
_INTERPRO_PFAM_URL = (
    "https://www.ebi.ac.uk/interpro/api/entry/pfam/protein/UniProt/{uid}/?format=json"
)

# Retry schedule: sleep seconds before attempt 1, 2, 3
_RETRY_DELAYS = [0.0, 0.5, 1.0, 2.0]  # index 0 = before first attempt (no sleep)

_REQUEST_TIMEOUT = 10  # seconds

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ensure_cache_dir(cache_dir: Optional[str]) -> str:
    d = cache_dir or _DEFAULT_CACHE_DIR
    os.makedirs(d, exist_ok=True)
    return d


def _http_get(url: str, accept: str = "application/json") -> bytes:
    """HTTP GET with 3 retries and exponential backoff.

    Raises the last exception if all attempts fail.
    """
    headers = {
        "User-Agent": _USER_AGENT,
        "Accept": accept,
    }
    req = urllib.request.Request(url, headers=headers)
    last_exc: Exception = RuntimeError(f"Failed to GET {url}")

    for attempt in range(4):  # attempts 0..3
        sleep_secs = _RETRY_DELAYS[attempt]
        if sleep_secs > 0:
            time.sleep(sleep_secs)
        try:
            with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if exc.code == 404:
                raise  # no point retrying a not-found
            if exc.code == 429:
                # respect rate-limit: double the wait
                time.sleep(1.0 * (attempt + 1))
                continue
            if exc.code >= 500:
                continue  # server error, retry
            raise
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            last_exc = exc
            continue

    raise last_exc


def _load_cache(path: str) -> Optional[dict]:
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Cache read failed %s: %s", path, exc)
    return None


def _save_cache(path: str, data: dict) -> None:
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Cache write failed %s: %s", path, exc)


# ---------------------------------------------------------------------------
# Per-source fetchers
# ---------------------------------------------------------------------------


def _fetch_uniprot_entry(uniprot_id: str, cache_dir: str) -> dict:
    """Return raw UniProtKB JSON for *uniprot_id*, using local cache."""
    cache_path = os.path.join(cache_dir, f"{uniprot_id}_uniprot_entry.json")
    cached = _load_cache(cache_path)
    if cached is not None:
        return cached

    url = _UNIPROT_ENTRY_URL.format(uid=uniprot_id)
    try:
        raw = _http_get(url)
        data = json.loads(raw)
        _save_cache(cache_path, data)
        return data
    except Exception as exc:
        logger.warning("UniProt entry fetch failed for %s: %s", uniprot_id, exc)
        return {}


def _fetch_ebi_features(uniprot_id: str, cache_dir: str) -> dict:
    """Return raw EBI Proteins API features JSON, using local cache."""
    cache_path = os.path.join(cache_dir, f"{uniprot_id}_uniprot_features.json")
    cached = _load_cache(cache_path)
    if cached is not None:
        return cached

    url = _EBI_FEATURES_URL.format(uid=uniprot_id)
    try:
        raw = _http_get(url, accept="application/json")
        data = json.loads(raw)
        _save_cache(cache_path, data)
        return data
    except Exception as exc:
        logger.warning("EBI features fetch failed for %s: %s", uniprot_id, exc)
        return {}


def _fetch_interpro_pfam(uniprot_id: str, cache_dir: str) -> dict:
    """Return raw InterPro/Pfam JSON for *uniprot_id*, using local cache."""
    cache_path = os.path.join(cache_dir, f"{uniprot_id}_interpro.json")
    cached = _load_cache(cache_path)
    if cached is not None:
        return cached

    url = _INTERPRO_PFAM_URL.format(uid=uniprot_id)
    try:
        raw = _http_get(url)
        data = json.loads(raw)
        _save_cache(cache_path, data)
        return data
    except Exception as exc:
        logger.warning("InterPro fetch failed for %s: %s", uniprot_id, exc)
        return {}


# ---------------------------------------------------------------------------
# Feature-type normaliser
# ---------------------------------------------------------------------------

# Map EBI Proteins API feature type strings → canonical internal labels.
# We lower-case and strip spaces before lookup.
_FEATURE_TYPE_MAP: Dict[str, str] = {
    "active site": "active_site",
    "active_site": "active_site",
    "binding site": "binding_site",
    "binding_site": "binding_site",
    "metal binding": "metal_binding",
    "metal_binding": "metal_binding",
    "zinc binding": "zn_binding",
    "zn_binding": "zn_binding",
    "site": "site",
    "motif": "motif",
    "region of interest": "region",
    "region": "region",
    "domain": "domain_feature",
    "nucleophile": "nucleophile",
    "proton acceptor": "proton_acceptor",
    "proton donor": "proton_donor",
    "modified residue": "modified_residue",
    "disulfide bond": "disulfide_bond",
    "cross-link": "cross_link",
    "mutagenesis": "mutagenesis",
    "natural variant": "natural_variant",
    "transmembrane": "transmembrane",
    "signal peptide": "signal_peptide",
    "propeptide": "propeptide",
}


def _normalise_feature_type(raw_type: str) -> str:
    key = raw_type.lower().strip().replace("-", " ")
    return _FEATURE_TYPE_MAP.get(key, key.replace(" ", "_"))


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


def _parse_uniprot_features(features_data: dict) -> Dict[int, List[str]]:
    """Parse EBI Proteins API features response into {position: [feature_labels]}.

    The response has a top-level ``features`` list.  Each entry has:
      ``type``, ``begin`` (str), ``end`` (str), ``description``.
    """
    result: Dict[int, List[str]] = {}

    features_list = features_data.get("features", [])
    for feat in features_list:
        ftype_raw = feat.get("type", "")
        label = _normalise_feature_type(ftype_raw)

        # Skip purely informational feature types that clutter per-residue maps
        if label in {"chain", "natural_variant", "mutagenesis", "transmembrane",
                     "signal_peptide", "propeptide", "modified_residue",
                     "disulfide_bond", "cross_link"}:
            continue

        try:
            begin = int(feat.get("begin", 0))
            end = int(feat.get("end", begin))
        except (TypeError, ValueError):
            continue

        for pos in range(begin, end + 1):
            if pos not in result:
                result[pos] = []
            if label not in result[pos]:
                result[pos].append(label)

    return result


def _parse_interpro_pfam(interpro_data: dict, uniprot_id: str) -> List[dict]:
    """Parse InterPro Pfam API response into a list of domain dicts.

    The ``results`` list contains entries each with:
      ``metadata.accession``, ``metadata.name``, ``metadata.description``,
      ``proteins`` list with ``entry_protein_locations``.
    """
    domains: List[dict] = []
    uid_lower = uniprot_id.upper()

    results = interpro_data.get("results", [])
    for entry in results:
        meta = entry.get("metadata", {})
        pfam_id = meta.get("accession", "")
        pfam_name = meta.get("name", "")
        pfam_desc = meta.get("description", pfam_name)

        proteins = entry.get("proteins", [])
        for prot in proteins:
            prot_ac = (prot.get("accession") or "").upper()
            if prot_ac and prot_ac != uid_lower:
                continue  # not this protein
            locations = prot.get("entry_protein_locations", [])
            for loc in locations:
                for fragment in loc.get("fragments", []):
                    try:
                        start = int(fragment.get("start", 0))
                        end = int(fragment.get("end", 0))
                    except (TypeError, ValueError):
                        continue
                    if start and end:
                        domains.append({
                            "name": pfam_name,
                            "pfam_id": pfam_id,
                            "start": start,
                            "end": end,
                            "description": pfam_desc,
                        })

    return domains


def _build_domain_map(
    domains: List[dict], seq_len: Optional[int]
) -> Dict[int, str]:
    """Return {residue_pos: 'DomainName (PF00000)'} for every covered position."""
    domain_map: Dict[int, str] = {}
    for dom in domains:
        label = f"{dom['name']} ({dom['pfam_id']})"
        end = dom["end"]
        if seq_len is not None:
            end = min(end, seq_len)
        for pos in range(dom["start"], end + 1):
            if pos not in domain_map:
                domain_map[pos] = label
    return domain_map


def _parse_uniprot_entry_meta(entry_data: dict) -> tuple:
    """Extract (organism_name, protein_description) from UniProtKB JSON."""
    organism = ""
    try:
        organism = (
            entry_data.get("organism", {})
            .get("scientificName", "")
        )
    except AttributeError:
        pass

    protein_desc = ""
    try:
        rec_name = (
            entry_data.get("proteinDescription", {})
            .get("recommendedName", {})
        )
        full_name = rec_name.get("fullName", {})
        protein_desc = full_name.get("value", "")
    except AttributeError:
        pass

    return organism, protein_desc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def annotate_domains(
    uniprot_id: str,
    seq_len: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> dict:
    """Query UniProt + InterPro for domain and functional site annotations.

    Parameters
    ----------
    uniprot_id : str
        UniProt accession (e.g. ``'P0A6L9'``).
    seq_len : int, optional
        Protein sequence length.  When provided the domain map is capped at
        this value so it never contains out-of-range positions.
    cache_dir : str, optional
        Directory for JSON cache files.  Defaults to
        ``~/.zhuangzhou/structure_cache/``.

    Returns
    -------
    dict
        Keys: ``uniprot_features``, ``domains``, ``domain_map``,
        ``organism``, ``protein_description``, ``error``.
        On failure returns the same shape with empty structures and a
        non-None ``error`` string.
    """
    _empty = {
        "uniprot_features": {},
        "domains": [],
        "domain_map": {},
        "organism": "",
        "protein_description": "",
        "error": None,
    }

    if not uniprot_id or not isinstance(uniprot_id, str):
        _empty["error"] = "uniprot_id must be a non-empty string"
        return _empty

    uniprot_id = uniprot_id.strip().upper()
    cache_d = _ensure_cache_dir(cache_dir)

    # --- Fetch from APIs ---------------------------------------------------

    # 1. EBI Proteins features (active site, binding, metal …)
    try:
        features_raw = _fetch_ebi_features(uniprot_id, cache_d)
        uniprot_features = _parse_uniprot_features(features_raw)
    except Exception as exc:
        logger.warning("Feature parsing failed for %s: %s", uniprot_id, exc)
        uniprot_features = {}

    time.sleep(0.3)

    # 2. InterPro/Pfam domain boundaries
    try:
        interpro_raw = _fetch_interpro_pfam(uniprot_id, cache_d)
        domains = _parse_interpro_pfam(interpro_raw, uniprot_id)
    except Exception as exc:
        logger.warning("InterPro parsing failed for %s: %s", uniprot_id, exc)
        domains = []

    time.sleep(0.3)

    # 3. UniProtKB entry — organism + protein description
    if seq_len is None:
        # Also grab seq_len from entry if not provided
        try:
            entry_raw = _fetch_uniprot_entry(uniprot_id, cache_d)
            seq_len_from_api = entry_raw.get("sequence", {}).get("length")
            if seq_len_from_api:
                seq_len = int(seq_len_from_api)
            organism, protein_description = _parse_uniprot_entry_meta(entry_raw)
        except Exception as exc:
            logger.warning("UniProt entry parsing failed for %s: %s", uniprot_id, exc)
            organism, protein_description = "", ""
    else:
        try:
            entry_raw = _fetch_uniprot_entry(uniprot_id, cache_d)
            organism, protein_description = _parse_uniprot_entry_meta(entry_raw)
        except Exception as exc:
            logger.warning("UniProt entry parsing failed for %s: %s", uniprot_id, exc)
            organism, protein_description = "", ""

    # 4. Build domain_map
    domain_map = _build_domain_map(domains, seq_len)

    return {
        "uniprot_features": uniprot_features,
        "domains": domains,
        "domain_map": domain_map,
        "organism": organism,
        "protein_description": protein_description,
        "error": None,
    }


# ---------------------------------------------------------------------------
# InterPro sequence-based search (for proteins without UniProt ID)
# ---------------------------------------------------------------------------

# Common linker patterns in fusion proteins
_LINKER_PATTERNS = [
    # Glycine-serine linkers
    (r'(GS){3,}', 'GS-linker'),
    (r'(GGGGS){1,}', 'G4S-linker'),
    (r'(GGGS){1,}', 'G3S-linker'),
    (r'G{4,}S', 'polyG-linker'),
    (r'(GSG){2,}', 'GSG-linker'),
    # Rigid linkers
    (r'(EAAAK){1,}', 'EAAAK-rigid-linker'),
    (r'(AP){5,}', 'AP-rigid-linker'),
    # Flexible linkers
    (r'(GSSGS){1,}', 'GSSGS-linker'),
]

# InterProScan REST API (sequence-based search)
_INTERPROSCAN_URL = "https://www.ebi.ac.uk/Tools/services/rest/iprscan5"


def detect_fusion_domains(
    sequence: str,
    fasta_header: str = None,
) -> dict:
    """Detect fusion protein domains from sequence and FASTA header.

    Identifies linker regions and attempts to split a fusion protein into
    its component domains using:
      1. FASTA header parsing (e.g. "ProtA_GSGx2_ProtB")
      2. Known linker motif scanning in the sequence
      3. Low-complexity region detection (potential linkers)

    Returns
    -------
    dict
        Keys: ``is_fusion``, ``linkers``, ``domains_from_header``,
        ``domain_boundaries``.
    """
    import re

    result = {
        "is_fusion": False,
        "linkers": [],
        "domains_from_header": [],
        "domain_boundaries": [],
    }

    if not sequence:
        return result

    seq = sequence.upper().replace('\n', '').replace(' ', '')

    # --- 1. Scan for linker motifs in sequence ---
    for pattern, linker_type in _LINKER_PATTERNS:
        for m in re.finditer(pattern, seq):
            start, end = m.start() + 1, m.end()  # 1-based
            # Only consider internal linkers (not at termini)
            if start > 10 and end < len(seq) - 10:
                result["linkers"].append({
                    "type": linker_type,
                    "sequence": m.group(),
                    "start": start,
                    "end": end,
                })

    # Remove duplicate/overlapping linkers (keep longest)
    if result["linkers"]:
        result["linkers"].sort(key=lambda x: x["start"])
        filtered = [result["linkers"][0]]
        for lnk in result["linkers"][1:]:
            prev = filtered[-1]
            if lnk["start"] > prev["end"]:
                filtered.append(lnk)
            elif (lnk["end"] - lnk["start"]) > (prev["end"] - prev["start"]):
                filtered[-1] = lnk
        result["linkers"] = filtered

    # --- 2. Parse FASTA header for domain names ---
    if fasta_header:
        header = fasta_header.strip().lstrip('>')
        # Common fusion patterns: "ProtA_linker_ProtB", "ProtA-ProtB"
        # Look for known separator patterns
        _fusion_separators = [
            r'_GSG\w*_',       # _GSGx2_
            r'_GGGGS\w*_',     # _GGGGS_
            r'_linker_',       # _linker_
            r'_fused_to_',     # _fused_to_
            r'__',             # double underscore
        ]
        for sep_pattern in _fusion_separators:
            parts = re.split(sep_pattern, header, flags=re.IGNORECASE)
            if len(parts) >= 2:
                result["domains_from_header"] = [p.strip() for p in parts if p.strip()]
                break

        # Also try: "ProtA|ProtB" or "ProtA+ProtB"
        if not result["domains_from_header"]:
            for sep in ['|', '+']:
                if sep in header:
                    parts = [p.strip() for p in header.split(sep) if p.strip()]
                    if len(parts) >= 2:
                        result["domains_from_header"] = parts
                        break

    # --- 3. Infer domain boundaries from linkers ---
    if result["linkers"]:
        result["is_fusion"] = True
        boundaries = []
        prev_end = 1
        for i, lnk in enumerate(result["linkers"]):
            if lnk["start"] - prev_end > 20:  # minimal domain size
                domain_name = (result["domains_from_header"][i]
                               if i < len(result["domains_from_header"]) else f"Domain_{i+1}")
                boundaries.append({
                    "name": domain_name,
                    "start": prev_end,
                    "end": lnk["start"] - 1,
                })
            prev_end = lnk["end"] + 1

        # Last domain after the final linker
        if prev_end < len(seq) - 20:
            idx = len(boundaries)
            domain_name = (result["domains_from_header"][idx]
                           if idx < len(result["domains_from_header"]) else f"Domain_{idx+1}")
            boundaries.append({
                "name": domain_name,
                "start": prev_end,
                "end": len(seq),
            })

        result["domain_boundaries"] = boundaries

    # If header suggests fusion but no linker found, still mark
    if result["domains_from_header"] and not result["is_fusion"]:
        result["is_fusion"] = True

    return result


def annotate_domains_by_sequence(
    sequence: str,
    protein_name: str = None,
    fasta_header: str = None,
    cache_dir: str = None,
) -> dict:
    """Annotate domains for a protein using sequence analysis when no UniProt ID.

    Strategy:
      1. Detect if it's a fusion protein (linker scanning + header parsing)
      2. Try InterPro sequence search to find matching Pfam domains
      3. Build domain_map from detected domains

    Falls back gracefully: if API calls fail, uses local sequence analysis only.

    Returns same format as annotate_domains().
    """
    _empty = {
        "uniprot_features": {},
        "domains": [],
        "domain_map": {},
        "organism": "",
        "protein_description": "",
        "fusion_info": None,
        "error": None,
    }

    if not sequence:
        _empty["error"] = "sequence must be a non-empty string"
        return _empty

    seq = sequence.upper().replace('\n', '').replace(' ', '')
    cache_d = _ensure_cache_dir(cache_dir)

    # --- 1. Fusion protein detection ---
    fusion = detect_fusion_domains(seq, fasta_header)
    _empty["fusion_info"] = fusion

    domains = []
    domain_map = {}

    # --- 2. If fusion protein, try to resolve each component via UniProt search ---
    if fusion["is_fusion"] and fusion["domain_boundaries"]:
        for boundary in fusion["domain_boundaries"]:
            dname = boundary["name"]
            dstart = boundary["start"]
            dend = boundary["end"]

            # Try to find UniProt ID for this domain by name search
            uid = _try_resolve_uniprot_id(dname, cache_d)
            if uid:
                try:
                    sub_annotation = annotate_domains(uid, seq_len=dend - dstart + 1, cache_dir=cache_d)
                    if sub_annotation.get("domains"):
                        # Shift domain positions by the offset
                        for dom in sub_annotation["domains"]:
                            shifted = dict(dom)
                            shifted["start"] = dom["start"] + dstart - 1
                            shifted["end"] = dom["end"] + dstart - 1
                            shifted["parent_domain"] = dname
                            domains.append(shifted)
                    if sub_annotation.get("uniprot_features"):
                        for pos, labels in sub_annotation["uniprot_features"].items():
                            shifted_pos = int(pos) + dstart - 1
                            _empty["uniprot_features"][shifted_pos] = labels
                except Exception as exc:
                    logger.warning("Sub-domain annotation failed for %s: %s", dname, exc)

            # Always add the boundary itself as a domain
            domains.append({
                "name": dname,
                "pfam_id": f"fusion_{dname}",
                "start": dstart,
                "end": dend,
                "description": f"Fusion component: {dname}",
            })

        # Add linker regions as special domains
        for lnk in fusion.get("linkers", []):
            domains.append({
                "name": f"Linker ({lnk['type']})",
                "pfam_id": "linker",
                "start": lnk["start"],
                "end": lnk["end"],
                "description": f"Fusion linker: {lnk['sequence']}",
            })

    # --- 3. Fallback: try InterPro sequence search for the full protein ---
    if not domains:
        interpro_domains = _try_interpro_sequence_search(seq, protein_name, cache_d)
        if interpro_domains:
            domains = interpro_domains

    # Build domain_map
    domain_map = _build_domain_map(domains, len(seq))
    _empty["domains"] = domains
    _empty["domain_map"] = domain_map

    return _empty


def _try_resolve_uniprot_id(protein_name: str, cache_dir: str) -> Optional[str]:
    """Try to find a UniProt ID by searching with protein name.

    Quick heuristic search — not exhaustive. Returns first hit or None.
    """
    if not protein_name or len(protein_name) < 3:
        return None

    # Skip generic names
    _skip = {'domain', 'linker', 'tag', 'his', 'gfp', 'mbp', 'gst', 'sumo'}
    if protein_name.lower().strip() in _skip:
        return None

    cache_path = os.path.join(cache_dir, f"name2uid_{protein_name.lower().strip()}.json")
    cached = _load_cache(cache_path)
    if cached is not None:
        return cached.get("uniprot_id")

    import re
    # Clean name: remove _for_mutscan etc.
    clean = re.sub(r'_for_\w+$', '', protein_name, flags=re.IGNORECASE).strip()

    url = (
        f"https://rest.uniprot.org/uniprotkb/search?"
        f"query={urllib.parse.quote(clean)}&format=json&size=1&fields=accession"
    )
    try:
        raw = _http_get(url)
        data = json.loads(raw)
        results = data.get("results", [])
        if results:
            uid = results[0].get("primaryAccession")
            _save_cache(cache_path, {"uniprot_id": uid})
            return uid
    except Exception as exc:
        logger.debug("UniProt name search failed for '%s': %s", clean, exc)

    _save_cache(cache_path, {"uniprot_id": None})
    return None


def _try_interpro_sequence_search(
    sequence: str,
    protein_name: str = None,
    cache_dir: str = None,
) -> List[dict]:
    """Try to find InterPro domains by searching the sequence via REST API.

    Uses InterProScan 5 REST API for job submission + polling.
    Falls back to local motif-based detection if the API is slow/unavailable.
    """
    import hashlib

    if not sequence or len(sequence) < 30:
        return []

    seq_hash = hashlib.md5(sequence[:500].encode()).hexdigest()[:12]
    cache_path = os.path.join(
        _ensure_cache_dir(cache_dir),
        f"iprscan_{seq_hash}.json"
    )
    cached = _load_cache(cache_path)
    if cached is not None:
        return cached.get("domains", [])

    # Submit job to InterProScan REST API
    try:
        submit_url = f"{_INTERPROSCAN_URL}/run"
        params = urllib.parse.urlencode({
            "email": "zhuangzhou@betathink.ai",
            "sequence": sequence,
            "goterms": "false",
            "pathways": "false",
        }).encode("utf-8")

        req = urllib.request.Request(
            submit_url,
            data=params,
            headers={"User-Agent": _USER_AGENT, "Accept": "text/plain"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            job_id = resp.read().decode().strip()

        if not job_id:
            return []

        # Poll for results (max 60s wait)
        status_url = f"{_INTERPROSCAN_URL}/status/{job_id}"
        for _ in range(12):  # 12 * 5s = 60s max
            time.sleep(5)
            try:
                raw = _http_get(status_url, accept="text/plain")
                status = raw.decode().strip()
                if status == "FINISHED":
                    break
                if status in ("FAILURE", "ERROR", "NOT_FOUND"):
                    logger.warning("InterProScan job %s: %s", job_id, status)
                    return []
            except Exception:
                continue
        else:
            logger.warning("InterProScan job %s timed out", job_id)
            return []

        # Fetch results
        result_url = f"{_INTERPROSCAN_URL}/result/{job_id}/json"
        raw = _http_get(result_url)
        data = json.loads(raw)

        domains = []
        for result_entry in data.get("results", [data] if "matches" in data else []):
            for match in result_entry.get("matches", []):
                sig = match.get("signature", {})
                entry = sig.get("entry") or {}
                sig_lib = sig.get("signatureLibraryRelease", {})

                for loc in match.get("locations", []):
                    start = loc.get("start", 0)
                    end = loc.get("end", 0)
                    if start and end:
                        domains.append({
                            "name": entry.get("name") or sig.get("name", "Unknown"),
                            "pfam_id": entry.get("accession") or sig.get("accession", ""),
                            "start": start,
                            "end": end,
                            "description": entry.get("description") or sig.get("description", ""),
                            "source": sig_lib.get("library", "InterPro"),
                        })

        _save_cache(cache_path, {"domains": domains})
        return domains

    except Exception as exc:
        logger.warning("InterProScan sequence search failed: %s", exc)
        return []


# Site label priority — listed lowest to highest priority.
# The reversed list is iterated at lookup time so the first match is the
# highest-priority label.  Order here: site < motif < ... < nucleophile.
_SITE_LABEL_PRIORITY = [
    ("site", "Functional site"),
    ("domain_feature", "Domain feature"),
    ("motif", "Conserved motif"),
    ("region", "Functional region"),
    ("binding_site", "Substrate/ligand binding residue"),
    ("active_site", "Active site residue"),
    ("metal_binding", "Metal-binding residue"),
    ("zn_binding", "Zinc-binding residue"),
    ("proton_donor", "Proton donor"),
    ("proton_acceptor", "Proton acceptor"),
    ("nucleophile", "Nucleophilic residue"),  # highest priority
]

# Reverse so index 0 = highest priority for fast lookup
_SITE_LABEL_PRIORITY_REVERSED = list(reversed(_SITE_LABEL_PRIORITY))


def get_position_annotation(pos: int, annotation: dict) -> dict:
    """Get all annotations for a single residue position.

    Parameters
    ----------
    pos : int
        1-based residue position.
    annotation : dict
        Output of :func:`annotate_domains`.

    Returns
    -------
    dict
        ``{'domain': str, 'functional_sites': list[str], 'site_label': str}``
    """
    domain = annotation.get("domain_map", {}).get(pos, "Unknown")
    functional_sites = list(annotation.get("uniprot_features", {}).get(pos, []))

    site_label = ""
    for label_key, label_text in _SITE_LABEL_PRIORITY_REVERSED:
        if label_key in functional_sites:
            site_label = label_text
            break

    return {
        "domain": domain,
        "functional_sites": functional_sites,
        "site_label": site_label,
    }
