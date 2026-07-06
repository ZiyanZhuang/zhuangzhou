"""
Literature Validator
====================
Cross-reference mutation positions against:
  A) UniProt natural variants (EBI Proteins API)
  B) PubMed literature (NCBI Entrez E-utilities)

Then synthesise structured evidence summaries, optionally using an LLM
for richer narrative.

All network calls use pure Python stdlib (urllib).  Results are cached
as JSON files under cache_dir.

Public API
----------
get_uniprot_variants(uniprot_id, positions=None, cache_dir=None) -> dict
search_pubmed_mutations(protein_name, positions=None, pfam_id=None,
                        max_results=5, cache_dir=None) -> list
synthesize_evidence(mutation, variants, papers, llm_client=None) -> dict
validate_mutations(mutations, uniprot_id, protein_name='', pfam_id=None,
                   llm_client=None, cache_dir=None,
                   max_pubmed_queries=3) -> list
"""

import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

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

# EBI Proteins variation endpoint
_EBI_VARIATION_URL = "https://www.ebi.ac.uk/proteins/api/variation/{uid}"

# NCBI Entrez E-utilities
_NCBI_ESEARCH_URL = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
)
_NCBI_EFETCH_URL = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
)

# Retry schedule: sleep seconds *before* attempt index
_RETRY_DELAYS = [0.0, 0.5, 1.0, 2.0]

_REQUEST_TIMEOUT = 10  # seconds

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ensure_cache_dir(cache_dir: Optional[str]) -> str:
    d = cache_dir or _DEFAULT_CACHE_DIR
    os.makedirs(d, exist_ok=True)
    return d


def _http_get(url: str, accept: str = "application/json") -> bytes:
    """HTTP GET with up to 3 retries and exponential backoff.

    Raises the last encountered exception when all attempts are exhausted.
    """
    headers = {"User-Agent": _USER_AGENT, "Accept": accept}
    req = urllib.request.Request(url, headers=headers)
    last_exc: Exception = RuntimeError(f"Failed to GET {url}")

    for attempt in range(4):
        sleep_secs = _RETRY_DELAYS[attempt]
        if sleep_secs > 0:
            time.sleep(sleep_secs)
        try:
            with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if exc.code == 404:
                raise
            if exc.code == 429:
                time.sleep(1.5 * (attempt + 1))
                continue
            if exc.code >= 500:
                continue
            raise
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            last_exc = exc
            continue

    raise last_exc


def _load_cache(path: str) -> Optional[Any]:
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        except Exception as exc:
            logger.warning("Cache read failed %s: %s", path, exc)
    return None


def _save_cache(path: str, data: Any) -> None:
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.warning("Cache write failed %s: %s", path, exc)


def _safe_cache_key(text: str) -> str:
    """Convert arbitrary text to a filesystem-safe ASCII slug."""
    import re
    slug = re.sub(r"[^\w\-]", "_", text.strip().lower())
    slug = re.sub(r"_+", "_", slug)
    return slug[:80]


# ---------------------------------------------------------------------------
# Part A: UniProt Natural Variants
# ---------------------------------------------------------------------------


def _fetch_ebi_variation(uniprot_id: str, cache_dir: str) -> dict:
    """Return raw EBI Proteins variation JSON, using local cache."""
    cache_path = os.path.join(cache_dir, f"{uniprot_id}_variation.json")
    cached = _load_cache(cache_path)
    if cached is not None:
        return cached

    url = _EBI_VARIATION_URL.format(uid=uniprot_id)
    try:
        raw = _http_get(url, accept="application/json")
        data = json.loads(raw)
        _save_cache(cache_path, data)
        return data
    except Exception as exc:
        logger.warning("EBI variation fetch failed for %s: %s", uniprot_id, exc)
        return {}


def _parse_variant_record(var: dict) -> dict:
    """Normalise a single EBI variation feature into a canonical variant record."""
    # Source type: 'uniprot', 'dbSNP', 'cosmic', 'clinvar', etc.
    source_raw = var.get("sourceType", "")
    source_map = {
        "uniprot": "UniProt",
        "dbsnp": "dbSNP",
        "cosmic": "COSMIC",
        "clinvar": "ClinVar",
    }
    source = source_map.get(source_raw.lower(), source_raw or "Unknown")

    # Clinical significance
    clin_sigs = var.get("clinicalSignificances", [])
    if isinstance(clin_sigs, list) and clin_sigs:
        # Each item may be a string or a dict with a 'type' key
        def _sig_str(s):
            if isinstance(s, str):
                return s
            return s.get("type", str(s))
        clinical_significance = "; ".join(_sig_str(s) for s in clin_sigs)
    else:
        clinical_significance = "Uncertain"

    # Disease associations
    associations = var.get("association", [])
    disease_parts = []
    for assoc in (associations or []):
        name = assoc.get("name", "") if isinstance(assoc, dict) else str(assoc)
        if name:
            disease_parts.append(name)
    disease = "; ".join(disease_parts)

    # Description from descriptions list or direct description field
    descs = var.get("descriptions", [])
    if isinstance(descs, list) and descs:
        desc_texts = []
        for d in descs:
            if isinstance(d, str):
                desc_texts.append(d)
            elif isinstance(d, dict):
                desc_texts.append(d.get("value", d.get("text", "")))
        description = "; ".join(t for t in desc_texts if t)
    else:
        description = var.get("description", "")

    try:
        position = int(var.get("begin", var.get("start", 0)))
    except (TypeError, ValueError):
        position = 0

    return {
        "wt": var.get("wildType", ""),
        "mut": var.get("mutatedType", ""),
        "position": position,
        "source": source,
        "clinical_significance": clinical_significance,
        "disease": disease,
        "description": description,
    }


def get_uniprot_variants(
    uniprot_id: str,
    positions: Optional[List[int]] = None,
    cache_dir: Optional[str] = None,
) -> Dict[int, List[dict]]:
    """Retrieve UniProt natural variants for *uniprot_id*.

    Parameters
    ----------
    uniprot_id : str
        UniProt accession.
    positions : list of int, optional
        If provided, only return variants at these residue positions.
    cache_dir : str, optional
        Cache directory.  Defaults to ``~/.zhuangzhou/structure_cache/``.

    Returns
    -------
    dict
        ``{position: [variant_record, ...]}``

        Each ``variant_record`` is::

            {
                'wt': 'G',
                'mut': 'D',
                'position': 59,
                'source': 'UniProt',
                'clinical_significance': 'Uncertain',
                'disease': '',
                'description': 'Natural variant from mouse homolog',
            }

        Returns an empty dict on error.
    """
    if not uniprot_id:
        return {}

    cache_d = _ensure_cache_dir(cache_dir)
    uniprot_id = uniprot_id.strip().upper()

    raw = _fetch_ebi_variation(uniprot_id, cache_d)
    if not raw:
        return {}

    # The variation API wraps features in 'features' key
    features = raw.get("features", raw.get("variants", []))
    if not features:
        # Some responses have a top-level list directly
        if isinstance(raw, list):
            features = raw
        else:
            return {}

    position_set = set(positions) if positions else None

    result: Dict[int, List[dict]] = {}
    for var in features:
        # Only include actual variant features, not other types
        feat_type = (var.get("type") or var.get("category") or "").lower()
        if feat_type and feat_type not in {"", "natural variant", "variant",
                                           "uniprot", "natural_variant"}:
            continue

        rec = _parse_variant_record(var)
        pos = rec["position"]
        if pos == 0:
            continue
        if position_set is not None and pos not in position_set:
            continue

        if pos not in result:
            result[pos] = []
        result[pos].append(rec)

    return result


# ---------------------------------------------------------------------------
# Part B: PubMed Literature Search
# ---------------------------------------------------------------------------


def _ncbi_esearch(query: str, max_results: int) -> List[str]:
    """Run an Entrez esearch and return list of PMIDs."""
    params = urllib.parse.urlencode({
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "sort": "relevance",
    })
    url = f"{_NCBI_ESEARCH_URL}?{params}"
    try:
        raw = _http_get(url, accept="application/json")
        data = json.loads(raw)
        return data.get("esearchresult", {}).get("idlist", [])
    except Exception as exc:
        logger.warning("PubMed esearch failed (query=%r): %s", query, exc)
        return []


def _ncbi_efetch_abstracts(pmids: List[str]) -> List[dict]:
    """Fetch PubMed abstracts for a list of PMIDs and return parsed records."""
    if not pmids:
        return []

    params = urllib.parse.urlencode({
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        "rettype": "abstract",
    })
    url = f"{_NCBI_EFETCH_URL}?{params}"

    try:
        raw = _http_get(url, accept="text/xml")
        raw_text = raw.decode("utf-8", errors="replace")
    except Exception as exc:
        logger.warning("PubMed efetch failed for PMIDs %s: %s", pmids, exc)
        return []

    records: List[dict] = []
    try:
        root = ET.fromstring(raw_text)
        for article in root.findall(".//PubmedArticle"):
            rec = _parse_pubmed_article(article)
            if rec:
                records.append(rec)
    except ET.ParseError as exc:
        logger.warning("PubMed XML parse error: %s", exc)

    return records


def _parse_pubmed_article(article_elem: ET.Element) -> Optional[dict]:
    """Extract title, abstract, year, PMID from a <PubmedArticle> element."""
    try:
        pmid = ""
        pmid_elem = article_elem.find(".//PMID")
        if pmid_elem is not None and pmid_elem.text:
            pmid = pmid_elem.text.strip()

        # Title
        title = ""
        title_elem = article_elem.find(".//ArticleTitle")
        if title_elem is not None:
            # Flatten all text, including child elements
            title = "".join(title_elem.itertext()).strip()

        # Abstract — may have multiple <AbstractText> sections
        abstract_parts = []
        for ab_elem in article_elem.findall(".//AbstractText"):
            label = ab_elem.get("Label", "")
            text = "".join(ab_elem.itertext()).strip()
            if text:
                if label:
                    abstract_parts.append(f"{label}: {text}")
                else:
                    abstract_parts.append(text)
        abstract = " ".join(abstract_parts)

        # Year — prefer PubDate/Year, fall back to MedlineDate
        year = 0
        year_elem = article_elem.find(".//PubDate/Year")
        if year_elem is not None and year_elem.text:
            try:
                year = int(year_elem.text.strip())
            except ValueError:
                pass
        if year == 0:
            medline_elem = article_elem.find(".//PubDate/MedlineDate")
            if medline_elem is not None and medline_elem.text:
                # e.g. "2019 Jan-Feb"
                try:
                    year = int(medline_elem.text.strip()[:4])
                except ValueError:
                    pass

        if not pmid:
            return None

        return {
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "year": year,
            "relevance_note": "",
        }
    except Exception as exc:
        logger.warning("PubMed article parse error: %s", exc)
        return None


def search_pubmed_mutations(
    protein_name: str,
    positions: Optional[List[int]] = None,
    pfam_id: Optional[str] = None,
    max_results: int = 5,
    cache_dir: Optional[str] = None,
) -> List[dict]:
    """Search PubMed for papers about mutations in *protein_name*.

    Parameters
    ----------
    protein_name : str
        Protein name (e.g. ``'nicotinamidase'`` or ``'PncA'``).
    positions : list of int, optional
        Residue positions to include in the query (each generates a separate
        search; results are deduplicated by PMID).
    pfam_id : str, optional
        Pfam accession (e.g. ``'PF01872'``) for a family-level search.
    max_results : int
        Maximum results per sub-query.
    cache_dir : str, optional
        Cache directory.

    Returns
    -------
    list of dict
        Each item::

            {
                'pmid': '12345678',
                'title': '...',
                'abstract': '...',
                'year': 2019,
                'relevance_note': '',
            }

        Empty list on error or no results.
    """
    cache_d = _ensure_cache_dir(cache_dir)
    seen_pmids: set = set()
    all_records: List[dict] = []

    def _query_and_collect(query: str, label: str) -> None:
        cache_key = _safe_cache_key(f"pubmed_{query}")
        cache_path = os.path.join(cache_d, f"{cache_key}.json")

        # Try cache first
        cached_pmids = _load_cache(cache_path)
        if cached_pmids is not None:
            pmids = [p for p in cached_pmids if p not in seen_pmids]
        else:
            pmids = _ncbi_esearch(query, max_results)
            _save_cache(cache_path, pmids)
            time.sleep(0.3)

        new_pmids = [p for p in pmids if p not in seen_pmids]
        if not new_pmids:
            return

        # Check abstract cache
        abstracts_cache_key = _safe_cache_key(f"pubmed_abs_{'_'.join(sorted(new_pmids))}")
        abstracts_cache_path = os.path.join(cache_d, f"{abstracts_cache_key}.json")
        cached_records = _load_cache(abstracts_cache_path)
        if cached_records is not None:
            records = cached_records
        else:
            records = _ncbi_efetch_abstracts(new_pmids)
            _save_cache(abstracts_cache_path, records)
            time.sleep(0.3)

        for rec in records:
            pmid = rec.get("pmid", "")
            if pmid and pmid not in seen_pmids:
                seen_pmids.add(pmid)
                all_records.append(rec)

    # 1. Protein + position searches
    if positions:
        for pos in positions:
            query = (
                f'"{protein_name}"[Title/Abstract] AND '
                f'"mutation"[Title/Abstract] AND '
                f'"{pos}"[Title/Abstract]'
            )
            _query_and_collect(query, f"pos_{pos}")

    # 2. Protein-level mutation search (broader)
    if protein_name:
        query = (
            f'"{protein_name}"[Title/Abstract] AND '
            f'"mutation"[Title/Abstract]'
        )
        _query_and_collect(query, "protein_mutation")

    # 3. Pfam family search
    if pfam_id:
        query = (
            f'"{pfam_id}"[Title/Abstract] AND '
            f'"mutation"[Title/Abstract]'
        )
        _query_and_collect(query, f"pfam_{pfam_id}")

    return all_records


# ---------------------------------------------------------------------------
# Part C: Evidence Synthesis
# ---------------------------------------------------------------------------

_EVIDENCE_LEVEL_NONE = "none"
_EVIDENCE_LEVEL_WEAK = "weak"
_EVIDENCE_LEVEL_MODERATE = "moderate"
_EVIDENCE_LEVEL_STRONG = "strong"


def _rule_based_evidence_level(has_natural_variant: bool, n_papers: int) -> str:
    if has_natural_variant and n_papers >= 1:
        return _EVIDENCE_LEVEL_STRONG
    if has_natural_variant or n_papers >= 2:
        return _EVIDENCE_LEVEL_MODERATE
    if n_papers == 1:
        return _EVIDENCE_LEVEL_WEAK
    return _EVIDENCE_LEVEL_NONE


def _rule_based_summary(
    mutation: dict,
    natural_variants: List[dict],
    papers: List[dict],
    evidence_level: str,
) -> str:
    """Produce a human-readable evidence summary without using an LLM."""
    parts = []
    mut_label = mutation.get("mutation", "")
    position = mutation.get("position", "")

    if natural_variants:
        sources = list({v.get("source", "UniProt") for v in natural_variants})
        mut_types = list({f"{v.get('wt', '?')}{v.get('position', position)}{v.get('mut', '?')}"
                          for v in natural_variants})
        parts.append(
            f"Natural variant(s) at position {position} documented in "
            f"{', '.join(sources)} ({', '.join(mut_types)})."
        )
        # Note disease associations
        diseases = [v.get("disease", "") for v in natural_variants if v.get("disease")]
        if diseases:
            parts.append(f"Associated with: {'; '.join(diseases)}.")

    if papers:
        n = len(papers)
        years = sorted({p.get("year", 0) for p in papers if p.get("year")}, reverse=True)
        year_range = f"{years[-1]}-{years[0]}" if len(years) > 1 else str(years[0]) if years else ""
        parts.append(
            f"{n} publication{'s' if n != 1 else ''} found for "
            f"{'position ' + str(position) + ' in ' if position else ''}"
            f"this protein" + (f" ({year_range})" if year_range else "") + "."
        )

    if not parts:
        parts.append(f"No experimental literature evidence found for mutation {mut_label}.")

    return " ".join(parts)


def _llm_synthesize(
    mutation: dict,
    natural_variants: List[dict],
    papers: List[dict],
    llm_client: Any,
) -> str:
    """Ask the LLM to synthesise evidence; fall back to rule-based on error."""
    mut_label = mutation.get("mutation", mutation.get("mut", ""))
    position = mutation.get("position", "")

    variant_text = ""
    if natural_variants:
        vlines = []
        for v in natural_variants[:5]:  # cap to avoid token explosion
            vlines.append(
                f"  - {v.get('wt', '?')}{v.get('position', '')}{v.get('mut', '?')}"
                f" | source={v.get('source', '')} | significance={v.get('clinical_significance', '')}"
                f" | disease={v.get('disease', '') or 'none'}"
            )
        variant_text = "UniProt natural variants:\n" + "\n".join(vlines)
    else:
        variant_text = "No UniProt natural variants at this position."

    paper_text = ""
    if papers:
        plines = []
        for p in papers[:5]:
            abstract_snippet = (p.get("abstract") or "")[:300]
            plines.append(
                f"  PMID {p.get('pmid', '?')} ({p.get('year', '?')}): "
                f"{p.get('title', '')} | {abstract_snippet}"
            )
        paper_text = "PubMed papers:\n" + "\n".join(plines)
    else:
        paper_text = "No PubMed papers found."

    prompt = (
        f"Summarise the experimental evidence for the protein mutation {mut_label} "
        f"(position {position}) in 2-3 concise sentences. Use only the data provided below.\n\n"
        f"{variant_text}\n\n{paper_text}\n\n"
        "Focus on: whether the variant is known naturally, what diseases or phenotypes are "
        "linked, and whether published mutagenesis studies confirm functional impact. "
        "If evidence is absent, state that clearly. Do not speculate beyond the data."
    )

    try:
        if hasattr(llm_client, "chat"):
            response = llm_client.chat(prompt)
        elif callable(llm_client):
            response = llm_client(prompt)
        else:
            raise TypeError(f"Unrecognised llm_client type: {type(llm_client)}")

        if isinstance(response, dict):
            return response.get("content", response.get("text", str(response)))
        return str(response).strip()
    except Exception as exc:
        logger.warning("LLM synthesis failed, falling back to rule-based: %s", exc)
        return ""


def synthesize_evidence(
    mutation: dict,
    variants: Dict[int, List[dict]],
    papers: List[dict],
    llm_client: Optional[Any] = None,
) -> dict:
    """Produce a structured evidence summary for a single mutation.

    Parameters
    ----------
    mutation : dict
        Must contain at minimum ``'position'`` (int), ``'wt'`` (str),
        ``'mut'`` (str).  May also contain ``'mutation'`` (e.g. ``'G59I'``).
    variants : dict
        Output of :func:`get_uniprot_variants` — ``{position: [records]}``.
    papers : list
        Output of :func:`search_pubmed_mutations`.
    llm_client : optional
        If provided, used to synthesise a narrative summary.  Should expose
        a ``.chat(prompt)`` method or be callable with a string prompt.

    Returns
    -------
    dict
        Keys: ``has_natural_variant``, ``natural_variants``, ``n_papers``,
        ``papers``, ``evidence_level``, ``evidence_summary``.
    """
    position = int(mutation.get("position", 0))
    natural_variants = variants.get(position, [])
    has_natural_variant = len(natural_variants) > 0
    n_papers = len(papers)

    evidence_level = _rule_based_evidence_level(has_natural_variant, n_papers)

    if llm_client is not None:
        llm_text = _llm_synthesize(mutation, natural_variants, papers, llm_client)
        evidence_summary = llm_text if llm_text else _rule_based_summary(
            mutation, natural_variants, papers, evidence_level
        )
    else:
        evidence_summary = _rule_based_summary(
            mutation, natural_variants, papers, evidence_level
        )

    return {
        "has_natural_variant": has_natural_variant,
        "natural_variants": natural_variants,
        "n_papers": n_papers,
        "papers": papers,
        "evidence_level": evidence_level,
        "evidence_summary": evidence_summary,
    }


# ---------------------------------------------------------------------------
# Part D: Main entry point
# ---------------------------------------------------------------------------


def validate_mutations(
    mutations: List[dict],
    uniprot_id: str,
    protein_name: str = "",
    pfam_id: Optional[str] = None,
    llm_client: Optional[Any] = None,
    cache_dir: Optional[str] = None,
    max_pubmed_queries: int = 3,
) -> List[dict]:
    """Enrich a list of mutation dicts with literature and variant evidence.

    Parameters
    ----------
    mutations : list of dict
        Each item must have at minimum ``'position'`` (int).  Should also
        have ``'wt'``, ``'mut'``, ``'score'`` (or similar ranking field).
    uniprot_id : str
        UniProt accession for the protein.
    protein_name : str, optional
        Human-readable protein name for PubMed queries.
    pfam_id : str, optional
        Pfam accession for family-level PubMed search.
    llm_client : optional
        LLM client for narrative synthesis.
    cache_dir : str, optional
        Cache directory.
    max_pubmed_queries : int
        Maximum number of *distinct* positions to query PubMed for
        (to limit API calls).  Positions are chosen by order of appearance
        (assumed to be pre-ranked by score descending).

    Returns
    -------
    list of dict
        Same list as *mutations* with additional keys added to each item:
        ``has_natural_variant``, ``natural_variants``, ``n_papers``,
        ``evidence_level``, ``evidence_summary``.
        Original dicts are modified in-place and also returned.
    """
    if not mutations:
        return mutations

    cache_d = _ensure_cache_dir(cache_dir)

    # --- Step 1: Fetch all UniProt variants for this protein in one call ---
    all_positions = [m.get("position") for m in mutations if m.get("position")]
    variants_by_pos: Dict[int, List[dict]] = {}
    if uniprot_id and all_positions:
        try:
            variants_by_pos = get_uniprot_variants(
                uniprot_id,
                positions=all_positions,
                cache_dir=cache_d,
            )
        except Exception as exc:
            logger.warning("UniProt variants fetch failed: %s", exc)

    time.sleep(0.3)

    # --- Step 2: Determine which positions to query PubMed for -------------
    # Take the first max_pubmed_queries unique positions (caller pre-ranked)
    seen_positions: list = []
    for m in mutations:
        pos = m.get("position")
        if pos and pos not in seen_positions:
            seen_positions.append(pos)
        if len(seen_positions) >= max_pubmed_queries:
            break

    # --- Step 3: Per-position PubMed lookup; cache at position granularity -
    papers_by_pos: Dict[int, List[dict]] = {}
    for pos in seen_positions:
        try:
            papers = search_pubmed_mutations(
                protein_name=protein_name,
                positions=[pos],
                pfam_id=pfam_id,
                max_results=5,
                cache_dir=cache_d,
            )
            papers_by_pos[pos] = papers
        except Exception as exc:
            logger.warning("PubMed search failed for position %s: %s", pos, exc)
            papers_by_pos[pos] = []
        time.sleep(0.3)

    # Also do a broad protein + Pfam search for positions NOT individually queried
    broad_papers: List[dict] = []
    if protein_name or pfam_id:
        try:
            broad_papers = search_pubmed_mutations(
                protein_name=protein_name,
                positions=None,
                pfam_id=pfam_id,
                max_results=5,
                cache_dir=cache_d,
            )
        except Exception as exc:
            logger.warning("Broad PubMed search failed: %s", exc)

    # --- Step 4: Synthesise evidence per mutation --------------------------
    for mutation in mutations:
        pos = mutation.get("position")

        # Merge position-specific papers with broad papers (deduplicate by PMID)
        pos_papers = papers_by_pos.get(pos, broad_papers)
        seen_pmids: set = set()
        merged_papers: List[dict] = []
        for p in pos_papers + broad_papers:
            pmid = p.get("pmid", "")
            if pmid not in seen_pmids:
                seen_pmids.add(pmid)
                merged_papers.append(p)

        try:
            evidence = synthesize_evidence(
                mutation=mutation,
                variants=variants_by_pos,
                papers=merged_papers,
                llm_client=llm_client,
            )
        except Exception as exc:
            logger.warning(
                "Evidence synthesis failed for mutation at pos %s: %s", pos, exc
            )
            evidence = {
                "has_natural_variant": False,
                "natural_variants": [],
                "n_papers": 0,
                "evidence_level": _EVIDENCE_LEVEL_NONE,
                "evidence_summary": f"Evidence synthesis error: {exc}",
            }

        # Enrich the mutation dict in-place
        mutation["has_natural_variant"] = evidence["has_natural_variant"]
        mutation["natural_variants"] = evidence["natural_variants"]
        mutation["n_papers"] = evidence["n_papers"]
        mutation["evidence_level"] = evidence["evidence_level"]
        mutation["evidence_summary"] = evidence["evidence_summary"]

    return mutations
