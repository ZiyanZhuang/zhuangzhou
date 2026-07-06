"""
StructureFetcher — automatically obtain protein 3D structures from:
  1. Local cache (~/.zhuangzhou/structures/)
  2. AlphaFold Database (AFDB) by UniProt ID
  3. ESMFold public API (by sequence, ~400aa limit)
  4. Workspace scan (*.pdb files in current directory)

All methods return a standardized result dict, never raise.

pLDDT quality thresholds (AlphaFold convention):
  >= 90: Very high confidence (dark blue)
  70-90: Confident (light blue)
  50-70: Low confidence (yellow) — treat with caution
  < 50:  Very low (orange) — disordered/unreliable
"""
import os
import re
import json
import time
import urllib.request
import urllib.parse
from typing import Optional, List, Dict


_HEADERS = {'User-Agent': 'Zhuangzhou-HuiZi/1.0'}
_TIMEOUT = 30

_AFDB_API = "https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
_AFDB_PDB_TEMPLATE = "https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
_ESMATLAS_URL = "https://api.esmatlas.com/foldSequence/v1/pdb/"

_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".zhuangzhou", "structures")


def _get_text(url: str, timeout: int = _TIMEOUT) -> Optional[str]:
    try:
        req = urllib.request.Request(url, headers=_HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode('utf-8')
    except Exception:
        return None


def _get_json(url: str, timeout: int = 15) -> Optional[dict]:
    try:
        req = urllib.request.Request(url, headers={**_HEADERS, 'Accept': 'application/json'})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode('utf-8'))
    except Exception:
        return None


def _post_text(url: str, data: bytes, headers: dict, timeout: int = _TIMEOUT) -> Optional[str]:
    try:
        req = urllib.request.Request(url, data=data, headers=headers, method='POST')
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode('utf-8')
    except Exception:
        return None


class StructureFetcher:
    """
    Automatically obtain protein 3D structures from multiple sources.

    Priority:
      1. Local cache (instant)
      2. AlphaFold Database (high quality, ~1s download)
      3. ESMFold API (sequence-based, slower, ~30s)
    """

    def __init__(self, cache_dir: str = None):
        self._cache_dir = cache_dir or _CACHE_DIR
        os.makedirs(self._cache_dir, exist_ok=True)

    # ── Main entry ───────────────────────────────────────────────────────────

    def get_structure(self, uniprot_id: str = None, sequence: str = None,
                      protein_name: str = None, output_dir: str = None,
                      force_refresh: bool = False) -> dict:
        """
        Get protein structure from best available source.

        Args:
            uniprot_id: UniProt accession (e.g. 'P0A3E0') — enables AFDB lookup
            sequence: Amino acid sequence string — enables ESMFold folding
            protein_name: Human-readable name for file naming
            output_dir: Where to save a copy of the PDB (besides cache)
            force_refresh: Skip cache and re-download

        Returns:
            {
                'pdb_path': str,              # Absolute path to PDB file
                'source': str,                # 'afdb'|'esmatlas'|'cache'
                'source_label': str,          # Human-readable source description
                'uniprot_id': str,
                'afdb_version': str,          # e.g. 'v4'
                'plddt_mean': float|None,
                'plddt_per_residue': list|None,
                'quality_summary': str,       # Text description of quality
                'quality_tier': str,          # 'high'|'confident'|'low'|'very_low'
            }
        """
        # Try AFDB first (requires UniProt ID)
        if uniprot_id:
            cached = self._cache_path(uniprot_id, 'afdb')
            if os.path.exists(cached) and not force_refresh:
                result = self._make_result_from_cached(cached, uniprot_id, 'afdb')
                if output_dir:
                    self._copy_to_output(cached, output_dir, protein_name or uniprot_id)
                return result

            result = self.fetch_from_afdb(uniprot_id, output_dir=output_dir,
                                           protein_name=protein_name)
            if result and not result.get('error'):
                return result

        # Try ESMFold (requires sequence)
        if sequence:
            seq_hash = f"{len(sequence)}_{sequence[:10]}"
            cached_key = protein_name or seq_hash
            cached = self._cache_path(cached_key, 'esmatlas')
            if os.path.exists(cached) and not force_refresh:
                result = self._make_result_from_cached(cached, cached_key, 'esmatlas')
                if output_dir:
                    self._copy_to_output(cached, output_dir, cached_key)
                return result

            result = self.fold_with_esmatlas(sequence, protein_name=protein_name,
                                              output_dir=output_dir)
            if result and not result.get('error'):
                return result

        return {
            'error': 'Could not obtain protein structure',
            'hint': 'Provide uniprot_id for AFDB lookup or sequence for ESMFold',
            'tried': ['afdb' if uniprot_id else None, 'esmatlas' if sequence else None],
        }

    # ── AlphaFold Database ───────────────────────────────────────────────────

    def fetch_from_afdb(self, uniprot_id: str, output_dir: str = None,
                        protein_name: str = None) -> dict:
        """
        Download predicted structure from AlphaFold Database.

        AFDB API: https://alphafold.ebi.ac.uk/api/prediction/{UNIPROT_ID}
        PDB URL:  https://alphafold.ebi.ac.uk/files/AF-{id}-F1-model_v4.pdb
        pLDDT:    Stored in PDB B-factor column (CA atoms)
        """
        if not uniprot_id:
            return {'error': 'uniprot_id required for AFDB lookup'}

        # Query API for metadata (version info)
        api_url = _AFDB_API.format(uniprot_id=uniprot_id)
        meta_list = _get_json(api_url, timeout=10)

        # Build PDB download URL
        pdb_url = None
        afdb_version = 'v4'  # Default to current version
        if meta_list and isinstance(meta_list, list) and meta_list:
            meta = meta_list[0]
            pdb_url = meta.get('pdbUrl') or meta.get('cifUrl', '').replace('.cif', '.pdb')
            # Extract version from URL
            v_match = re.search(r'model_(v\d+)', pdb_url or '')
            if v_match:
                afdb_version = v_match.group(1)

        if not pdb_url:
            # Fallback to known URL pattern
            pdb_url = _AFDB_PDB_TEMPLATE.format(uniprot_id=uniprot_id)

        # Download PDB
        pdb_text = _get_text(pdb_url, timeout=30)
        if not pdb_text or not pdb_text.strip().startswith(('ATOM', 'REMARK', 'HEADER', 'TITLE')):
            return {
                'error': f'AFDB download failed for {uniprot_id}',
                'hint': f'Check if {uniprot_id} exists in AlphaFold Database',
                'tried_url': pdb_url,
            }

        # Save to cache
        cache_path = self._cache_path(uniprot_id, 'afdb')
        self._write_pdb(pdb_text, cache_path)

        # Save metadata alongside cache
        meta_path = cache_path.replace('.pdb', '_meta.json')
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump({
                'uniprot_id': uniprot_id,
                'source': 'afdb',
                'version': afdb_version,
                'pdb_url': pdb_url,
                'downloaded_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            }, f, indent=2)

        # Copy to output_dir if requested
        output_path = cache_path
        if output_dir:
            output_path = self._copy_to_output(cache_path, output_dir,
                                                protein_name or uniprot_id)

        # Extract pLDDT from B-factors
        plddt = self.extract_plddt_from_pdb(cache_path)

        return {
            'pdb_path': os.path.abspath(output_path),
            'source': 'afdb',
            'source_label': f'AlphaFold Database ({afdb_version})',
            'uniprot_id': uniprot_id,
            'afdb_version': afdb_version,
            'plddt_mean': plddt.get('mean'),
            'plddt_per_residue': plddt.get('per_residue'),
            'quality_summary': self._quality_summary(plddt),
            'quality_tier': self._quality_tier(plddt.get('mean')),
        }

    # ── ESMFold API ──────────────────────────────────────────────────────────

    def fold_with_esmatlas(self, sequence: str, protein_name: str = None,
                           output_dir: str = None) -> dict:
        """
        Fold sequence using ESMFold public API (ESM Atlas).

        Endpoint: POST https://api.esmatlas.com/foldSequence/v1/pdb/
        Body: raw sequence string (plain text, NOT form-encoded)
        Limit: ~400 amino acids for reliable results
        Note: Rate-limited; may be slow or unavailable at times
        """
        if not sequence:
            return {'error': 'sequence required for ESMFold'}

        seq_len = len(sequence)
        if seq_len > 400:
            # Warn but still try — may work for some longer sequences
            pass

        # POST sequence to ESMAtlas
        # ESMAtlas API 仅支持完整序列，不截断（截断序列会破坏结构完整性，
        # 产生的坐标无法与原始残基编号对应，不符合学术规范）
        headers = {
            'User-Agent': 'Zhuangzhou-HuiZi/1.0',
            'Content-Type': 'application/x-www-form-urlencoded',
        }
        clean_seq = sequence.strip().upper()
        body = clean_seq.encode('ascii')
        pdb_text = _post_text(_ESMATLAS_URL, body, headers, timeout=120)

        if not pdb_text:
            return {
                'error': f'ESMFold API failed for sequence of {seq_len} aa '
                         f'(may exceed API limit or be rate-limited)',
                'hint': 'Use a local structure file or AFDB (requires UniProt ID). '
                        'ESMAtlas API is limited to ~400 aa.',
                'sequence_length': seq_len,
            }

        if not pdb_text.strip().startswith(('ATOM', 'REMARK', 'HEADER', 'MODEL')):
            return {
                'error': 'ESMFold returned unexpected response',
                'hint': 'Check sequence contains only standard amino acids',
                'response_preview': pdb_text[:200],
            }

        # Save to cache
        cache_key = protein_name or f"seq_{seq_len}aa"
        cache_path = self._cache_path(cache_key, 'esmatlas')
        self._write_pdb(pdb_text, cache_path)

        # Save metadata
        meta_path = cache_path.replace('.pdb', '_meta.json')
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump({
                'protein_name': protein_name,
                'source': 'esmatlas',
                'sequence_length': seq_len,
                'folded_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            }, f, indent=2)

        # Copy to output_dir
        output_path = cache_path
        if output_dir:
            output_path = self._copy_to_output(cache_path, output_dir, cache_key)

        plddt = self.extract_plddt_from_pdb(cache_path)

        return {
            'pdb_path': os.path.abspath(output_path),
            'source': 'esmatlas',
            'source_label': 'ESMFold (ESM Atlas API)',
            'uniprot_id': None,
            'afdb_version': None,
            'sequence_length': seq_len,
            'plddt_mean': plddt.get('mean'),
            'plddt_per_residue': plddt.get('per_residue'),
            'quality_summary': self._quality_summary(plddt),
            'quality_tier': self._quality_tier(plddt.get('mean')),
        }

    # ── pLDDT extraction ─────────────────────────────────────────────────────

    def extract_plddt_from_pdb(self, pdb_path: str) -> dict:
        """
        Extract pLDDT scores from B-factor column of CA atoms.

        AlphaFold and ESMFold store per-residue confidence (pLDDT) in B-factors.
        Returns: {'mean': float, 'per_residue': list, 'high_confidence': int, 'total': int}
        """
        if not pdb_path or not os.path.exists(pdb_path):
            return {}

        scores: List[float] = []
        try:
            with open(pdb_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if (line.startswith('ATOM') and
                            line[12:16].strip() == 'CA'):
                        try:
                            bfactor = float(line[60:66].strip())
                            scores.append(bfactor)
                        except ValueError:
                            continue
        except Exception:
            return {}

        if not scores:
            return {}

        mean_plddt = sum(scores) / len(scores)
        high_conf = sum(1 for s in scores if s >= 70)

        return {
            'mean': round(mean_plddt, 1),
            'per_residue': [round(s, 1) for s in scores],
            'high_confidence': high_conf,
            'total': len(scores),
            'high_confidence_pct': round(100 * high_conf / len(scores), 1),
        }

    # ── Workspace scan ───────────────────────────────────────────────────────

    def scan_workspace(self, directory: str = None) -> List[str]:
        """
        Scan a directory for PDB files, preferring AlphaFold/ESMFold predictions.
        Returns list of PDB file paths found.
        """
        search_dir = directory or os.getcwd()
        pdb_files = []
        try:
            for fname in os.listdir(search_dir):
                if fname.lower().endswith('.pdb'):
                    pdb_files.append(os.path.join(search_dir, fname))
        except Exception:
            pass
        # Sort: prefer files with 'afdb', 'alphafold', 'esm' in name
        def _priority(p):
            name = os.path.basename(p).lower()
            if any(k in name for k in ('afdb', 'alphafold', 'af-')):
                return 0
            if any(k in name for k in ('esm', 'fold')):
                return 1
            return 2
        return sorted(pdb_files, key=_priority)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _cache_path(self, key: str, source: str) -> str:
        """Get cache file path for a structure."""
        # Sanitize key for filesystem
        safe_key = re.sub(r'[^\w\-]', '_', key)
        filename = f"{safe_key}_{source}.pdb"
        return os.path.join(self._cache_dir, filename)

    def _write_pdb(self, pdb_text: str, path: str):
        """Write PDB text to file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(pdb_text)

    def _copy_to_output(self, src_path: str, output_dir: str, name: str) -> str:
        """Copy PDB to output directory with a descriptive name."""
        import shutil
        os.makedirs(output_dir, exist_ok=True)
        safe_name = re.sub(r'[^\w\-]', '_', name)
        dest = os.path.join(output_dir, f"{safe_name}_structure.pdb")
        try:
            shutil.copy2(src_path, dest)
            return dest
        except Exception:
            return src_path

    def _make_result_from_cached(self, cache_path: str, key: str, source: str) -> dict:
        """Build result dict from cached PDB (load metadata if available)."""
        meta_path = cache_path.replace('.pdb', '_meta.json')
        meta = {}
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
            except Exception:
                pass

        plddt = self.extract_plddt_from_pdb(cache_path)
        labels = {
            'afdb': f"AlphaFold Database ({meta.get('version', 'v4')}) [cached]",
            'esmatlas': "ESMFold (ESM Atlas API) [cached]",
        }

        return {
            'pdb_path': os.path.abspath(cache_path),
            'source': source,
            'source_label': labels.get(source, f'{source} [cached]'),
            'uniprot_id': meta.get('uniprot_id', key),
            'afdb_version': meta.get('version'),
            'plddt_mean': plddt.get('mean'),
            'plddt_per_residue': plddt.get('per_residue'),
            'quality_summary': self._quality_summary(plddt),
            'quality_tier': self._quality_tier(plddt.get('mean')),
            'from_cache': True,
        }

    def _quality_summary(self, plddt: dict) -> str:
        """Generate human-readable quality summary."""
        if not plddt or plddt.get('mean') is None:
            return 'Quality information not available'

        mean = plddt['mean']
        total = plddt.get('total', 0)
        high_pct = plddt.get('high_confidence_pct', 0)

        tier = self._quality_tier(mean)
        tier_desc = {
            'high': '非常高置信度 (Very high confidence)',
            'confident': '高置信度 (Confident)',
            'low': '低置信度 (Low confidence — use with caution)',
            'very_low': '极低置信度 (Very low — likely disordered)',
        }.get(tier, '')

        summary = (
            f"平均pLDDT: {mean:.1f}/100 | {tier_desc} | "
            f"共{total}残基，{high_pct:.0f}%高置信度(>=70)"
        )

        # Docking warning
        if mean < 70:
            summary += (
                "\n注意：平均pLDDT<70，预测置信度较低，"
                "建议谨慎用于对接分析；活性位点区域可能不可靠。"
            )

        return summary

    def _quality_tier(self, mean_plddt: Optional[float]) -> str:
        if mean_plddt is None:
            return 'unknown'
        if mean_plddt >= 90:
            return 'high'
        elif mean_plddt >= 70:
            return 'confident'
        elif mean_plddt >= 50:
            return 'low'
        else:
            return 'very_low'
