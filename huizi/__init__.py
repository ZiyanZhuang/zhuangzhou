"""
HuiZi — Zhuangzhou's research context companion
====================================
Named after Zhuangzi's sparring partner: precise, logical, context-aware.

HuiZi maintains session-level awareness:
- What protein is being analyzed (identity, function, sequence)
- What ligand is involved (name, SMILES)
- What analyses have been done
- Provides rich context for report generation
- Translates Chinese academic queries to English keywords
- Fetches supporting literature automatically

Usage:
    from zhuangzhou.huizi import HuiZi

    hz = HuiZi()
    hz.set_protein(pdb_path="WT.pdb")
    ctx = hz.get_report_context()
"""

import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from .context import SessionContext, AnalysisTarget
from .protein_resolver import ProteinResolver
from .query_translator import QueryTranslator
from .literature_fetcher import LiteratureFetcher
from .structure_fetcher import StructureFetcher


class HuiZi:
    """
    Research-context awareness object for Zhuangzhou.

    HuiZi tracks the current analysis session and provides rich context
    to downstream modules (report generation, GraphRAG queries, etc.)
    """

    def __init__(self, llm_backend=None, session_id: str = "default"):
        self.session = SessionContext(session_id=session_id)
        self.resolver = ProteinResolver()
        self.translator = QueryTranslator(llm_backend=llm_backend)
        self.lit_fetcher = LiteratureFetcher()
        self.structure_fetcher = StructureFetcher()
        self._llm_backend = llm_backend

    # ── Protein context ──────────────────────────────────────────────────────

    def set_protein(self, pdb_path=None, name=None, uniprot_id=None,
                    sequence=None, organism=None, auto_resolve=True):
        """Set or update the current protein being analyzed."""
        target = self.session.target
        if pdb_path:
            target.pdb_path = pdb_path
        if name:
            target.protein_name = name
        if uniprot_id:
            target.uniprot_id = uniprot_id
        if sequence:
            target.fasta_sequence = sequence
        if organism:
            target.organism = organism

        if auto_resolve and (pdb_path or name or uniprot_id):
            self._auto_resolve()

    def set_ligand(self, name=None, smiles=None, pubchem_cid=None):
        """Set or update the current ligand."""
        t = self.session.target
        if name:
            t.ligand_name = name
        if smiles:
            t.ligand_smiles = smiles
        if pubchem_cid:
            t.ligand_pubchem_cid = pubchem_cid

    def record_analysis(self, analysis_type: str, result_path: str = None,
                        key_finding: str = None):
        """Record a completed analysis step."""
        t = self.session.target
        if analysis_type and analysis_type not in t.analyses:
            t.analyses.append(analysis_type)
        if result_path and analysis_type:
            t.result_paths[analysis_type] = result_path
        if key_finding:
            t.key_findings.append(key_finding)

    # ── Context extraction ───────────────────────────────────────────────────

    def get_report_context(self, fetch_literature: bool = False) -> dict:
        """
        Build comprehensive context dict for report generation.
        """
        target = self.session.target
        ctx = target.to_dict()

        if fetch_literature and target.protein_name:
            try:
                papers = self.lit_fetcher.fetch(
                    target.protein_name,
                    target.analyses or ['structure'],
                    max_results=5,
                )
                ctx['literature'] = papers
            except Exception:
                ctx['literature'] = []
        else:
            ctx['literature'] = []

        return ctx

    def get_system_prompt_block(self) -> str:
        """Compact context block for injection into agent system prompt."""
        t = self.session.target
        if not t.has_any():
            return ""

        lines = ["## HuiZi Current Analysis Target (Session Context)"]
        if t.protein_name:
            org = f" ({t.organism})" if t.organism else ""
            lines.append(f"- Protein: {t.protein_name}{org}")
        if t.uniprot_id:
            lines.append(f"- UniProt: {t.uniprot_id}")
        if t.ec_number:
            lines.append(f"- EC Number: {t.ec_number}")
        if t.pdb_path:
            src_label = ''
            if t.structure_source == 'afdb':
                src_label = ' [AlphaFold DB prediction]'
            elif t.structure_source == 'esmatlas':
                src_label = ' [ESMFold prediction]'
            plddt_label = f', pLDDT={t.plddt_mean:.0f}' if t.plddt_mean else ''
            lines.append(f"- PDB File: {t.pdb_path}{src_label}{plddt_label}")
        if t.fasta_sequence:
            lines.append(f"- Sequence Length: {len(t.fasta_sequence)} aa")
        if t.ligand_name:
            smiles_info = f" (SMILES: {t.ligand_smiles})" if t.ligand_smiles else ""
            lines.append(f"- Ligand: {t.ligand_name}{smiles_info}")
        if t.analyses:
            lines.append(f"- Completed Analyses: {', '.join(t.analyses)}")
        if t.function_description:
            short = t.function_description[:200]
            if len(t.function_description) > 200:
                short += "..."
            lines.append(f"- Function: {short}")
        if t.key_findings:
            lines.append(f"- Key Findings: {'; '.join(t.key_findings[:3])}")

        return "\n".join(lines)

    # ── Query translation ────────────────────────────────────────────────────

    def translate_query(self, query: str) -> dict:
        """Translate Chinese scientific query to English keywords."""
        return self.translator.translate(query)

    def ensure_structure(self, output_dir: str = None, force_refresh: bool = False) -> dict:
        """
        Ensure the current analysis target has a PDB structure.
        If no PDB path is set, automatically fetch from AFDB or ESMFold.

        Priority:
          1. Already have pdb_path -> verify file exists -> return it
          2. Have UniProt ID -> fetch from AFDB
          3. Have sequence -> fold with ESMFold
          4. Have protein name -> try to resolve UniProt first then fetch

        Returns:
            {'pdb_path': str, 'source': str, 'plddt_mean': float, ...}
        """
        t = self.session.target

        # 1. Already have a valid PDB
        if t.pdb_path and os.path.exists(t.pdb_path):
            # Mark as experimental/user-provided if not already tagged
            if not t.structure_source:
                t.structure_source = 'experimental'
            return {
                'pdb_path': t.pdb_path,
                'source': t.structure_source,
                'source_label': 'User-provided / experimental structure',
                'plddt_mean': t.plddt_mean,
                'quality_tier': t.structure_quality_tier or 'unknown',
                'already_available': True,
            }

        # 2. Resolve UniProt ID if we don't have one
        if not t.uniprot_id and t.protein_name:
            try:
                info = self.resolver.from_name(t.protein_name, t.organism)
                self._merge(info)
            except Exception:
                pass

        # 3. Try AFDB with UniProt ID
        result = self.structure_fetcher.get_structure(
            uniprot_id=t.uniprot_id,
            sequence=t.fasta_sequence,
            protein_name=t.protein_name,
            output_dir=output_dir,
            force_refresh=force_refresh,
        )

        if result and not result.get('error'):
            # Update session context
            t.pdb_path = result['pdb_path']
            t.structure_source = result['source']
            t.plddt_mean = result.get('plddt_mean')
            t.structure_quality_tier = result.get('quality_tier')
            self.session.save()

            # Also extract FASTA if we don't have it
            if not t.fasta_sequence and result.get('pdb_path'):
                try:
                    fasta = self.resolver.extract_fasta_from_pdb(result['pdb_path'])
                    if fasta:
                        t.fasta_sequence = fasta
                except Exception:
                    pass

            return result

        return result  # Return error dict as-is

    # ── Internal ─────────────────────────────────────────────────────────────

    def _auto_resolve(self):
        """Fill missing protein fields from available data."""
        t = self.session.target
        try:
            if t.pdb_path:
                info = self.resolver.from_pdb_file(t.pdb_path)
                self._merge(info)
            elif t.uniprot_id:
                info = self.resolver.from_uniprot_id(t.uniprot_id)
                self._merge(info)
            elif t.protein_name:
                info = self.resolver.from_name(t.protein_name, t.organism)
                self._merge(info)
        except Exception:
            pass

    def _merge(self, info: dict):
        """Merge resolved info into target (only fill gaps, never overwrite)."""
        t = self.session.target
        mapping = {
            'protein_name': 'protein_name',
            'gene_name': 'protein_gene',
            'organism': 'organism',
            'uniprot_id': 'uniprot_id',
            'fasta_sequence': 'fasta_sequence',
            'function': 'function_description',
            'ec_number': 'ec_number',
            'pdb_id': 'pdb_id',
            'active_site_residues': 'active_site_residues',
        }
        for src, dst in mapping.items():
            if info.get(src) and not getattr(t, dst, None):
                setattr(t, dst, info[src])


__all__ = ['HuiZi', 'SessionContext', 'AnalysisTarget',
           'ProteinResolver', 'QueryTranslator', 'LiteratureFetcher',
           'StructureFetcher']
