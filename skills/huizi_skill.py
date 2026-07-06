"""
HuiZi Skills — register HuiZi context capabilities as agent-callable skills.

Bug fixes applied (v0.8.6):
  - SkillOutput does not exist in registry → use SkillParam for outputs
  - Skill field is 'fn' not 'execute'
  - SkillParam positional order: (name, type, description, required=True)
  - Type strings: "string" not "str" (per registry convention)
"""
import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def register_huizi_skills(registry):
    """Register HuiZi context skills. Call after skill registry is initialized."""
    try:
        from .registry import Skill, SkillParam
    except ImportError:
        return

    # ── set_analysis_target ────────────────────────────────────────────────
    registry.register(Skill(
        name="set_analysis_target",
        description=(
            "Set context for the current analysis target (protein name, PDB file path, ligand, etc.). "
            "HuiZi automatically parses protein name/species from PDB files and queries UniProt for function descriptions. "
            "Call this skill before starting any protein analysis."
        ),
        category="context",
        fn=_exec_set_target,
        inputs=[
            SkillParam("protein_name", "string",
                       "Protein name, e.g. 'Nicotinamidase', 'CYP3A4', 'CYP714E19'",
                       required=False),
            SkillParam("pdb_path", "path",
                       "PDB file path; HuiZi will auto-extract protein information",
                       required=False, is_output_path=False),
            SkillParam("organism", "string",
                       "Organism, e.g. 'Mycobacterium tuberculosis', 'Arabidopsis thaliana'",
                       required=False),
            SkillParam("uniprot_id", "string",
                       "UniProt accession, e.g. 'P0A3E0'",
                       required=False),
            SkillParam("ligand_name", "string",
                       "Ligand/small-molecule name",
                       required=False),
            SkillParam("ligand_smiles", "string",
                       "Ligand SMILES string",
                       required=False),
            SkillParam("auto_resolve", "bool",
                       "Whether to auto-query UniProt/PDB to complete protein info (default: True)",
                       required=False, default=True),
        ],
        outputs=[
            SkillParam("context_summary", "string", "Summary of the current analysis target"),
            SkillParam("protein_info", "dict", "Resolved full protein information"),
        ],
        when_to_use=(
            "ALWAYS call at the start when user mentions a protein name, PDB file path, "
            "or ligand. Call BEFORE esm2_scan, smart_docking, enrich_mutations, or "
            "generate_report. This ensures reports say the correct protein name instead "
            "of 'Unknown Protein'."
        ),
        estimated_time="5-30s",
        workflow_stage="prepare",
    ))

    # ── get_analysis_context ───────────────────────────────────────────────
    registry.register(Skill(
        name="get_analysis_context",
        description=(
            "Retrieve HuiZi's current analysis context, including full protein info, ligand SMILES, "
            "completed analysis types, key findings, and related literature. Must be called before report generation."
        ),
        category="context",
        fn=_exec_get_context,
        inputs=[
            SkillParam("fetch_literature", "bool",
                       "Whether to automatically retrieve related PubMed literature (default: True; network required)",
                       required=False, default=True),
        ],
        outputs=[
            SkillParam("context", "dict",
                       "Complete analysis context, used as generate_report.protein_context"),
            SkillParam("report_ready", "bool", "Whether enough information is available to generate a report"),
            SkillParam("summary", "string", "One-line summary"),
        ],
        when_to_use=(
            "Call BEFORE generate_report. Pass the returned 'context' as the "
            "'protein_context' parameter of generate_report."
        ),
        workflow_stage="report",
    ))

    # ── translate_scientific_query ─────────────────────────────────────────
    registry.register(Skill(
        name="translate_scientific_query",
        description=(
            "Translate Chinese scientific queries into English academic keywords for GraphRAG knowledge-graph retrieval. "
            "Supports domain terminology in protein engineering, structural biology, and drug discovery."
        ),
        category="context",
        fn=_exec_translate,
        inputs=[
            SkillParam("query", "string",
                       "Chinese query text, e.g. 'catalytic mechanism and substrate binding of nicotinamidase'",
                       required=True),
        ],
        outputs=[
            SkillParam("keywords", "list", "English keyword list (directly usable in graphrag_query)"),
            SkillParam("translated", "string", "Full English translation"),
            SkillParam("domain", "string", "Detected scientific domain"),
        ],
        when_to_use=(
            "Call BEFORE graphrag_query when the user's question is in Chinese. "
            "Use the returned 'keywords' list as search terms for the knowledge graph."
        ),
        workflow_stage="fetch",
    ))

    # ── record_key_finding ─────────────────────────────────────────────────
    registry.register(Skill(
        name="record_key_finding",
        description=(
            "Record key findings into HuiZi context; these findings will appear in the report conclusion."
        ),
        category="context",
        fn=_exec_record_finding,
        inputs=[
            SkillParam("finding", "string",
                       "Key finding description, e.g. 'G59I mutation ΔLL=2.04, top beneficial mutation'",
                       required=True),
            SkillParam("analysis_type", "string",
                       "Analysis type, e.g. 'esm2_scan', 'docking', 'plip'",
                       required=False),
            SkillParam("result_path", "string",
                       "Related result file path",
                       required=False),
        ],
        outputs=[
            SkillParam("total_findings", "int", "Total number of currently recorded findings"),
        ],
        workflow_stage="analyze",
    ))

    # ── reset_analysis_context ─────────────────────────────────────────────
    registry.register(Skill(
        name="reset_analysis_context",
        description="Clear HuiZi's current analysis context; use this when starting a new protein analysis.",
        category="context",
        fn=_exec_reset,
        inputs=[],
        outputs=[
            SkillParam("status", "string", "Operation status"),
        ],
    ))

    # ── fetch_protein_structure ────────────────────────────────────────────
    registry.register(Skill(
        name="fetch_protein_structure",
        description=(
            "Automatically fetch a protein 3D structure file (PDB). "
            "Prefer downloading from AlphaFold DB (AFDB); if UniProt ID is unavailable, predict from sequence via ESMFold API. "
            "Results are cached to ~/.zhuangzhou/structures/ and HuiZi context is updated."
        ),
        category="structure",
        fn=_exec_fetch_structure,
        inputs=[
            SkillParam("uniprot_id", "string",
                       "UniProt accession (recommended), e.g. 'P0A3E0', 'Q9Y255'. "
                       "With a UniProt ID, fetch a high-quality predicted structure from AFDB.",
                       required=False),
            SkillParam("sequence", "string",
                       "Amino-acid sequence. If UniProt ID is unavailable, fold with ESMFold; recommended length <= 400 aa.",
                       required=False),
            SkillParam("protein_name", "string",
                       "Protein name, used for file naming and context updates",
                       required=False),
            SkillParam("output_dir", "path",
                       "Directory to save the structure file. By default, save to cache only.",
                       required=False, is_output_path=True),
            SkillParam("force_refresh", "bool",
                       "Whether to force re-download (ignore local cache), default: False",
                       required=False, default=False),
        ],
        outputs=[
            SkillParam("pdb_path", "path", "Fetched PDB file path"),
            SkillParam("source", "string", "Source: 'afdb'|'esmatlas'|'cache'"),
            SkillParam("source_label", "string", "Human-readable source description"),
            SkillParam("plddt_mean", "float", "Mean pLDDT confidence score (0-100)"),
            SkillParam("quality_tier", "string",
                       "Quality tier: 'high'|'confident'|'low'|'very_low'"),
            SkillParam("quality_summary", "string", "Quality description with usage recommendations"),
        ],
        when_to_use=(
            "Call when the user does not have a local PDB file but wants to analyze a protein. "
            "Use BEFORE smart_docking, esm2_scan, pocket_detection, or structural_analysis. "
            "Also call when only protein name/UniProt ID/sequence is available. "
            "Supports: AFDB (UniProt ID -> ~5s download) and ESMFold (sequence -> ~30s fold)."
        ),
        estimated_time="5-60s (AFDB: ~5s, ESMFold: ~30-60s)",
        workflow_stage="fetch",
        requires_network=True,
    ))


# ── Singleton HuiZi instance ──────────────────────────────────────────────────

_huizi_instance = None


def _get_huizi():
    global _huizi_instance
    if _huizi_instance is None:
        try:
            from zhuangzhou.huizi import HuiZi
            _huizi_instance = HuiZi()
        except Exception as e:
            raise RuntimeError(f"HuiZi init failed: {e}")
    return _huizi_instance


# ── Executors ─────────────────────────────────────────────────────────────────

def _exec_set_target(protein_name=None, pdb_path=None, organism=None,
                     uniprot_id=None, ligand_name=None, ligand_smiles=None,
                     auto_resolve=True, **kwargs):
    try:
        hz = _get_huizi()
        hz.set_protein(
            pdb_path=pdb_path, name=protein_name, uniprot_id=uniprot_id,
            organism=organism, auto_resolve=bool(auto_resolve),
        )
        if ligand_name or ligand_smiles:
            hz.set_ligand(name=ligand_name, smiles=ligand_smiles)
        hz.session.save()
        t = hz.session.target
        return {
            'context_summary': t.summary(),
            'protein_info': {
                'name': t.protein_name,
                'gene': t.protein_gene,
                'organism': t.organism,
                'uniprot_id': t.uniprot_id,
                'ec_number': t.ec_number,
                'function': (t.function_description or '')[:300],
                'fasta_length': len(t.fasta_sequence) if t.fasta_sequence else 0,
                'pdb_path': t.pdb_path,
                'structure_source': t.structure_source,
                'plddt_mean': t.plddt_mean,
            },
            'status': 'ok',
        }
    except Exception as e:
        return {'error': str(e), 'hint': 'You can pass the protein_name parameter directly.'}


def _exec_get_context(fetch_literature=True, **kwargs):
    try:
        hz = _get_huizi()
        ctx = hz.get_report_context(fetch_literature=bool(fetch_literature))
        t = hz.session.target
        return {
            'context': ctx,
            'report_ready': bool(t.protein_name or t.pdb_path),
            'summary': t.summary(),
        }
    except Exception as e:
        return {'error': str(e), 'hint': 'Call set_analysis_target first to set the analysis target.'}


def _exec_translate(query, **kwargs):
    try:
        hz = _get_huizi()
        result = hz.translate_query(query)
        return {
            'keywords': result.get('keywords', []),
            'concepts': result.get('concepts', []),
            'translated': result.get('translated', query),
            'domain': result.get('domain', 'biochemistry'),
            'source': result.get('source', 'dict'),
        }
    except Exception as e:
        return {'error': str(e), 'hint': 'Translation failed. Please use English keywords directly.'}


def _exec_record_finding(finding, analysis_type=None, result_path=None, **kwargs):
    try:
        hz = _get_huizi()
        hz.record_analysis(
            analysis_type=analysis_type or 'general',
            result_path=result_path,
            key_finding=finding,
        )
        hz.session.save()
        return {
            'total_findings': len(hz.session.target.key_findings),
            'status': 'recorded',
        }
    except Exception as e:
        return {'error': str(e)}


def _exec_reset(**kwargs):
    try:
        hz = _get_huizi()
        hz.session.reset()
        return {'status': 'context cleared'}
    except Exception as e:
        return {'error': str(e)}


def _exec_fetch_structure(uniprot_id=None, sequence=None, protein_name=None,
                          output_dir=None, force_refresh=False, **kwargs):
    try:
        hz = _get_huizi()

        # Fill gaps from HuiZi session context
        t = hz.session.target
        if not uniprot_id:
            uniprot_id = t.uniprot_id
        if not sequence:
            sequence = t.fasta_sequence
        if not protein_name:
            protein_name = t.protein_name

        # Try ensure_structure first (uses existing pdb_path or fetches)
        result = hz.ensure_structure(
            output_dir=output_dir,
            force_refresh=bool(force_refresh),
        )

        # If ensure_structure failed and we have explicit params, try direct fetch
        if result.get('error') and (uniprot_id or sequence):
            result = hz.structure_fetcher.get_structure(
                uniprot_id=uniprot_id,
                sequence=sequence,
                protein_name=protein_name,
                output_dir=output_dir,
                force_refresh=bool(force_refresh),
            )
            if result and not result.get('error') and result.get('pdb_path'):
                t.pdb_path = result['pdb_path']
                t.structure_source = result.get('source')
                t.plddt_mean = result.get('plddt_mean')
                t.structure_quality_tier = result.get('quality_tier')
                hz.session.save()

        if result.get('error'):
            return {
                'error': result['error'],
                'hint': result.get('hint', 'Please provide uniprot_id or sequence.'),
                'tried': result.get('tried', []),
            }

        return {
            'pdb_path': result.get('pdb_path'),
            'source': result.get('source'),
            'source_label': result.get('source_label'),
            'plddt_mean': result.get('plddt_mean'),
            'quality_tier': result.get('quality_tier'),
            'quality_summary': result.get('quality_summary'),
            'from_cache': result.get('from_cache', False),
            'afdb_version': result.get('afdb_version'),
        }
    except Exception as e:
        return {'error': str(e), 'hint': 'Provide uniprot_id (AFDB) or sequence (ESMFold).'}
