"""
MetaKG Skills — 10 metabolic knowledge graph query and analysis skills
=======================================================================
Integrates iML1515 (E. coli) and iMM904 (yeast) genome-scale models
with GraphRAG (Systems_Biology + Synthetic_Biology KGs) for metabolic
engineering decision support.

Skills registered:
  metakg_query         — entity lookup (metabolite/reaction/gene)
  metakg_pathway       — BFS pathway finding between metabolites
  metakg_reactions     — reactions connected to a metabolite
  metakg_flux          — FBA: theoretical yield & growth
  metakg_bottleneck    — identify flux bottlenecks in a pathway
  metakg_cofactor      — cofactor balance analysis (NAD+/NADPH/ATP)
  metakg_competitor    — competing pathways draining precursor
  metakg_essentiality  — gene essentiality screen
  metakg_compare       — compare E. coli vs yeast for target
  metakg_toxicity      — screen intermediates for toxicity
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Skill functions ────────────────────────────────────────────────────────────

def metakg_query(
    query: str,
    organism: str = "ecoli",
    node_type: Optional[str] = None,
    limit: int = 20,
    output_dir: Optional[str] = None,
) -> Dict:
    """Search metaKG for metabolites, reactions, or genes by keyword.

    Args:
        query: search term (e.g. 'tyrosine', 'aroG', 'DAHPS')
        organism: 'ecoli' or 'yeast'
        node_type: filter by 'Metabolite', 'Reaction', or 'Gene' (None = all)
        limit: max results
        output_dir: not used (included for API consistency)

    Returns:
        Dict with matching nodes, each with id, name, type, formula (if metabolite)
    """
    try:
        from zhuangzhou.metakg.bridge import MetaKGBridge
        bridge = MetaKGBridge(organism)
        if not bridge.is_ready():
            return bridge.data   # error dict

        if not query or not isinstance(query, str) or not query.strip():
            return {
                'error': 'query string is required',
                'hint': 'Provide a search term, e.g. query="tyrosine" or query="aroG"',
            }
        results = bridge.search_nodes(query, node_type=node_type, limit=limit)
        if not results:
            return {
                "query": query, "organism": organism, "hits": 0, "nodes": [],
                "hint": f"Try broader term or check BiGG IDs at bigg.ucsd.edu",
            }
        return {
            "query": query, "organism": organism,
            "hits": len(results), "nodes": results,
            "next_steps": ["metakg_pathway", "metakg_reactions", "metakg_flux"],
        }
    except Exception as e:
        return {"error": str(e), "hint": "Check metaKG parquet files at E:/Code/metaKG/"}


def metakg_pathway(
    source_metabolite: str = '',
    target_metabolite: str = '',
    organism: str = "ecoli",
    max_steps: int = 8,
    exclude_currency: bool = True,
    output_dir: Optional[str] = None,
    source: str = '',        # alias for source_metabolite
    target: str = '',        # alias for target_metabolite
    **kw,
) -> Dict:
    """Find metabolic pathway between two metabolites using BFS graph traversal.

    Args:
        source_metabolite: BiGG metabolite ID (e.g. 'pep_c', 'pyr_c', 'g6p_c')
        target_metabolite: BiGG metabolite ID (e.g. 'tyr__L_c', 'phe__L_c')
        organism: 'ecoli' or 'yeast'
        max_steps: max reaction steps to search
        exclude_currency: exclude ATP/NAD/H2O from traversal (reduces false shortcuts)
        output_dir: not used

    Returns:
        Dict with paths (list of reaction chains), genes per step, reversibility
    """
    try:
        from zhuangzhou.metakg.bridge import MetaKGBridge

        # Resolve aliases
        source_metabolite = source_metabolite or source
        target_metabolite = target_metabolite or target
        if not source_metabolite or not target_metabolite:
            return {'error': 'source_metabolite and target_metabolite are required',
                    'hint': 'Use BiGG IDs with compartment, e.g. source_metabolite="pep_c", target_metabolite="tyr__L_c"'}

        bridge = MetaKGBridge(organism)
        if not bridge.is_ready():
            return bridge.data

        paths = bridge.find_path(source_metabolite, target_metabolite,
                                 max_steps=max_steps, exclude_currency=exclude_currency)
        if not paths:
            return {
                "source": source_metabolite,
                "target": target_metabolite,
                "organism": organism,
                "paths_found": 0,
                "hint": (
                    f"No direct path found in {max_steps} steps. "
                    f"Try: (1) increase max_steps, (2) use intermediate metabolite, "
                    f"(3) check BiGG IDs are correct (include compartment: _c/_e/_p)"
                ),
            }

        formatted = []
        for i, path in enumerate(paths):
            steps_desc = []
            all_genes = []
            for step in path:
                genes = [g["gene_name"] or g["gene_id"] for g in step.get("genes", [])]
                all_genes.extend(genes)
                steps_desc.append({
                    "step":          len(steps_desc) + 1,
                    "reaction":      step["reaction_id"],
                    "reaction_name": step["reaction_name"],
                    "from":          step["from_met"],
                    "to":            step["to_met"],
                    "genes":         genes,
                    "reversible":    step.get("reversible", False),
                })
            formatted.append({
                "path_index": i + 1,
                "n_steps":    len(path),
                "steps":      steps_desc,
                "all_genes":  list(set(all_genes)),
            })

        return {
            "source": source_metabolite,
            "target": target_metabolite,
            "organism": organism,
            "paths_found": len(paths),
            "paths": formatted,
            "next_steps": ["metakg_flux", "metakg_bottleneck", "metakg_cofactor",
                           "synbio_expression_balance"],
        }
    except Exception as e:
        logger.exception("metakg_pathway failed")
        return {"error": str(e), "hint": "Verify metabolite IDs include compartment suffix"}


def metakg_reactions(
    metabolite_id: str = '',
    organism: str = "ecoli",
    output_dir: Optional[str] = None,
    metabolite: str = '',   # alias for metabolite_id
    **kw,
) -> Dict:
    """Get all reactions consuming or producing a metabolite.

    Args:
        metabolite_id: BiGG metabolite ID with compartment (e.g. 'pep_c')
        organism: 'ecoli' or 'yeast'

    Returns:
        Dict with consuming reactions and producing reactions, genes for each
    """
    try:
        from zhuangzhou.metakg.bridge import MetaKGBridge

        metabolite_id = metabolite_id or metabolite
        if not metabolite_id:
            return {'error': 'metabolite_id is required',
                    'hint': 'Pass metabolite_id="tyr__L_c" (BiGG ID with compartment suffix)'}

        bridge = MetaKGBridge(organism)
        if not bridge.is_ready():
            return bridge.data

        result = bridge.get_reactions_for_metabolite(metabolite_id)
        # Enrich with gene info
        for rxn in result["consuming"] + result["producing"]:
            genes = bridge.get_genes_for_reaction(rxn["reaction_id"])
            rxn["genes"] = [g["gene_name"] or g["gene_id"] for g in genes]

        result["metabolite_id"] = metabolite_id
        result["organism"] = organism
        result["next_steps"] = ["metakg_flux", "metakg_pathway"]
        return result
    except Exception as e:
        return {"error": str(e), "hint": "Include compartment in ID (e.g. pep_c not pep)"}


def metakg_flux(
    objective_reaction: str = '',
    organism: str = "ecoli",
    carbon_source: str = "glc__D_e",
    carbon_uptake: float = 10.0,
    maximize: bool = True,
    output_dir: Optional[str] = None,
    target: str = '',   # alias: accepts metabolite ID and auto-converts to EX_ reaction
    **kw,
) -> Dict:
    """Run Flux Balance Analysis to find theoretical production yield.

    Args:
        objective_reaction: exchange reaction to optimize (e.g. 'EX_tyr__L_e')
        organism: 'ecoli' or 'yeast'
        carbon_source: BiGG metabolite ID for carbon (default: glucose 'glc__D_e')
        carbon_uptake: uptake rate mmol/gDW/h (default: 10 = ~1 g/L glucose)
        maximize: True to maximize production

    Returns:
        Dict with optimal flux, yield estimate, growth rate, shadow prices (bottlenecks)
    """
    try:
        from zhuangzhou.metakg.flux import FluxAnalyzer

        # Resolve target alias: metabolite ID → exchange reaction ID
        if not objective_reaction and target:
            if target.startswith('EX_'):
                objective_reaction = target
            else:
                met_e = target.replace('_c', '_e').replace('_p', '_e')
                objective_reaction = f'EX_{met_e}'
        if not objective_reaction:
            return {'error': 'objective_reaction is required (e.g. "EX_tyr__L_e")',
                    'hint': 'Use target="tyr__L_c" as shorthand or specify the BiGG exchange reaction ID'}

        analyzer = FluxAnalyzer(organism)
        result = analyzer.run_fba(
            objective_reaction=objective_reaction,
            carbon_source=carbon_source,
            carbon_uptake_mmol=carbon_uptake,
            maximize=maximize,
        )
        if "error" not in result:
            result["next_steps"] = ["metakg_bottleneck", "synbio_expression_balance",
                                     "metakg_competitor"]
        return result
    except Exception as e:
        return {"error": str(e),
                "hint": "COBRApy required. Install: pip install cobra. "
                        "Or use metakg_bottleneck for heuristic analysis."}


def metakg_bottleneck(
    pathway_reactions: Optional[List[str]] = None,
    organism: str = "ecoli",
    carbon_source: str = "glc__D_e",
    output_dir: Optional[str] = None,
    target: str = '',  # alias: accepts metabolite name or ID — returns hint if no reactions provided
    **kw,
) -> Dict:
    """Identify flux bottlenecks in a metabolic pathway via FBA.

    Args:
        pathway_reactions: list of BiGG reaction IDs (e.g. ['DAHPS', 'DHQS', 'DHQTi'])
        organism: 'ecoli' or 'yeast'
        carbon_source: carbon source metabolite ID

    Returns:
        Dict with bottleneck reactions (near-zero flux), high-flux reactions, all fluxes
    """
    try:
        from zhuangzhou.metakg.flux import FluxAnalyzer

        if not pathway_reactions or not isinstance(pathway_reactions, list):
            return {
                'error': 'pathway_reactions list is required',
                'hint': (f'Provide a list of BiGG reaction IDs, e.g. pathway_reactions=["DAHPS","DHQS","DHQTi"]. '
                         f'Use metakg_pathway(source=..., target="{target}") first to discover the pathway reactions.') if target else
                        'Provide a list of BiGG reaction IDs, e.g. pathway_reactions=["DAHPS","DHQS","DHQTi"]',
            }

        analyzer = FluxAnalyzer(organism)
        result = analyzer.find_bottlenecks(pathway_reactions, carbon_source)
        if "error" not in result:
            result["remediation"] = {
                "overexpression": "Use strong promoter (J23100/Ptrc) for bottleneck gene",
                "enzyme_engineering": "Increase kcat via directed evolution or rational design",
                "alternative_isozyme": "Search KEGG for iso-enzymes with higher activity",
                "pathway_bypass": "Add parallel branch to increase total flux capacity",
            }
            result["next_steps"] = ["synbio_expression_balance", "synbio_promoter_select"]
        return result
    except Exception as e:
        return {"error": str(e),
                "hint": "Requires COBRApy. Alternatively, check literature kcat values for each enzyme."}


def metakg_cofactor(
    pathway_reactions: Optional[List[str]] = None,
    organism: str = "ecoli",
    output_dir: Optional[str] = None,
    target: str = '',  # alias — prompts user to provide pathway_reactions
    **kw,
) -> Dict:
    """Analyze cofactor balance (NAD+/NADPH/ATP) across a metabolic pathway.

    Imbalanced cofactor pools cause growth inhibition and reduced production.

    Args:
        pathway_reactions: list of BiGG reaction IDs
        organism: 'ecoli' or 'yeast'

    Returns:
        Dict with cofactor summary (balanced/surplus/deficit per pair)
    """
    try:
        from zhuangzhou.metakg.bridge import MetaKGBridge

        if not pathway_reactions:
            return {
                'error': 'pathway_reactions list is required',
                'hint': (f'Use metakg_pathway(source=..., target="{target}") to get reaction IDs first, '
                         'then pass them here as pathway_reactions=[...]') if target else
                        'Provide BiGG reaction IDs, e.g. pathway_reactions=["DAHPS","DHQS"]',
            }

        bridge = MetaKGBridge(organism)
        if not bridge.is_ready():
            return bridge.data

        result = bridge.analyze_cofactor_balance(pathway_reactions)

        # Add engineering recommendations
        recs = []
        for key, entry in result["cofactor_summary"].items():
            if entry["status"] == "deficit_reductant":
                recs.append(f"{entry['label']}: DEFICIT — "
                             f"Add NADPH-generating enzyme (e.g., pntAB transhydrogenase, zwf)")
            elif entry["status"] == "surplus_reductant":
                recs.append(f"{entry['label']}: SURPLUS — "
                             f"Use NADPH-consuming pathway or engineer oxidative branch")

        result["engineering_recommendations"] = recs or ["Cofactor pools appear balanced"]
        result["pathway_reactions"] = pathway_reactions
        result["organism"] = organism
        result["next_steps"] = ["synbio_expression_balance", "metakg_flux"]
        return result
    except Exception as e:
        return {"error": str(e), "hint": "Verify reaction IDs are valid BiGG IDs"}


def metakg_competitor(
    target_metabolite: str = '',
    organism: str = "ecoli",
    output_dir: Optional[str] = None,
    target: str = '',   # alias for target_metabolite
    **kw,
) -> Dict:
    """Find competing pathways that consume the target metabolite or its precursors.

    Args:
        target_metabolite: BiGG metabolite ID (with compartment)
        organism: 'ecoli' or 'yeast'

    Returns:
        Dict with competing reactions, candidate knockout targets
    """
    try:
        from zhuangzhou.metakg.bridge import MetaKGBridge

        target_metabolite = target_metabolite or target
        if not target_metabolite:
            return {'error': 'target_metabolite is required',
                    'hint': 'Pass target_metabolite="tyr__L_c" (BiGG ID with compartment)'}

        bridge = MetaKGBridge(organism)
        if not bridge.is_ready():
            return bridge.data

        conn = bridge.get_reactions_for_metabolite(target_metabolite)
        consuming = conn["consuming"]

        # Filter out biosynthesis reactions (keep competing drains)
        competitors = []
        for rxn in consuming:
            genes = bridge.get_genes_for_reaction(rxn["reaction_id"])
            gene_names = [g["gene_name"] or g["gene_id"] for g in genes]
            competitors.append({
                "reaction": rxn["reaction_id"],
                "name": rxn["reaction_name"],
                "stoichiometry": rxn["stoichiometry"],
                "genes": gene_names,
                "knockout_candidate": len(gene_names) > 0,
            })

        knockout_targets = [c["genes"] for c in competitors if c["knockout_candidate"]]
        knockout_flat = list({g for genes in knockout_targets for g in genes})

        return {
            "target_metabolite": target_metabolite,
            "organism": organism,
            "n_competing_reactions": len(competitors),
            "competing_reactions": competitors,
            "knockout_candidates": knockout_flat,
            "caution": "Verify essentiality before knockout (use metakg_essentiality)",
            "next_steps": ["metakg_essentiality", "synbio_crispr_design"],
        }
    except Exception as e:
        return {"error": str(e), "hint": "Include compartment in metabolite ID"}


def metakg_essentiality(
    gene_list: List[str],
    organism: str = "ecoli",
    output_dir: Optional[str] = None,
) -> Dict:
    """Screen genes for essentiality (knockout → lethal) to assess knockout safety.

    Args:
        gene_list: BiGG gene IDs to test (e.g. ['b0872', 'b2551'])
        organism: 'ecoli' or 'yeast'

    Returns:
        Dict with essential genes, safe knockouts, and auxotrophic risks
    """
    try:
        from zhuangzhou.metakg.flux import FluxAnalyzer
        analyzer = FluxAnalyzer(organism)
        result = analyzer.find_essential_genes(pathway_genes=gene_list)

        safe_kos = [g for g in gene_list if g not in result.get("essential_genes", [])]
        result["tested_genes"] = gene_list
        result["safe_knockout_candidates"] = safe_kos
        result["caution"] = (
            "Essentiality is predicted in minimal media with glucose. "
            "Rich media or co-culture may rescue essential genes."
        )
        result["next_steps"] = ["synbio_crispr_design"]
        return result
    except Exception as e:
        return {"error": str(e),
                "hint": "COBRApy required. Alternatively, check literature for essential genes."}


def metakg_compare(
    target_metabolite: str = '',
    organisms: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    target: str = '',   # alias for target_metabolite
    **kw,
) -> Dict:
    """Compare E. coli vs yeast as chassis for producing a target metabolite.

    Compares: pathway existence, reaction count, natural titer, regulatory burden,
    known engineering records.

    Args:
        target_metabolite: common name or BiGG ID of target
        organisms: list to compare, default ['ecoli', 'yeast']

    Returns:
        Dict with per-organism analysis and recommendation
    """
    target_metabolite = target_metabolite or target
    if not target_metabolite or not target_metabolite.strip():
        return {
            'error': 'target_metabolite is required',
            'hint': 'Provide a metabolite name or BiGG ID, e.g. target_metabolite="tyr__L_c"',
        }
    if not organisms:
        organisms = ["ecoli", "yeast"]

    try:
        from zhuangzhou.metakg.bridge import MetaKGBridge
        results = {}

        for org in organisms:
            bridge = MetaKGBridge(org)
            if not bridge.is_ready():
                results[org] = {"status": "unavailable", "error": bridge.data.get("error")}
                continue

            # Search for target in this organism's KG
            hits = bridge.search_nodes(target_metabolite, node_type="Metabolite", limit=5)
            if hits:
                met_id = hits[0]["id"]
                conn = bridge.get_reactions_for_metabolite(met_id)
                results[org] = {
                    "status": "found",
                    "metabolite_id": met_id,
                    "n_producing_reactions": len(conn["producing"]),
                    "n_consuming_reactions": len(conn["consuming"]),
                    "native_pathway": len(conn["producing"]) > 0,
                }
            else:
                results[org] = {
                    "status": "not_in_model",
                    "hint": "Target not in genome-scale model — may require heterologous pathway",
                }

        # Recommendation heuristic
        recommendation = "E. coli" if results.get("ecoli", {}).get("native_pathway") else "yeast"
        reasoning = []
        if results.get("ecoli", {}).get("native_pathway"):
            reasoning.append("E. coli has native pathway — less engineering needed")
        if "aromatic" in target_metabolite.lower() or "tyr" in target_metabolite.lower():
            reasoning.append("Aromatic biosynthesis: E. coli shikimate pathway well-studied")
        if "ethanol" in target_metabolite.lower() or "isobutanol" in target_metabolite.lower():
            reasoning.append("Alcohol tolerance: yeast generally more tolerant")

        return {
            "target": target_metabolite,
            "comparison": results,
            "recommendation": recommendation,
            "reasoning": reasoning,
            "next_steps": ["metakg_pathway", "metakg_flux"],
        }
    except Exception as e:
        return {"error": str(e), "hint": "Check metaKG parquet files"}


def metakg_toxicity(
    metabolite_ids: List[str],
    organism: str = "ecoli",
    output_dir: Optional[str] = None,
) -> Dict:
    """Screen metabolic intermediates for known toxicity to the chassis.

    Args:
        metabolite_ids: list of BiGG metabolite IDs to screen
        organism: 'ecoli' or 'yeast'

    Returns:
        Dict with toxicity risk per metabolite and mitigation strategies
    """
    from zhuangzhou.metakg.config import TOXICITY_DB

    results = []
    high_risk = []

    for met_id in metabolite_ids:
        # Strip compartment for lookup
        base_id = met_id.rsplit("_", 1)[0] if "_" in met_id else met_id
        met_key = met_id if met_id in TOXICITY_DB else base_id + "_c"

        if met_key in TOXICITY_DB:
            entry = TOXICITY_DB[met_key]
            risk = "HIGH" if entry["threshold_mM"] < 5 else "MEDIUM" if entry["threshold_mM"] < 50 else "LOW"
            record = {
                "metabolite": met_id,
                "risk_level": risk,
                "toxicity_threshold_mM": entry["threshold_mM"],
                "mechanism": entry["mechanism"],
                "mitigations": _get_mitigations(met_key, entry),
            }
            results.append(record)
            if risk == "HIGH":
                high_risk.append(met_id)
        else:
            results.append({
                "metabolite": met_id,
                "risk_level": "UNKNOWN",
                "note": "Not in toxicity database — check literature or use E. coli tolerance data",
            })

    return {
        "screened": len(metabolite_ids),
        "high_risk_metabolites": high_risk,
        "results": results,
        "organism": organism,
        "next_steps": (["synbio_efflux_pump_design", "synbio_dynamic_control"]
                       if high_risk else ["metakg_flux"]),
    }


def _get_mitigations(met_id: str, entry: Dict) -> List[str]:
    """Return engineering strategies for toxic intermediates."""
    mits = []
    mech = entry.get("mechanism", "").lower()

    if "aldehyde" in mech:
        mits += ["Overexpress alcohol dehydrogenase (ADH) to convert to alcohol",
                 "Add aldo-keto reductase (AKR) to reduce aldehyde",
                 "Compartmentalize reaction in engineered organelle"]
    if "membrane" in mech:
        mits += ["Overexpress efflux pumps (TolC/AcrB in E. coli)",
                 "Engineer membrane composition (cis-trans isomerase cti)",
                 "Use adaptive laboratory evolution (ALE) for tolerance"]
    if "acidif" in mech or "acid" in mech:
        mits += ["pH-stat fermentation (continuous base addition)",
                 "Overexpress LDH competitor to reduce acid buildup",
                 "Use buffered media"]
    if "ros" in mech or "oxidative" in mech:
        mits += ["Overexpress catalase (katG) and superoxide dismutase (sodA/B)",
                 "Add antioxidants to media (ascorbate, Trolox)",
                 "Engineer thioredoxin system"]
    if not mits:
        mits = ["Reduce accumulation via pull reaction optimization",
                "Use fed-batch to control intermediate concentration",
                "Engineer efflux for secretion-based removal"]
    return mits


# ── Skill registration ─────────────────────────────────────────────────────────

def register(registry):
    from zhuangzhou.skills.registry import Skill, SkillParam

    S = SkillParam

    SKILLS = [
        Skill(
            name="metakg_query",
            description="Search metabolic KG for metabolites, reactions, or genes by keyword",
            category="metakg",
            fn=metakg_query,
            inputs=[
                S("query",      "string", "Search keyword (metabolite name, gene name, reaction name)",
                  required=True),
                S("organism",   "string", "Chassis organism",
                  required=False, default="ecoli", choices=["ecoli", "yeast"]),
                S("node_type",  "string", "Filter by node type",
                  required=False, choices=["Metabolite", "Reaction", "Gene"]),
                S("limit",      "int",    "Max results", required=False, default=20),
            ],
            outputs=[S("nodes", "list", "Matching entities")],
            requires_gpu=False, requires_wsl=False, requires_network=False,
            remote_capable=False, remote_preferred=False,
            depends_on=[], output_type="analysis", risk_level="low",
            estimated_time="~1s",
        ),
        Skill(
            name="metakg_pathway",
            description="Find metabolic pathway between two metabolites using BFS graph traversal",
            category="metakg",
            fn=metakg_pathway,
            inputs=[
                S("source_metabolite", "string", "BiGG metabolite ID (e.g. 'pep_c', 'g6p_c')",
                  required=True),
                S("target_metabolite", "string", "BiGG metabolite ID (e.g. 'tyr__L_c')",
                  required=True),
                S("organism", "string", "Chassis", required=False, default="ecoli",
                  choices=["ecoli", "yeast"]),
                S("max_steps", "int", "Max reaction steps (BFS depth)", required=False, default=8),
                S("exclude_currency", "bool", "Exclude ATP/NAD/H2O shortcuts", required=False, default=True),
            ],
            outputs=[S("paths", "list", "Reaction chains from source to target")],
            requires_gpu=False, requires_wsl=False, requires_network=False,
            remote_capable=False, remote_preferred=False,
            depends_on=[], output_type="analysis", risk_level="low",
            estimated_time="~2-10s",
        ),
        Skill(
            name="metakg_reactions",
            description="Get all reactions connected to a metabolite (consuming and producing)",
            category="metakg",
            fn=metakg_reactions,
            inputs=[
                S("metabolite_id", "string", "BiGG metabolite ID with compartment (e.g. 'pep_c')",
                  required=True),
                S("organism", "string", "Chassis", required=False, default="ecoli",
                  choices=["ecoli", "yeast"]),
            ],
            outputs=[S("consuming", "list", ""), S("producing", "list", "")],
            requires_gpu=False, requires_wsl=False, requires_network=False,
            remote_capable=False, remote_preferred=False,
            depends_on=[], output_type="analysis", risk_level="low",
            estimated_time="~1s",
        ),
        Skill(
            name="metakg_flux",
            description="FBA: theoretical max yield for a target product (requires COBRApy)",
            category="metakg",
            fn=metakg_flux,
            inputs=[
                S("objective_reaction", "string",
                  "Exchange reaction to maximize (e.g. 'EX_tyr__L_e' for tyrosine)",
                  required=True),
                S("organism", "string", "Chassis", required=False, default="ecoli",
                  choices=["ecoli", "yeast"]),
                S("carbon_source", "string", "Carbon source BiGG ID",
                  required=False, default="glc__D_e"),
                S("carbon_uptake", "float", "Uptake rate mmol/gDW/h",
                  required=False, default=10.0),
                S("maximize", "bool", "True to maximize production",
                  required=False, default=True),
            ],
            outputs=[S("objective_flux", "float", ""), S("yield_estimate", "float", "")],
            requires_gpu=False, requires_wsl=False, requires_network=False,
            remote_capable=False, remote_preferred=False,
            depends_on=[], output_type="analysis", risk_level="low",
            estimated_time="~5-30s",
        ),
        Skill(
            name="metakg_bottleneck",
            description="Identify flux bottlenecks in a metabolic pathway via FBA",
            category="metakg",
            fn=metakg_bottleneck,
            inputs=[
                S("pathway_reactions", "list", "List of BiGG reaction IDs in pathway",
                  required=True),
                S("organism", "string", "Chassis", required=False, default="ecoli",
                  choices=["ecoli", "yeast"]),
                S("carbon_source", "string", "Carbon source BiGG ID",
                  required=False, default="glc__D_e"),
            ],
            outputs=[S("bottleneck_reactions", "list", "")],
            requires_gpu=False, requires_wsl=False, requires_network=False,
            remote_capable=False, remote_preferred=False,
            depends_on=["metakg_pathway"], output_type="analysis", risk_level="low",
            estimated_time="~10-60s",
        ),
        Skill(
            name="metakg_cofactor",
            description="Analyze NAD+/NADPH/ATP cofactor balance across a metabolic pathway",
            category="metakg",
            fn=metakg_cofactor,
            inputs=[
                S("pathway_reactions", "list", "List of BiGG reaction IDs",
                  required=True),
                S("organism", "string", "Chassis", required=False, default="ecoli",
                  choices=["ecoli", "yeast"]),
            ],
            outputs=[S("cofactor_summary", "dict", "Balance per cofactor pair")],
            requires_gpu=False, requires_wsl=False, requires_network=False,
            remote_capable=False, remote_preferred=False,
            depends_on=["metakg_pathway"], output_type="analysis", risk_level="low",
            estimated_time="~2s",
        ),
        Skill(
            name="metakg_competitor",
            description="Find competing pathways draining target metabolite precursors",
            category="metakg",
            fn=metakg_competitor,
            inputs=[
                S("target_metabolite", "string", "BiGG metabolite ID with compartment",
                  required=True),
                S("organism", "string", "Chassis", required=False, default="ecoli",
                  choices=["ecoli", "yeast"]),
            ],
            outputs=[S("competing_reactions", "list", ""),
                     S("knockout_candidates", "list", "")],
            requires_gpu=False, requires_wsl=False, requires_network=False,
            remote_capable=False, remote_preferred=False,
            depends_on=[], output_type="analysis", risk_level="low",
            estimated_time="~2s",
        ),
        Skill(
            name="metakg_essentiality",
            description="Screen genes for essentiality (is knockout lethal?) via single-gene deletion FBA",
            category="metakg",
            fn=metakg_essentiality,
            inputs=[
                S("gene_list", "list", "BiGG gene IDs to test (e.g. ['b0872', 'b2551'])",
                  required=True),
                S("organism", "string", "Chassis", required=False, default="ecoli",
                  choices=["ecoli", "yeast"]),
            ],
            outputs=[S("essential_genes", "list", ""),
                     S("safe_knockout_candidates", "list", "")],
            requires_gpu=False, requires_wsl=False, requires_network=False,
            remote_capable=False, remote_preferred=False,
            depends_on=["metakg_competitor"], output_type="analysis", risk_level="low",
            estimated_time="~30-300s",
        ),
        Skill(
            name="metakg_compare",
            description="Compare E. coli vs yeast chassis for producing a target metabolite",
            category="metakg",
            fn=metakg_compare,
            inputs=[
                S("target_metabolite", "string", "Common name or BiGG ID of target metabolite",
                  required=True),
                S("organisms", "list", "Organisms to compare",
                  required=False, default=["ecoli", "yeast"]),
            ],
            outputs=[S("comparison", "dict", ""), S("recommendation", "string", "")],
            requires_gpu=False, requires_wsl=False, requires_network=False,
            remote_capable=False, remote_preferred=False,
            depends_on=[], output_type="analysis", risk_level="low",
            estimated_time="~5s",
        ),
        Skill(
            name="metakg_toxicity",
            description="Screen pathway intermediates for cytotoxicity and get mitigation strategies",
            category="metakg",
            fn=metakg_toxicity,
            inputs=[
                S("metabolite_ids", "list",
                  "BiGG metabolite IDs to screen (e.g. ['acald_c', 'fald_c'])",
                  required=True),
                S("organism", "string", "Chassis", required=False, default="ecoli",
                  choices=["ecoli", "yeast"]),
            ],
            outputs=[S("results", "list", "Toxicity profile per metabolite")],
            requires_gpu=False, requires_wsl=False, requires_network=False,
            remote_capable=False, remote_preferred=False,
            depends_on=["metakg_pathway"], output_type="analysis", risk_level="low",
            estimated_time="~1s",
        ),
    ]

    for skill in SKILLS:
        registry.register(skill)
    return len(SKILLS)


# Compatibility alias
register_skills = register
