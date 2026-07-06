"""
FluxAnalyzer — COBRApy FBA wrapper + heuristic fallbacks
=========================================================
Provides:
  - Flux Balance Analysis (theoretical max yield)
  - Flux Variability Analysis (reaction flux ranges)
  - Gene essentiality (single/double knockout)
  - Production envelope (growth vs. production trade-off)
  - Bottleneck identification (reactions with near-zero flux)
  - Heuristic mode (when COBRApy unavailable)
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class FluxAnalyzer:
    """COBRApy-backed metabolic flux analysis."""

    def __init__(self, organism: str = "ecoli"):
        self.organism = organism
        self._model = None   # lazy

    def _get_model(self):
        if self._model is not None:
            return self._model
        from .bridge import MetaKGBridge
        bridge = MetaKGBridge(self.organism)
        result = bridge.load_cobra_model()
        if isinstance(result, dict) and "error" in result:
            return result
        self._model = result
        return self._model

    # ── FBA ───────────────────────────────────────────────────────────────────

    def run_fba(
        self,
        objective_reaction: str,
        carbon_source: str = "glc__D_e",
        carbon_uptake_mmol: float = 10.0,
        maximize: bool = True,
    ) -> Dict:
        """Maximize/minimize a reaction flux via FBA.

        Args:
            objective_reaction: Reaction ID to optimize (e.g., 'EX_succ_e' for succinate export)
            carbon_source: Exchange reaction metabolite (e.g., 'glc__D_e')
            carbon_uptake_mmol: Carbon source uptake rate (mmol/gDW/h)
            maximize: True to maximize production, False to minimize

        Returns:
            Dict with objective_value, shadow_prices, growth_rate, yield_estimate
        """
        model = self._get_model()
        if isinstance(model, dict):
            return model  # error

        import copy
        with model:  # COBRApy context manager restores original state
            # Set carbon source uptake
            carbon_ex = f"EX_{carbon_source}"
            if carbon_ex in model.reactions:
                model.reactions.get_by_id(carbon_ex).lower_bound = -carbon_uptake_mmol
            else:
                # Try without compartment suffix
                carbon_ex2 = f"EX_{carbon_source.rsplit('_',1)[0]}_e"
                if carbon_ex2 in model.reactions:
                    model.reactions.get_by_id(carbon_ex2).lower_bound = -carbon_uptake_mmol

            # Set objective
            if objective_reaction not in [r.id for r in model.reactions]:
                # Try to find exchange reaction
                candidates = [r.id for r in model.reactions
                              if objective_reaction.lower() in r.id.lower()]
                if not candidates:
                    return {"error": f"Reaction '{objective_reaction}' not found",
                            "hint": f"Search with metakg_query to find the correct reaction ID"}
                objective_reaction = candidates[0]

            model.objective = objective_reaction
            model.objective.direction = "max" if maximize else "min"

            solution = model.optimize()

            if solution.status != "optimal":
                return {"error": f"FBA infeasible: {solution.status}",
                        "hint": "Check carbon source bounds or reaction direction"}

            obj_flux = solution.objective_value
            growth_rxn = model.reactions.get_by_id("BIOMASS_Ec_iML1515_core_75p37M") \
                         if "BIOMASS_Ec_iML1515_core_75p37M" in [r.id for r in model.reactions] \
                         else None

            # Find biomass reaction robustly: check known IDs in order
            _biomass_ids = [
                "BIOMASS_Ec_iML1515_core_75p37M",  # iML1515 E. coli
                "BIOMASS_SC5_notrace",               # iMM904 S. cerevisiae
                "BIOMASS_SC4_bal",                   # alternative yeast
                "BIOMASS_Ec_iJO1366_WT_53p95M",     # iJO1366
            ]
            growth_rate = 0.0
            for _bid in _biomass_ids:
                _val = solution.fluxes.get(_bid, None)
                if _val is not None and _val > 0:
                    growth_rate = float(_val)
                    break

            # Yield: mmol product / mmol carbon source consumed
            carbon_consumed = abs(solution.fluxes.get(carbon_ex, carbon_uptake_mmol))
            yield_estimate = (obj_flux / carbon_consumed) if carbon_consumed > 0 else 0.0

            # Top shadow prices (metabolic bottlenecks)
            try:
                shadow = solution.shadow_prices.abs().sort_values(ascending=False).head(10)
                shadow_dict = shadow.to_dict()
            except Exception:
                shadow_dict = {}

            return {
                "objective_reaction": objective_reaction,
                "objective_flux": round(obj_flux, 4),
                "growth_rate": round(growth_rate, 4),
                "carbon_source": carbon_source,
                "carbon_uptake": carbon_uptake_mmol,
                "yield_estimate": round(yield_estimate, 4),
                "units": "mmol/gDW/h",
                "top_shadow_prices": {k: round(v, 4) for k, v in list(shadow_dict.items())[:10]},
                "status": "optimal",
            }

    # ── FVA ───────────────────────────────────────────────────────────────────

    def run_fva(
        self,
        reaction_ids: Optional[List[str]] = None,
        fraction_of_optimum: float = 0.9,
    ) -> Dict:
        """Flux Variability Analysis: find min/max flux for each reaction."""
        model = self._get_model()
        if isinstance(model, dict):
            return model

        try:
            from cobra.flux_analysis import flux_variability_analysis
        except ImportError:
            return {"error": "COBRApy flux_analysis not available"}

        with model:
            if reaction_ids:
                rxns = [model.reactions.get_by_id(r) for r in reaction_ids
                        if r in [rx.id for rx in model.reactions]]
            else:
                rxns = None

            fva_result = flux_variability_analysis(
                model, reaction_list=rxns,
                fraction_of_optimum=fraction_of_optimum
            )

            # Identify blocked reactions (max_flux ≈ 0) and bottlenecks (high variability)
            blocked = []
            flexible = []
            for rxn_id, row in fva_result.iterrows():
                rng = abs(row["maximum"] - row["minimum"])
                if abs(row["maximum"]) < 1e-6 and abs(row["minimum"]) < 1e-6:
                    blocked.append(rxn_id)
                elif rng > 10.0:
                    flexible.append({"reaction": rxn_id,
                                     "min": round(row["minimum"], 4),
                                     "max": round(row["maximum"], 4),
                                     "range": round(rng, 4)})

            return {
                "n_reactions": len(fva_result),
                "n_blocked": len(blocked),
                "blocked_reactions": blocked[:20],
                "highly_flexible": flexible[:10],
                "fraction_of_optimum": fraction_of_optimum,
            }

    # ── Gene essentiality ──────────────────────────────────────────────────────

    def find_essential_genes(self, pathway_genes: Optional[List[str]] = None) -> Dict:
        """Identify essential genes (knockout → no growth)."""
        model = self._get_model()
        if isinstance(model, dict):
            return model

        try:
            from cobra.flux_analysis import single_gene_deletion
        except ImportError:
            return {"error": "COBRApy not available"}

        with model:
            if pathway_genes:
                gene_list = [model.genes.get_by_id(g) for g in pathway_genes
                             if g in [gn.id for gn in model.genes]]
            else:
                gene_list = None

            deletion_results = single_gene_deletion(model, gene_list=gene_list)
            essential = deletion_results[deletion_results["growth"] < 1e-6]

            return {
                "n_tested": len(deletion_results),
                "n_essential": len(essential),
                "essential_genes": list(essential.index.get_level_values(0))[:30],
            }

    # ── Production envelope ────────────────────────────────────────────────────

    def production_envelope(
        self,
        product_reaction: str,
        n_points: int = 10,
    ) -> Dict:
        """Growth vs. production trade-off (Pareto frontier)."""
        model = self._get_model()
        if isinstance(model, dict):
            return model

        try:
            from cobra.flux_analysis import production_envelope as cobra_pe
        except ImportError:
            return {"error": "COBRApy not available"}

        with model:
            if product_reaction not in [r.id for r in model.reactions]:
                return {"error": f"Reaction '{product_reaction}' not in model"}

            env = cobra_pe(model, [model.reactions.get_by_id(product_reaction)],
                           points=n_points)

            return {
                "product_reaction": product_reaction,
                "envelope": env.to_dict(orient="records"),
                "max_theoretical_yield": round(env.iloc[:, 1].max(), 4),
                "note": "Trade-off: higher growth → lower product yield typically",
            }

    # ── Bottleneck identification ───────────────────────────────────────────────

    def find_bottlenecks(
        self,
        pathway_reactions: List[str],
        carbon_source: str = "glc__D_e",
    ) -> Dict:
        """Find reactions with near-zero flux in the pathway under FBA."""
        model = self._get_model()
        if isinstance(model, dict):
            return model

        with model:
            carbon_ex = f"EX_{carbon_source}"
            if carbon_ex in [r.id for r in model.reactions]:
                model.reactions.get_by_id(carbon_ex).lower_bound = -10.0

            solution = model.optimize()
            if solution.status != "optimal":
                return {"error": f"FBA infeasible: {solution.status}"}

            bottlenecks = []
            high_flux   = []
            for rxn_id in pathway_reactions:
                flux = solution.fluxes.get(rxn_id, None)
                if flux is None:
                    continue
                entry = {"reaction": rxn_id, "flux": round(flux, 4)}
                if abs(flux) < 1e-4:
                    bottlenecks.append(entry)
                elif abs(flux) > 1.0:
                    high_flux.append(entry)

            return {
                "bottleneck_reactions": bottlenecks,
                "high_flux_reactions": high_flux,
                "all_fluxes": {r: round(solution.fluxes.get(r, 0.0), 4)
                               for r in pathway_reactions},
                "interpretation": (
                    f"{len(bottlenecks)} bottleneck(s) with near-zero flux — "
                    "consider: overexpression, enzyme engineering, or alternative isozymes"
                ),
            }
