"""
MetaKGBridge — Query interface for metabolic knowledge graphs
============================================================
Loads iML1515 (E. coli) and iMM904 (yeast) parquet files and provides:
  - Entity lookup (metabolite/reaction/gene)
  - Neighbor traversal (one-hop reactions around a metabolite)
  - Stoichiometry matrix access
  - BFS pathway finding between two metabolites
  - Cofactor usage analysis
  - JSON model loading (for COBRApy FBA if available)

All heavy loading is lazy and cached. Thread-safe for concurrent queries.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

_cache_lock = threading.Lock()
_graph_cache: Dict[str, Any] = {}   # organism -> graph data


class MetaKGBridge:
    """Primary interface for querying metabolic KGs from parquet files."""

    def __init__(self, organism: str = "ecoli"):
        from .config import MODELS
        if organism not in MODELS:
            raise ValueError(f"Unknown organism '{organism}'. Available: {list(MODELS)}")
        self.organism = organism
        self.model_cfg = MODELS[organism]
        self._data: Optional[Dict] = None

    # ── Lazy load ──────────────────────────────────────────────────────────────

    def _load(self) -> Dict:
        """Load and cache graph data from parquet or JSON fallback."""
        with _cache_lock:
            if self.organism in _graph_cache:
                return _graph_cache[self.organism]

            # Try parquet first (fast columnar format)
            nodes_path = self.model_cfg["nodes"]
            edges_path = self.model_cfg["edges"]
            if os.path.exists(nodes_path) and os.path.exists(edges_path):
                result = self._load_from_parquet(nodes_path, edges_path)
                if "error" not in result:
                    _graph_cache[self.organism] = result
                    return result
                logger.debug("Parquet load failed, trying JSON: %s", result.get("error"))

            # Fallback: load directly from BiGG JSON model
            json_path = self.model_cfg["json"]
            if os.path.exists(json_path):
                result = self._load_from_json(json_path)
                if "error" not in result:
                    _graph_cache[self.organism] = result
                    logger.info("MetaKG loaded from JSON: %s (%d nodes, %d edges)",
                                self.organism, result["node_count"], result["edge_count"])
                    return result
                return result

            return {"error": f"Neither parquet nor JSON model found for {self.organism}. "
                             f"Expected: {json_path}"}

    def _load_from_parquet(self, nodes_path: str, edges_path: str) -> Dict:
        """Load from parquet files (fast path, requires pyarrow)."""
        try:
            import pandas as pd
        except ImportError:
            return {"error": "pandas not available"}

        try:
            nodes_df = pd.read_parquet(nodes_path)
            edges_df = pd.read_parquet(edges_path)
        except Exception as e:
            return {"error": f"Parquet load failed: {e}"}

        # Index nodes by id
        node_index = {str(row["id"]): dict(row) for _, row in nodes_df.iterrows()}

        # Build adjacency list: id → list of (neighbor_id, edge_type, stoichiometry)
        adj_out: Dict[str, List] = {}
        adj_in:  Dict[str, List] = {}

        for _, row in edges_df.iterrows():
            src    = str(row["source"])
            tgt    = str(row["target"])
            etype  = str(row.get("type", ""))
            stoich = float(row.get("stoichiometry", 1.0))
            adj_out.setdefault(src, []).append((tgt, etype, stoich))
            adj_in.setdefault(tgt, []).append((src, etype, stoich))

        return {
            "nodes": node_index,
            "adj_out": adj_out,
            "adj_in": adj_in,
            "node_count": len(node_index),
            "edge_count": len(edges_df),
        }

    def _load_from_json(self, json_path: str) -> Dict:
        """Load directly from BiGG JSON model (no pandas dependency)."""
        try:
            with open(json_path, encoding="utf-8") as f:
                model = json.load(f)
        except Exception as e:
            return {"error": f"JSON load failed: {e}"}

        node_index: Dict[str, Dict] = {}
        adj_out:    Dict[str, List] = {}
        adj_in:     Dict[str, List] = {}
        edge_count  = 0

        # Genes
        for g in model.get("genes", []):
            gid = str(g["id"])
            node_index[gid] = {"id": gid, "name": g.get("name", gid), "type": "Gene"}

        # Metabolites
        for m in model.get("metabolites", []):
            mid = f"{m['id']}_{m['compartment']}" if "_" not in m["id"] else m["id"]
            # BiGG IDs may already include compartment
            raw_id = m["id"]
            compartment = m.get("compartment", "c")
            # Unique ID: raw_id if it ends with _c/_e/etc., else append compartment
            uid = raw_id if raw_id.endswith(f"_{compartment}") else f"{raw_id}_{compartment}"
            node_index[uid] = {
                "id": uid, "original_id": raw_id,
                "name": m.get("name", ""), "type": "Metabolite",
                "compartment": compartment,
                "formula": m.get("formula", ""),
                "charge": m.get("charge", None),
            }

        # Reactions + edges
        import re as _re
        for r in model.get("reactions", []):
            rxn_id = r["id"]
            lb, ub = r.get("lower_bound", 0), r.get("upper_bound", 1000)
            node_index[rxn_id] = {
                "id": rxn_id, "name": r.get("name", rxn_id), "type": "Reaction",
                "lower_bound": lb, "upper_bound": ub,
                "reversible": lb < 0 and ub > 0,
                "gene_reaction_rule": r.get("gene_reaction_rule", ""),
            }

            # Stoichiometry edges
            for met_raw_id, coeff in r.get("metabolites", {}).items():
                # Resolve metabolite UID
                met_uid = met_raw_id  # BiGG JSON already has compartment in key
                if met_uid not in node_index:
                    # Try appending compartment
                    for cpt in ("c", "e", "p", "m"):
                        alt = f"{met_raw_id}_{cpt}"
                        if alt in node_index:
                            met_uid = alt
                            break

                if coeff < 0:  # substrate: met → rxn (CONSUMES)
                    adj_out.setdefault(met_uid, []).append((rxn_id, "CONSUMES", abs(coeff)))
                    adj_in.setdefault(rxn_id, []).append((met_uid, "CONSUMES", abs(coeff)))
                else:           # product: rxn → met (PRODUCES)
                    adj_out.setdefault(rxn_id, []).append((met_uid, "PRODUCES", coeff))
                    adj_in.setdefault(met_uid, []).append((rxn_id, "PRODUCES", coeff))
                edge_count += 1

            # Gene edges: gene → reaction (ENCODES_ENZYME_FOR)
            gpr = r.get("gene_reaction_rule", "")
            if gpr:
                for gene_id in _re.findall(r"[a-zA-Z0-9_]+", gpr):
                    if gene_id in node_index and node_index[gene_id]["type"] == "Gene":
                        adj_out.setdefault(gene_id, []).append(
                            (rxn_id, "ENCODES_ENZYME_FOR", 1.0))
                        adj_in.setdefault(rxn_id, []).append(
                            (gene_id, "ENCODES_ENZYME_FOR", 1.0))
                        edge_count += 1

        return {
            "nodes": node_index,
            "adj_out": adj_out,
            "adj_in": adj_in,
            "node_count": len(node_index),
            "edge_count": edge_count,
        }

    @property
    def data(self) -> Dict:
        if self._data is None:
            self._data = self._load()
        return self._data

    def is_ready(self) -> bool:
        return "error" not in self.data

    # ── Entity lookup ──────────────────────────────────────────────────────────

    def get_node(self, node_id: str) -> Optional[Dict]:
        """Get node info by exact ID."""
        return self.data.get("nodes", {}).get(node_id)

    def search_nodes(self, query: str, node_type: Optional[str] = None,
                     limit: int = 20) -> List[Dict]:
        """Case-insensitive substring search across node IDs and names.

        Chinese queries are automatically translated to English before
        matching, since the metabolic KG uses English/BiGG identifiers.
        """
        if not query or not isinstance(query, str) or not query.strip():
            return []
        # Translate Chinese terms to English for metabolic KG search
        try:
            from zhuangzhou.brain.graphrag import translate_query_terms
            translated = translate_query_terms(query)
            # If bilingual format returned, use only the English part
            if '[EN:' in translated:
                en_part = translated.split('[EN:', 1)[1].rstrip(']').strip()
                q = en_part.lower()
            else:
                q = translated.lower()
        except Exception:
            q = query.lower()

        results = []
        for nid, node in self.data.get("nodes", {}).items():
            if node_type and node.get("type", "").lower() != node_type.lower():
                continue
            if q in nid.lower() or q in str(node.get("name", "")).lower():
                results.append({"id": nid, **node})
                if len(results) >= limit:
                    break
        return results

    def get_reactions_for_metabolite(self, met_id: str) -> Dict[str, List]:
        """Get all reactions consuming/producing a metabolite."""
        adj_out = self.data.get("adj_out", {})
        adj_in  = self.data.get("adj_in", {})

        consuming = []   # metabolite → reaction (CONSUMES edge)
        producing = []   # reaction → metabolite (PRODUCES edge)

        # met → reaction: CONSUMES edge (source=met, target=rxn)
        for (tgt, etype, stoich) in adj_out.get(met_id, []):
            if etype == "CONSUMES":
                rxn = self.get_node(tgt)
                consuming.append({"reaction_id": tgt,
                                   "reaction_name": rxn.get("name","") if rxn else "",
                                   "stoichiometry": stoich})

        # reaction → met: PRODUCES edge (source=rxn, target=met)
        for (src, etype, stoich) in adj_in.get(met_id, []):
            if etype == "PRODUCES":
                rxn = self.get_node(src)
                producing.append({"reaction_id": src,
                                   "reaction_name": rxn.get("name","") if rxn else "",
                                   "stoichiometry": stoich})

        return {"consuming": consuming, "producing": producing,
                "total_connections": len(consuming) + len(producing)}

    def get_genes_for_reaction(self, rxn_id: str) -> List[Dict]:
        """Get all genes encoding enzymes for a reaction."""
        adj_in = self.data.get("adj_in", {})
        genes = []
        for (src, etype, _) in adj_in.get(rxn_id, []):
            if etype == "ENCODES_ENZYME_FOR":
                gene = self.get_node(src)
                genes.append({"gene_id": src,
                               "gene_name": gene.get("name","") if gene else ""})
        return genes

    def get_reaction_metabolites(self, rxn_id: str) -> Dict[str, List]:
        """Get all metabolites consumed/produced by a reaction."""
        adj_out = self.data.get("adj_out", {})
        adj_in  = self.data.get("adj_in", {})

        substrates = []   # rxn → met: CONSUMES edges FROM met TO rxn
        products   = []   # rxn → met: PRODUCES edges FROM rxn TO met

        # reaction → metabolite (PRODUCES)
        for (tgt, etype, stoich) in adj_out.get(rxn_id, []):
            if etype == "PRODUCES":
                met = self.get_node(tgt)
                products.append({"met_id": tgt,
                                  "name": met.get("name","") if met else "",
                                  "formula": met.get("formula","") if met else "",
                                  "stoichiometry": stoich})

        # metabolite → reaction (CONSUMES) — find mets pointing to this rxn
        for nid, edges in adj_out.items():
            node = self.get_node(nid)
            if not node or node.get("type") != "Metabolite":
                continue
            for (tgt, etype, stoich) in edges:
                if tgt == rxn_id and etype == "CONSUMES":
                    substrates.append({"met_id": nid,
                                        "name": node.get("name",""),
                                        "formula": node.get("formula",""),
                                        "stoichiometry": stoich})

        return {"substrates": substrates, "products": products}

    # ── Pathway traversal ──────────────────────────────────────────────────────

    def find_path(
        self,
        source_met: str,
        target_met: str,
        max_steps: int = 8,
        exclude_currency: bool = True,
    ) -> List[Dict]:
        """BFS: find reaction path from source metabolite to target metabolite.

        Returns a list of paths (each path = list of reaction dicts).
        Excludes currency metabolites (ATP, NAD, H2O, H+) from traversal
        to avoid spurious short-circuits.
        """
        CURRENCY = {
            # Energy
            "atp_c","adp_c","amp_c","pi_c","ppi_c",
            "gtp_c","gdp_c","gmp_c","utp_c","udp_c","ctp_c","cdp_c",
            # Redox
            "nad_c","nadh_c","nadp_c","nadph_c",
            "fad_c","fadh2_c",
            # Quinones (electron carriers, respiratory chain)
            "q8_c","q8h2_c","mqn8_c","mql8_c","2dmmql8_c","2dmmq8_c",
            # Small molecules / protons / water
            "h_c","h2o_c","h_e","h2o_e","h_p","h2o_p",
            "co2_c","o2_c","o2_e","co2_e",
            # CoA carriers (lump as currency, resolved downstream)
            "coa_c","accoa_c",
            # Thioredoxin/glutaredoxin
            "thf_c","mlthf_c","5mthf_c","10fthf_c",
        }

        adj_out = self.data.get("adj_out", {})
        adj_in  = self.data.get("adj_in", {})
        nodes   = self.data.get("nodes", {})

        def _get_neighbors(met_id: str):
            """Yield (rxn_id, next_met_id, direction) for all reachable reactions.

            direction='fwd': met is substrate, traverse to products
            direction='rev': met is product of reversible rxn, traverse to substrates
            """
            # Forward: reactions consuming this metabolite
            for (rxn_id, etype, _) in adj_out.get(met_id, []):
                if etype != "CONSUMES":
                    continue
                rxn = nodes.get(rxn_id, {})
                if rxn.get("type") != "Reaction":
                    continue
                for (prod_id, p_etype, p_stoich) in adj_out.get(rxn_id, []):
                    if p_etype == "PRODUCES":
                        yield rxn_id, prod_id, "fwd"

            # Reverse: reversible reactions that produce this metabolite
            # (run them backwards: products → substrates)
            for (rxn_id, p_etype, _) in adj_in.get(met_id, []):
                if p_etype != "PRODUCES":
                    continue
                rxn = nodes.get(rxn_id, {})
                if rxn.get("type") != "Reaction":
                    continue
                # When loaded from parquet, 'reversible' may be absent — default to True
                # (parquet nodes lack lb/ub; assume potentially reversible, user verifies thermodynamics)
                if rxn.get("reversible", None) is False:
                    continue
                # Substrates of this reaction are the "products" when run in reverse
                for (sub_id, s_etype, _) in adj_in.get(rxn_id, []):
                    if s_etype == "CONSUMES":
                        yield rxn_id, sub_id, "rev"

        # State: (current_met_id, path_so_far)
        from collections import deque
        queue = deque([(source_met, [])])
        visited_mets: Set[str] = {source_met}
        found_paths: List[List] = []

        while queue and len(found_paths) < 5:
            met_id, path = queue.popleft()
            if len(path) >= max_steps:
                continue

            for rxn_id, next_met_id, direction in _get_neighbors(met_id):
                if exclude_currency and next_met_id in CURRENCY:
                    continue

                rxn = nodes.get(rxn_id, {})
                step = {
                    "reaction_id":   rxn_id,
                    "reaction_name": rxn.get("name", rxn_id),
                    "from_met":      met_id,
                    "to_met":        next_met_id,
                    "direction":     direction,
                    "genes":         self.get_genes_for_reaction(rxn_id),
                    "reversible":    rxn.get("reversible", False),
                }
                new_path = path + [step]

                if next_met_id == target_met:
                    found_paths.append(new_path)
                    continue

                if next_met_id not in visited_mets:
                    visited_mets.add(next_met_id)
                    queue.append((next_met_id, new_path))

        return found_paths

    # ── Cofactor analysis ──────────────────────────────────────────────────────

    def analyze_cofactor_balance(self, reaction_ids: List[str]) -> Dict:
        """Count net cofactor consumption/production across a set of reactions."""
        from .config import COFACTOR_PAIRS

        balance: Dict[str, float] = {}
        detail: List[Dict] = []

        adj_out = self.data.get("adj_out", {})

        for rxn_id in reaction_ids:
            rxn_node = self.get_node(rxn_id)
            rxn_name = rxn_node.get("name", rxn_id) if rxn_node else rxn_id

            # Products (positive)
            for (prod_id, etype, stoich) in adj_out.get(rxn_id, []):
                if etype == "PRODUCES":
                    balance[prod_id] = balance.get(prod_id, 0) + stoich

            # Substrates (negative) — find mets pointing TO this rxn
            adj_in = self.data.get("adj_in", {})
            for (src_id, etype, stoich) in adj_in.get(rxn_id, []):
                if etype == "ENCODES_ENZYME_FOR":
                    continue
                # These are CONSUMES edges: met → rxn means met is consumed
                balance[src_id] = balance.get(src_id, 0) - stoich

        # Summarize cofactor pairs
        summary = {}
        for key, (ox_id, red_id, label) in COFACTOR_PAIRS.items():
            net_ox  = balance.get(ox_id,  0.0)
            net_red = balance.get(red_id, 0.0)
            net_transfer = net_red - net_ox  # positive = reducing power generated
            summary[key] = {
                "label": label,
                "net_oxidized": round(net_ox, 3),
                "net_reduced":  round(net_red, 3),
                "net_transfer": round(net_transfer, 3),
                "status": (
                    "balanced" if abs(net_transfer) < 0.01
                    else "surplus_reductant" if net_transfer > 0
                    else "deficit_reductant"
                ),
            }
        return {"cofactor_summary": summary, "raw_balance": {
            k: round(v, 4) for k, v in balance.items() if abs(v) > 0.001
        }}

    # ── JSON model access (for COBRApy) ────────────────────────────────────────

    def load_cobra_model(self):
        """Load the BiGG JSON model with COBRApy. Returns model or error dict."""
        json_path = self.model_cfg["json"]
        if not os.path.exists(json_path):
            return {"error": f"JSON model not found: {json_path}"}
        try:
            import cobra
            model = cobra.io.load_json_model(json_path)
            logger.info("COBRApy model loaded: %s (%d rxns, %d mets)",
                        self.organism, len(model.reactions), len(model.metabolites))
            return model
        except ImportError:
            return {"error": "COBRApy not installed. Run: pip install cobra",
                    "hint": "FBA requires COBRApy. Install in native env or use flux estimation heuristics."}
        except Exception as e:
            return {"error": f"COBRApy load failed: {e}"}

    # ── Stats ──────────────────────────────────────────────────────────────────

    def stats(self) -> Dict:
        data = self.data
        if "error" in data:
            return data
        return {
            "organism": self.organism,
            "label": self.model_cfg["label"],
            "node_count": data["node_count"],
            "edge_count": data["edge_count"],
            "bigg_id": self.model_cfg["bigg_id"],
        }


def get_bridge(organism: str = "ecoli") -> MetaKGBridge:
    """Factory function."""
    return MetaKGBridge(organism)


def clear_cache():
    """Clear all cached graph data (use after model file updates)."""
    with _cache_lock:
        _graph_cache.clear()
