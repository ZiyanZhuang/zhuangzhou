"""
Synthetic Biology Parts Library
================================
Curated registry of standardized biological parts:
  - Promoters (constitutive + inducible, E. coli + yeast)
  - RBS elements (Shine-Dalgarno sequences + strength estimates)
  - Terminators
  - Integration sites

All strength values are relative (1.0 = strong constitutive reference).
Sequences are 5'→3' for sense strand.

References:
  - Anderson promoter library: partsregistry.org, Kelly et al. 2009
  - Yeast promoters: Mumberg et al. 1995; Lee et al. 2015 (CEN.PK background)
  - RBS strengths: Salis RBS Calculator v2.1 approximations
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from .config import ECOLI_PROMOTERS, YEAST_PROMOTERS


# ── RBS Library ───────────────────────────────────────────────────────────────

ECOLI_RBS = {
    # BioBrick standard RBS parts (BBa_B series)
    "BBa_B0034": {
        "sequence": "AAAGAGGAGAAA",
        "strength": 1.00,
        "tir": 100000,       # Translation Initiation Rate (au)
        "description": "Strong consensus RBS (Elowitz 2000)",
    },
    "BBa_B0031": {
        "sequence": "TCACACAGGAAAC",
        "strength": 0.30,
        "tir": 30000,
        "description": "Medium RBS",
    },
    "BBa_B0032": {
        "sequence": "TCACACAGGAAAG",
        "strength": 0.08,
        "tir": 8000,
        "description": "Weak-medium RBS",
    },
    "BBa_B0033": {
        "sequence": "TCAAAGAGGAAAC",
        "strength": 0.01,
        "tir": 1000,
        "description": "Weak RBS",
    },
    "BBa_B0035": {
        "sequence": "ACACAGGAAAC",
        "strength": 0.006,
        "tir": 600,
        "description": "Very weak RBS",
    },
    # Salis-optimized synthetic RBS for high expression
    "Salis_ultra": {
        "sequence": "AGGAGG",
        "strength": 2.50,
        "tir": 250000,
        "description": "Salis lab ultra-strong synthetic SD (AGGAGG + optimal spacing)",
    },
    "Salis_medium": {
        "sequence": "AAGGAG",
        "strength": 0.50,
        "tir": 50000,
        "description": "Medium-strength Salis design",
    },
}

YEAST_RBS = {
    # Yeast uses Kozak context instead of SD sequences
    "Kozak_strong":  {"sequence": "AAAAATGTCT", "strength": 1.00,
                      "description": "Optimal Kozak for yeast (Cavener 1987)"},
    "Kozak_medium":  {"sequence": "AAAGATGTCT", "strength": 0.60,
                      "description": "Medium Kozak"},
    "Kozak_weak":    {"sequence": "ATCAATGTCT", "strength": 0.20,
                      "description": "Weak Kozak"},
    "5UTR_TEF1":     {"sequence": "ATAGCTTCAAAATGTTTCTACTCC",
                      "strength": 1.20,
                      "description": "TEF1 5' UTR (includes natural Kozak, strongest in yeast)"},
}

# ── Terminators ───────────────────────────────────────────────────────────────

ECOLI_TERMINATORS = {
    "BBa_B0010": {"efficiency": 0.95, "type": "rho-independent",
                  "description": "T1 terminator from rrnB operon"},
    "BBa_B0012": {"efficiency": 0.99, "type": "rho-independent",
                  "description": "T2 terminator, very strong"},
    "BBa_B0015": {"efficiency": 0.99, "type": "rho-independent", "combination": "B0010+B0012",
                  "description": "Double terminator (standard BioBrick)"},
    "fdT":       {"efficiency": 0.88, "type": "rho-independent",
                  "description": "fd phage terminator, compact"},
    "rrnBT1T2":  {"efficiency": 0.99, "type": "rho-independent",
                  "description": "rrnB dual terminator (T7 + T1 for insulation)"},
}

YEAST_TERMINATORS = {
    "CYC1t":   {"efficiency": 0.95, "description": "Most commonly used yeast terminator"},
    "ADH1t":   {"efficiency": 0.90, "description": "ADH1 terminator, strong"},
    "TEF1t":   {"efficiency": 0.88, "description": "TEF1 3' region"},
    "PGK1t":   {"efficiency": 0.92, "description": "PGK1 terminator, efficient"},
}


# ── Part Selector ─────────────────────────────────────────────────────────────

class PartSelector:
    """Select appropriate biological parts based on expression requirements."""

    @staticmethod
    def select_promoter(
        organism: str = "ecoli",
        strength: float = 1.0,         # target relative strength (0-2)
        inducible: bool = False,
        inducer: Optional[str] = None,
        n_candidates: int = 3,
    ) -> List[Dict]:
        """Select promoters matching target expression level.

        Args:
            organism: 'ecoli' or 'yeast'
            strength: target relative strength (0=weak, 1=medium, 2=strong)
            inducible: True to return inducible promoters only
            inducer: filter by specific inducer (e.g. 'IPTG', 'galactose')
            n_candidates: number of top candidates to return

        Returns:
            List of promoter dicts sorted by match score
        """
        lib = ECOLI_PROMOTERS if organism == "ecoli" else YEAST_PROMOTERS

        candidates = []
        for name, props in lib.items():
            if inducible and props["type"] == "constitutive":
                continue
            if not inducible and props["type"] != "constitutive":
                continue
            if inducer and props.get("inducer", "").lower() != inducer.lower():
                continue

            score = 1.0 - abs(props["strength"] - strength)
            candidates.append({"name": name, "match_score": round(score, 3), **props})

        candidates.sort(key=lambda x: x["match_score"], reverse=True)
        return candidates[:n_candidates]

    @staticmethod
    def select_rbs(
        organism: str = "ecoli",
        strength: float = 1.0,          # 0-2.5 range
        n_candidates: int = 3,
    ) -> List[Dict]:
        """Select RBS elements matching target translation strength."""
        lib = ECOLI_RBS if organism == "ecoli" else YEAST_RBS
        candidates = []
        for name, props in lib.items():
            score = 1.0 - abs(props["strength"] - strength)
            candidates.append({"name": name, "match_score": round(score, 3), **props})
        candidates.sort(key=lambda x: x["match_score"], reverse=True)
        return candidates[:n_candidates]

    @staticmethod
    def balance_pathway_expression(
        enzyme_names: List[str],
        target_fluxes: Optional[List[float]] = None,
        organism: str = "ecoli",
    ) -> List[Dict]:
        """Recommend promoter+RBS combinations to balance a multi-gene pathway.

        Strategy: normalize relative expression to flux targets.
        Bottleneck enzymes get higher expression; downstream enzymes get lower
        to prevent intermediate accumulation.

        Args:
            enzyme_names: ordered list of enzyme names/genes in pathway
            target_fluxes: target flux for each enzyme (normalized to max=1.0)
                           If None, uses decreasing gradient (0.8 → 0.2)
            organism: host organism

        Returns:
            List of dicts with enzyme, recommended promoter, RBS, and rationale
        """
        n = len(enzyme_names)
        if not target_fluxes:
            # Default: first enzyme highest (substrate capture), taper downstream
            target_fluxes = [1.0 - 0.6 * i / max(n - 1, 1) for i in range(n)]

        recommendations = []
        for i, (name, flux) in enumerate(zip(enzyme_names, target_fluxes)):
            flux = max(0.01, min(2.0, flux))

            promoters = PartSelector.select_promoter(organism, strength=flux, n_candidates=2)
            rbs_list  = PartSelector.select_rbs(organism, strength=flux, n_candidates=2)

            rec = {
                "step":        i + 1,
                "enzyme":      name,
                "target_flux": round(flux, 2),
                "promoter":    promoters[0]["name"] if promoters else "J23100",
                "promoter_strength": promoters[0]["strength"] if promoters else 1.0,
                "rbs":         rbs_list[0]["name"] if rbs_list else "BBa_B0034",
                "rbs_strength": rbs_list[0]["strength"] if rbs_list else 1.0,
                "rationale": (
                    f"Step {i+1} ({name}): target relative flux {flux:.2f} → "
                    f"{'strong' if flux > 0.7 else 'medium' if flux > 0.3 else 'weak'} expression"
                ),
                "alternatives": [p["name"] for p in promoters[1:]],
            }
            recommendations.append(rec)

        return recommendations
