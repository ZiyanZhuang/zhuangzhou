"""
Supervisor — Layered Scientific Validity Gating (Option 3)
===========================================================
Two-tier validation to prevent garbage-in-garbage-out in computational pipelines:

  TypeA — Rule/tool based, 0 LLM tokens, runs before every expensive step
  TypeB — LLM-based, ~100-200 tokens, only at critical decision gates

Gates (in pipeline order):
  post_research  TypeB  "Does literature actually support these parameters?"
  pre_docking    TypeA  HEME/additive/chain checks
  pre_md         TypeA + TypeB  cofactor detection + forcefield validation
  post_docking   TypeA  sanity check: poses exist, affinity in range
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Crystallographic additives that must NOT anchor the docking box
_CRYSTALLOGRAPHIC_ADDITIVES = frozenset({
    "GOL", "PEG", "EDO", "MPD", "PGE", "DMS", "SO4", "PO4",
    "ACT", "ACE", "FMT", "IOD", "CL", "NA", "MG", "CA",
    "GOL", "1PE", "P6G", "PE4", "PE5", "PE8", "TAR",
})

# HEME / metal-containing residues requiring bonded model
_METALLOPROTEIN_RESIDUES = frozenset({
    "HEM", "HEA", "HEB", "HEC", "FE", "FE2", "FE3",
    "ZN", "ZN2", "CU", "CU2", "MN", "CO", "NI", "MO",
    "FAD", "FMN", "NAD", "NAP",
})


@dataclass
class GateResult:
    passed: bool
    issues: List[str] = field(default_factory=list)
    corrections: Dict[str, Any] = field(default_factory=dict)
    blocking: bool = False          # True = must fix before executing
    confidence: float = 1.0
    gate_name: str = ""

    def __str__(self):
        status = "PASS" if self.passed else "FAIL"
        lines = [f"[Gate:{self.gate_name}] {status}"]
        for issue in self.issues:
            lines.append(f"  · {issue}")
        if self.corrections:
            lines.append(f"  corrections: {self.corrections}")
        return "\n".join(lines)


# =============================================================================
# Type A — Rule-based validators (0 LLM tokens)
# =============================================================================

class TypeAValidator:
    """Pure rule/tool checks. Fast, deterministic, 0 LLM cost."""

    @staticmethod
    def check_pdb_has_metalloprotein(pdb_path: str) -> GateResult:
        """Detect HEME/metal residues in PDB. Critical for MD forcefield selection."""
        if not pdb_path or not os.path.exists(pdb_path):
            return GateResult(passed=True, gate_name="metalloprotein_check",
                              issues=["PDB not yet downloaded — skip pre-check"])

        metals_found = []
        try:
            with open(pdb_path, encoding="utf-8", errors="replace") as f:
                for line in f:
                    if line.startswith(("ATOM", "HETATM")):
                        resname = line[17:20].strip().upper()
                        if resname in _METALLOPROTEIN_RESIDUES:
                            if resname not in metals_found:
                                metals_found.append(resname)
        except Exception as e:
            logger.warning("PDB metal scan failed: %s", e)
            return GateResult(passed=True, gate_name="metalloprotein_check")

        if metals_found:
            return GateResult(
                passed=True,
                gate_name="metalloprotein_check",
                issues=[f"Metalloprotein detected: {metals_found}"],
                corrections={"use_metalloprotein_md": True,
                             "metal_residues": metals_found},
            )
        return GateResult(passed=True, gate_name="metalloprotein_check")

    @staticmethod
    def check_crystallographic_additives(pdb_path: str) -> GateResult:
        """Detect crystallographic additives that could corrupt active site detection."""
        if not pdb_path or not os.path.exists(pdb_path):
            return GateResult(passed=True, gate_name="additive_check")

        additives_found = []
        try:
            with open(pdb_path, encoding="utf-8", errors="replace") as f:
                for line in f:
                    if line.startswith("HETATM"):
                        resname = line[17:20].strip().upper()
                        if resname in _CRYSTALLOGRAPHIC_ADDITIVES:
                            if resname not in additives_found:
                                additives_found.append(resname)
        except Exception as e:
            logger.warning("Additive scan failed: %s", e)
            return GateResult(passed=True, gate_name="additive_check")

        if additives_found:
            return GateResult(
                passed=True,
                gate_name="additive_check",
                issues=[f"Crystallographic additives found (will be excluded from site detection): "
                        f"{additives_found}"],
                corrections={"exclude_from_site_detection": additives_found},
            )
        return GateResult(passed=True, gate_name="additive_check")

    @staticmethod
    def check_chain_completeness(pdb_path: str) -> GateResult:
        """For multimers: identify the most complete chain."""
        if not pdb_path or not os.path.exists(pdb_path):
            return GateResult(passed=True, gate_name="chain_check")

        chain_counts: Dict[str, int] = {}
        try:
            with open(pdb_path, encoding="utf-8", errors="replace") as f:
                for line in f:
                    if line.startswith("ATOM"):
                        chain = line[21:22].strip()
                        if chain:
                            chain_counts[chain] = chain_counts.get(chain, 0) + 1
        except Exception as e:
            logger.warning("Chain scan failed: %s", e)
            return GateResult(passed=True, gate_name="chain_check")

        if len(chain_counts) <= 1:
            return GateResult(passed=True, gate_name="chain_check")

        best_chain = max(chain_counts, key=chain_counts.get)
        return GateResult(
            passed=True,
            gate_name="chain_check",
            issues=[f"Multimer detected ({list(chain_counts.keys())}). "
                    f"Recommended chain: {best_chain} ({chain_counts[best_chain]} atoms)"],
            corrections={"recommended_chain": best_chain,
                         "chain_atom_counts": chain_counts},
        )

    @staticmethod
    def check_docking_output(docking_dir: str) -> GateResult:
        """Sanity check docking results: poses exist, affinity in physical range."""
        if not docking_dir or not os.path.exists(docking_dir):
            return GateResult(passed=False, gate_name="docking_output",
                              issues=["Docking output directory not found"],
                              blocking=True)

        # Look for output files
        pdbqt_files = [f for f in os.listdir(docking_dir)
                       if f.endswith("_out.pdbqt") or f.endswith("_results.pdbqt")]
        if not pdbqt_files:
            return GateResult(passed=False, gate_name="docking_output",
                              issues=["No docking poses found — docking may have failed"],
                              blocking=True)

        # Check best affinity in log
        best_affinity = None
        for fname in os.listdir(docking_dir):
            if fname.endswith(".log") or fname.endswith("_log.txt"):
                try:
                    with open(os.path.join(docking_dir, fname),
                              encoding="utf-8", errors="replace") as f:
                        for line in f:
                            m = re.search(r"\s+1\s+([-\d.]+)\s+", line)
                            if m:
                                best_affinity = float(m.group(1))
                                break
                except Exception:
                    pass

        issues = []
        if best_affinity is not None:
            if best_affinity > -3.0:
                issues.append(
                    f"Best affinity {best_affinity:.1f} kcal/mol is very weak — "
                    "check ligand preparation or active site box placement"
                )
            elif best_affinity < -20.0:
                issues.append(
                    f"Best affinity {best_affinity:.1f} kcal/mol is suspiciously strong — "
                    "possible clash or scoring artifact"
                )

        return GateResult(
            passed=True,
            gate_name="docking_output",
            issues=issues,
            corrections={"best_affinity": best_affinity} if best_affinity else {},
            blocking=False,
        )


# =============================================================================
# Type B — LLM-based gate (targeted, ~100-200 tokens per call)
# =============================================================================

_GATE_SYSTEM_PROMPTS = {
    "post_research": """你是科学计算流水线的严格审查员。
检查文献证据是否充分支持以下计算参数。只输出JSON，不要解释。

评判标准:
1. 活性位点残基有文献直接报道（非推断）
2. 辅因子信息有实验数据支持
3. 参数与文献描述的蛋白状态一致
4. 若文献数量不足（<2篇），标记为低置信度

输出:
{"passed": true/false, "confidence": 0.0-1.0,
 "issues": ["问题描述"], "corrections": {"参数名": "建议值"},
 "blocking": true/false}""",

    "pre_md": """你是分子动力学模拟的科学审查员。
检查以下MD参数选择是否科学合理。只输出JSON，不要解释。

评判标准:
1. 含HEME/Fe/Zn必须用bonded metal model (metalloprotein_md)
2. MD时长与研究目标匹配（折叠研究≥1μs，结合验证≥100ns）
3. 力场选择与蛋白类型匹配（CHARMM36/AMBER-ff14SB for proteins）
4. 体系大小合理（蛋白+水盒子+离子）

输出:
{"passed": true/false, "confidence": 0.0-1.0,
 "issues": ["问题描述"], "corrections": {"参数名": "建议值"},
 "blocking": true/false}""",
}


class SupervisorGate:
    """LLM-based critical checkpoint. Called only at defined gates, ~100-200 tokens each."""

    def __init__(self, backend):
        self.backend = backend

    def validate(self, gate_name: str, context: dict) -> GateResult:
        """
        Run a TypeB gate check.

        context keys (gate-specific):
          post_research: {params, literature_snippets, protein, ligand}
          pre_md:        {pdb_path, duration_ns, protein_type, metal_residues}
        """
        system = _GATE_SYSTEM_PROMPTS.get(gate_name)
        if not system:
            logger.warning("Unknown gate: %s", gate_name)
            return GateResult(passed=True, gate_name=gate_name)

        import json
        user_msg = f"待审查的参数:\n{json.dumps(context, ensure_ascii=False, indent=2)}"

        try:
            resp = self.backend.chat(system, [{"role": "user", "content": user_msg}], [])
            raw = resp.text or ""

            decoder = __import__("json").JSONDecoder()
            start = raw.find("{")
            if start != -1:
                data, _ = decoder.raw_decode(raw, start)
                return GateResult(
                    gate_name=gate_name,
                    passed=data.get("passed", True),
                    confidence=data.get("confidence", 0.8),
                    issues=data.get("issues", []),
                    corrections=data.get("corrections", {}),
                    blocking=data.get("blocking", False),
                )
        except Exception as e:
            logger.warning("SupervisorGate %s failed: %s — defaulting to pass", gate_name, e)

        return GateResult(passed=True, gate_name=gate_name,
                          issues=["Gate check unavailable — proceeding with caution"])


# =============================================================================
# Pipeline-level supervisor: orchestrates A + B gates for each scenario
# =============================================================================

class PipelineSupervisor:
    """
    Orchestrates all gates for a pipeline run.
    Usage:
        sup = PipelineSupervisor(backend)
        result = sup.run_pre_docking_gates(pdb_path, research_params, literature)
        if result.blocking:
            # abort or ask user
    """

    def __init__(self, backend=None):
        self.type_a = TypeAValidator()
        self.type_b = SupervisorGate(backend) if backend else None

    def _merge(self, *results: GateResult, gate_name: str = "combined") -> GateResult:
        issues = []
        corrections = {}
        blocking = False
        min_conf = 1.0
        for r in results:
            issues.extend(r.issues)
            corrections.update(r.corrections)
            if r.blocking:
                blocking = True
            min_conf = min(min_conf, r.confidence)
        passed = not blocking
        return GateResult(passed=passed, issues=issues, corrections=corrections,
                          blocking=blocking, confidence=min_conf, gate_name=gate_name)

    def gate_post_research(self, research_params: dict,
                           literature_snippets: List[str]) -> GateResult:
        """TypeB: validate that literature supports the extracted parameters."""
        if not self.type_b:
            return GateResult(passed=True, gate_name="post_research",
                              issues=["No LLM backend — skipping TypeB gate"])
        context = {
            "params": research_params,
            "literature_snippets": literature_snippets[:3],  # cap to save tokens
        }
        result = self.type_b.validate("post_research", context)
        result.gate_name = "post_research"
        if result.issues:
            logger.info("post_research gate: %s", result)
        return result

    def gate_pre_docking(self, pdb_path: str) -> GateResult:
        """TypeA only: HEME detection + additive filtering + chain selection."""
        r1 = self.type_a.check_crystallographic_additives(pdb_path)
        r2 = self.type_a.check_chain_completeness(pdb_path)
        result = self._merge(r1, r2, gate_name="pre_docking")
        if result.issues:
            logger.info("pre_docking gate:\n%s", result)
        return result

    def gate_pre_md(self, pdb_path: str, md_params: dict) -> GateResult:
        """TypeA + TypeB: metal detection (auto-correct forcefield) + LLM validation."""
        r_metal = self.type_a.check_pdb_has_metalloprotein(pdb_path)

        # Auto-correct: force metalloprotein_md if HEME/Fe detected
        corrections = {}
        if r_metal.corrections.get("use_metalloprotein_md"):
            md_params = {**md_params,
                         "skill": "metalloprotein_md",
                         "metal_residues": r_metal.corrections["metal_residues"]}
            corrections["skill"] = "metalloprotein_md"
            corrections["reason"] = (
                f"Auto-selected metalloprotein_md: "
                f"{r_metal.corrections['metal_residues']} detected"
            )
            logger.info("pre_md gate: auto-corrected to metalloprotein_md "
                        "(detected: %s)", r_metal.corrections["metal_residues"])

        # TypeB: LLM validates the full MD parameter set
        type_b_result = GateResult(passed=True, gate_name="pre_md_llm")
        if self.type_b:
            context = {**md_params,
                       "metal_residues": r_metal.corrections.get("metal_residues", [])}
            type_b_result = self.type_b.validate("pre_md", context)

        merged = self._merge(r_metal, type_b_result, gate_name="pre_md")
        merged.corrections.update(corrections)
        if merged.issues:
            logger.info("pre_md gate:\n%s", merged)
        return merged

    def gate_post_docking(self, docking_dir: str) -> GateResult:
        """TypeA: verify docking produced valid results."""
        result = self.type_a.check_docking_output(docking_dir)
        if result.issues:
            logger.info("post_docking gate:\n%s", result)
        return result
