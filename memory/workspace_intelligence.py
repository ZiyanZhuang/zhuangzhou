"""
Workspace Intelligence 鈥?auto-detect project state from files
================================================================
LLM agent runtime pattern: read the workspace BEFORE every interaction to know
what stage the user is at and what they likely need next.

This module scans CWD for molecular biology artifacts and infers:
  - Project stage (preparation 鈫?simulation 鈫?analysis 鈫?reporting)
  - Available inputs (PDB, SDF, SMILES, trajectories)
  - Completed steps (docking done? MD done? PLIP done?)
  - Suggested next actions (based on what exists and what's missing)

The output is a compact text block injected into the LLM system prompt,
giving it instant situational awareness without the user having to explain.
"""

import fnmatch
import os
import time
from typing import Dict, List, Optional, Tuple


# File signatures that identify project stages
_STAGE_SIGNATURES = {
    "structure": {
        "patterns": ["*.pdb", "*.cif", "*.ent"],
        "label": "Structure available",
    },
    "ligand": {
        "patterns": ["*.sdf", "*.mol2", "*.pdbqt", "*.smi"],
        "label": "Ligand available",
    },
    "docking": {
        "patterns": ["*_docking/", "*docking*report*", "*_vina.pdbqt"],
        "label": "Docking completed",
    },
    "md_setup": {
        "patterns": ["topol.top", "*.mdp", "em.gro"],
        "label": "MD setup done",
    },
    "md_running": {
        "patterns": ["md.log", "*.xtc", "*.trr"],
        "label": "MD trajectory available",
    },
    "md_analysis": {
        "patterns": ["*rmsd*", "*rmsf*", "*gyration*", "*_analysis/"],
        "label": "MD analysis done",
    },
    "plip": {
        "patterns": ["*_plip/", "*plip_report*"],
        "label": "PLIP analysis done",
    },
    "visualization": {
        "patterns": ["*.png", "*.pml", "*.cxc", "*_viz/"],
        "label": "Visualization available",
    },
    "report": {
        "patterns": ["*.docx", "*_report.md", "*_report/"],
        "label": "Report generated",
    },
    "mutation": {
        "patterns": ["*_mutant*", "*mutate*", "*_minimized*"],
        "label": "Mutagenesis done",
    },
}

# Stage progression order
_STAGE_ORDER = [
    "structure",
    "ligand",
    "mutation",
    "docking",
    "plip",
    "md_setup",
    "md_running",
    "md_analysis",
    "visualization",
    "report",
]

# Suggested next steps based on completed stages
_NEXT_STEPS = {
    frozenset(["structure"]): [
        "analyze_mutation (mutation analysis)",
        "smart_docking (requires ligand SDF)",
        "auto_md (molecular dynamics setup)",
        "plip_analyze (interaction analysis)",
    ],
    frozenset(["structure", "ligand"]): [
        "smart_docking (protein-ligand docking)",
        "auto_md (protein-ligand complex MD)",
    ],
    frozenset(["structure", "docking"]): [
        "plip_analyze (analyze docking-result interactions)",
        "pymol_render (visualize docking results)",
        "auto_md (validate docking pose)",
    ],
    frozenset(["structure", "docking", "plip"]): [
        "pymol_render (render interactions)",
        "generate_docking_report (generate docking report)",
    ],
    frozenset(["structure", "md_running"]): [
        "gromacs_process_trajectory (trajectory processing)",
        "gromacs_analysis (RMSD/RMSF/Rg analysis)",
    ],
    frozenset(["structure", "md_running", "md_analysis"]): [
        "gromacs_visualize (trajectory animation)",
        "generate_md_report (generate MD report)",
    ],
}


class WorkspaceIntelligence:
    """Scan workspace and infer project state for LLM context injection."""

    def __init__(self, work_dir: str = None):
        self.work_dir = work_dir or os.getcwd()
        self._cache_ttl = 300
        self._cached_scan: Dict = {}
        self._cached_at: float = 0.0

    def scan(self) -> Dict:
        """Scan the workspace and return structured intelligence."""
        now = time.time()
        if self._cached_scan and (now - self._cached_at) < self._cache_ttl:
            return dict(self._cached_scan)
        files = self._list_files()
        stages = self._detect_stages(files)
        key_files = self._identify_key_files(files)
        next_steps = self._suggest_next(stages)
        subdirs = self._list_subdirs()

        result = {
            "stages": stages,
            "key_files": key_files,
            "next_steps": next_steps,
            "subdirs": subdirs,
            "total_files": len(files),
        }
        self._cached_scan = dict(result)
        self._cached_at = now
        return result

    def get_context_block(self) -> str:
        """Generate a compact text block for system prompt injection.

        Returns empty string if workspace has no relevant files.
        """
        info = self.scan()

        if not info["stages"] and info["total_files"] < 3:
            return ""

        parts = []

        # Stage summary
        if info["stages"]:
            stage_str = ", ".join(info["stages"])
            parts.append(f"椤圭洰闃舵: {stage_str}")

        # Key files
        if info["key_files"]:
            files_str = ", ".join(info["key_files"][:10])
            parts.append(f"鍏抽敭鏂囦欢: {files_str}")

        # Output subdirectories
        if info["subdirs"]:
            dirs_str = ", ".join(info["subdirs"][:8])
            parts.append(f"杈撳嚭鐩綍: {dirs_str}")

        # Suggested next steps
        if info["next_steps"]:
            steps_str = " | ".join(info["next_steps"][:3])
            parts.append(f"寤鸿涓嬩竴姝? {steps_str}")

        if not parts:
            return ""

        return "### 宸ヤ綔绌洪棿鎰熺煡\n" + "\n".join(f"- {p}" for p in parts)

    def _list_files(self) -> List[str]:
        """List files in work_dir + one level of subdirectories.

        Prefixes subdir files with their parent name so pattern matching
        still works on the filename part. Scans subdirs that look like
        output directories to catch MD trajectories, analysis results, etc.
        """
        try:
            entries = os.listdir(self.work_dir)
        except OSError:
            return []

        files = list(entries)

        # Also check first-level output subdirs for key file types
        for entry in entries:
            # Skip entries that are path traversals
            if os.path.basename(entry) != entry:
                continue
            full = os.path.join(self.work_dir, entry)
            if os.path.isdir(full) and not os.path.islink(full):
                try:
                    for sub in os.listdir(full):
                        files.append(sub)  # add filename (without path) for extension matching
                except OSError:
                    pass

        return files

    def _list_subdirs(self) -> List[str]:
        """List immediate subdirectories that look like outputs."""
        try:
            entries = os.listdir(self.work_dir)
        except OSError:
            return []

        output_suffixes = (
            "_docking",
            "_plip",
            "_viz",
            "_analysis",
            "_md",
            "_report",
            "_mutant",
            "_minimized",
            "_cgenff",
            "_conformers",
        )
        dirs = []
        for e in entries:
            full = os.path.join(self.work_dir, e)
            if os.path.isdir(full) and any(e.endswith(s) for s in output_suffixes):
                dirs.append(e)
        return sorted(dirs)

    def _detect_stages(self, files: List[str]) -> List[str]:
        """Detect which project stages are present."""
        detected = []
        files_lower = {f.lower() for f in files}

        # Also check subdirectories
        try:
            all_entries = os.listdir(self.work_dir)
        except OSError:
            all_entries = []

        for stage in _STAGE_ORDER:
            sig = _STAGE_SIGNATURES[stage]
            for pattern in sig["patterns"]:
                if self._pattern_matches(pattern, files_lower, all_entries):
                    detected.append(stage)
                    break

        return detected

    def _pattern_matches(
        self, pattern: str, files_lower: set, all_entries: List[str]
    ) -> bool:
        """Check if a glob-like pattern matches any file or directory.

        Uses fnmatch for correct wildcard handling (including mid-string * like *docking*report*).
        """
        if pattern.endswith("/"):
            # Directory pattern 鈥?match against directory names only
            dir_pattern = pattern.rstrip("/")
            return any(
                fnmatch.fnmatch(e.lower(), dir_pattern.lower())
                and os.path.isdir(os.path.join(self.work_dir, e))
                for e in all_entries
            )

        # File pattern 鈥?match against all collected filenames
        pat_lower = pattern.lower()
        return any(fnmatch.fnmatch(f, pat_lower) for f in files_lower)

    def _identify_key_files(self, files: List[str]) -> List[str]:
        """Identify key molecular biology files."""
        key_extensions = {
            ".pdb",
            ".sdf",
            ".mol2",
            ".gro",
            ".top",
            ".xtc",
            ".pdbqt",
            ".fasta",
            ".docx",
        }
        key = []
        for f in sorted(files):
            ext = os.path.splitext(f)[1].lower()
            if ext in key_extensions:
                key.append(f)
        return key[:15]

    def _suggest_next(self, stages: List[str]) -> List[str]:
        """Suggest next actions based on completed stages."""
        stage_set = frozenset(stages)

        # Try exact match first, then subset matches
        if stage_set in _NEXT_STEPS:
            return _NEXT_STEPS[stage_set]

        # Find best matching subset
        best_match = None
        best_size = 0
        for key, steps in _NEXT_STEPS.items():
            if key.issubset(stage_set) and len(key) > best_size:
                best_match = steps
                best_size = len(key)

        if best_match:
            return best_match

        # Default suggestions based on individual stages
        if not stages:
            return ["download PDB structure (get_pdb_entry)", "search protein info (get_protein_profile)"]
        if "structure" in stages and "md_running" not in stages:
            return ["auto_md (automated molecular dynamics)", "smart_docking (molecular docking)"]
        return []

