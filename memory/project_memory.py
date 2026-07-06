"""
Project Memory 鈥?project-context-style accumulating plain text knowledge
=====================================================================
LLM agent runtime's killer insight: plain text > vector DB for agent memory.
High signal-to-noise ratio, directly readable, no retrieval noise.

This module maintains a plain text file that accumulates learned knowledge:
  - Enzyme-specific notes (active sites, cofactors, known mutations)
  - Successful workflow patterns ("for P450 docking, use these params")
  - Common pitfalls and solutions
  - Tool usage tips learned from errors

Structure:
  ~/.zhuangzhou/project_memory.md   鈥?the main knowledge file
  ~/.zhuangzhou/enzyme_notes/       鈥?per-enzyme accumulated notes
    CYP3A4.md
    lipaseB.md
    ...

Why plain text:
  1. LLM can read it directly as context (no embedding/retrieval step)
  2. Human can read and edit it (transparent, debuggable)
  3. No infrastructure dependency (no vector DB to maintain)
  4. Perfect for prefix caching (stable prefix = cheap repeated reads)
"""

import os
import re
import json
import urllib.request
from datetime import datetime
from typing import Dict, List, Optional

PROJECT_MEMORY_DIR = os.path.expanduser("~/.zhuangzhou")
PROJECT_MEMORY_FILE = os.path.join(PROJECT_MEMORY_DIR, "project_memory.md")
ENZYME_NOTES_DIR = os.path.join(PROJECT_MEMORY_DIR, "enzyme_notes")

# v0.7.0: Primary project-local memory file
LOCAL_PROJECT_FILE = "ZHUANGZHOU.md"


class ProjectMemory:
    """
    Accumulating plain text memory 鈥?gets richer with every session.

    Like ZHUANGZHOU.md but for enzyme engineering knowledge.
    Structured as markdown sections for easy LLM consumption.
    """

    def __init__(self, memory_dir: str = None):
        self.memory_dir = memory_dir or PROJECT_MEMORY_DIR
        self.enzyme_dir = os.path.join(self.memory_dir, "enzyme_notes")
        os.makedirs(self.memory_dir, exist_ok=True)
        os.makedirs(self.enzyme_dir, exist_ok=True)

        # Priority: ZHUANGZHOU.md in CWD > project_memory.md in global dir
        if os.path.exists(LOCAL_PROJECT_FILE):
            self.memory_file = os.path.abspath(LOCAL_PROJECT_FILE)
        else:
            self.memory_file = os.path.join(self.memory_dir, "project_memory.md")

        if not os.path.exists(self.memory_file):
            if self.memory_file.endswith("ZHUANGZHOU.md"):
                self._init_zhuangzhou_md()
            else:
                self._init_memory_file()

    def _init_memory_file(self):
        """Create the initial project memory file."""
        content = """# Zhuangzhou Project Memory

> Auto-maintained by Zhuangzhou. Editable by user or agent.
> This file is read as context before every interaction.

## Learned Patterns

## Common Pitfalls

## Tool Tips

## Enzyme Notes
"""
        with open(self.memory_file, "w", encoding="utf-8") as f:
            f.write(content)

    def _init_zhuangzhou_md(self):
        """Create the v0.7 structured template for ZHUANGZHOU.md."""
        content = """# Project Context
> This is the Project Constitution for Zhuangzhou v0.7.
> Agent MUST follow the norms and verification steps defined here.

## Environment
(Use /init to populate this section with local tool paths and file summary)

## Lab Norms (Experimental Standards)
- **Precision:** All docking results must be reported with RMSD if available.
- **Safety:** Do not delete any .pdb or .sdf files without explicit confirmation.
- **Documentation:** Every successful simulation should be logged in the Enzyme Notes.

## Verification Steps
- [ ] Tool availability check (GROMACS, Vina, Uni-Mol)
- [ ] Structural integrity check for input PDBs
- [ ] Force field parameter validation
"""
        with open(self.memory_file, "w", encoding="utf-8") as f:
            f.write(content)

    def ensure_local_file(self):
        """Force creation of ZHUANGZHOU.md in CWD if it doesn't exist."""
        if not os.path.exists(LOCAL_PROJECT_FILE):
            self.memory_file = os.path.abspath(LOCAL_PROJECT_FILE)
            self._init_zhuangzhou_md()
            return True
        return False

    # 鈹€鈹€ Read 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    def get_context(self, enzyme: str = None, max_lines: int = 200) -> str:
        """
        Build a context string from accumulated memory.
        This is what gets injected into the LLM context before each interaction.
        """
        parts = []

        # Main project memory (truncated to max_lines)
        if os.path.exists(self.memory_file):
            self.compact_if_needed()
            with open(self.memory_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
            parts.append("".join(lines[:max_lines]))

        kg_patterns = self._get_local_graph_patterns(limit=5)
        if kg_patterns:
            parts.append(
                "## Local KG Patterns\n"
                + "\n".join(f"- {item}" for item in kg_patterns)
            )

        # Enzyme-specific notes
        if enzyme:
            enzyme_file = os.path.join(self.enzyme_dir, f"{enzyme}.md")
            if os.path.exists(enzyme_file):
                with open(enzyme_file, "r", encoding="utf-8") as f:
                    parts.append(f"\n## {enzyme} Notes\n" + f.read())

        return "\n".join(parts)

    def get_enzyme_notes(self, enzyme: str) -> str:
        """Get notes for a specific enzyme."""
        path = os.path.join(self.enzyme_dir, f"{enzyme}.md")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        return ""

    def list_enzymes(self) -> List[str]:
        """List enzymes that have accumulated notes."""
        if not os.path.exists(self.enzyme_dir):
            return []
        return [f[:-3] for f in os.listdir(self.enzyme_dir) if f.endswith(".md")]

    # 鈹€鈹€ Write (append-only, accumulating) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    def add_pattern(self, pattern: str):
        """Add a learned pattern (successful approach, useful parameter combo)."""
        self._append_to_section("Learned Patterns", pattern)

    def add_pitfall(self, pitfall: str):
        """Add a common pitfall (error + solution)."""
        self._append_to_section("Common Pitfalls", pitfall)

    def add_tool_tip(self, tip: str):
        """Add a tool usage tip."""
        self._append_to_section("Tool Tips", tip)

    def add_enzyme_note(self, enzyme: str, note: str):
        """Add a note for a specific enzyme."""
        path = os.path.join(self.enzyme_dir, f"{enzyme}.md")
        date = datetime.now().strftime("%Y-%m-%d")
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"# {enzyme}\n\n")
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"- [{date}] {note}\n")

    def record_successful_workflow(
        self, enzyme: str, workflow: str, params: dict, result_summary: str
    ):
        """Record a successful workflow for future reference."""
        date = datetime.now().strftime("%Y-%m-%d")
        param_str = ", ".join(
            f"{k}={v}" for k, v in params.items() if not k.startswith("_")
        )
        entry = f"[{date}] {workflow}({param_str}) -> {result_summary}"
        self.add_pattern(entry)
        if enzyme:
            self.add_enzyme_note(enzyme, f"{workflow}: {result_summary}")

    def record_error_solution(self, error: str, solution: str):
        """Record an error and its solution for future avoidance."""
        date = datetime.now().strftime("%Y-%m-%d")
        self.add_pitfall(f"[{date}] {error} -> {solution}")

    def update_section(self, section: str, new_content: str):
        """Replace the content of a specific section in the memory file."""
        if not os.path.exists(self.memory_file):
            if self.memory_file.endswith("ZHUANGZHOU.md"):
                self._init_zhuangzhou_md()
            else:
                self._init_memory_file()

        with open(self.memory_file, "r", encoding="utf-8") as f:
            text = f.read()

        marker = f"## {section}"
        if marker not in text:
            text = text.rstrip() + f"\n\n{marker}\n{new_content}\n"
        else:
            # Find the section and the next section
            idx = text.index(marker)
            after_header = idx + len(marker)
            # Find next ## section
            next_section = text.find("\n## ", after_header)

            if next_section == -1:
                # Replace until end
                text = text[:after_header] + "\n" + new_content.strip() + "\n"
            else:
                # Replace between headers
                text = (
                    text[:after_header]
                    + "\n"
                    + new_content.strip()
                    + "\n"
                    + text[next_section:]
                )

        with open(self.memory_file, "w", encoding="utf-8") as f:
            f.write(text)

    # 鈹€鈹€ Internal 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    def _append_to_section(self, section: str, content: str):
        """Append a line to a specific section in project_memory.md."""
        if not os.path.exists(self.memory_file):
            self._init_memory_file()

        with open(self.memory_file, "r", encoding="utf-8") as f:
            text = f.read()

        marker = f"## {section}"
        if marker not in text:
            text += f"\n{marker}\n"

        # Find the section and the next section
        idx = text.index(marker)
        after_header = idx + len(marker)
        # Find next ## section
        next_section = text.find("\n## ", after_header)
        if next_section == -1:
            # Append at end
            text = text.rstrip() + f"\n- {content}\n"
        else:
            # Insert before next section
            text = text[:next_section] + f"- {content}\n" + text[next_section:]

        with open(self.memory_file, "w", encoding="utf-8") as f:
            f.write(text)

    def auto_compact(self, max_lines: int = 300):
        """Deduplicate and trim old entries when file grows too large.

        Unlike LLM-based compression, this is a deterministic pass that:
        1. Removes exact-duplicate lines within each section
        2. Trims oldest entries (keeping most recent) if still over limit
        """
        if not os.path.exists(self.memory_file):
            return
        with open(self.memory_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if len(lines) <= max_lines:
            return

        # Split into sections
        sections = []
        current_header = ""
        current_lines = []
        for line in lines:
            if line.startswith("## "):
                if current_header or current_lines:
                    sections.append((current_header, current_lines))
                current_header = line
                current_lines = []
            else:
                current_lines.append(line)
        if current_header or current_lines:
            sections.append((current_header, current_lines))

        # Deduplicate within each section (preserve order, keep last)
        rebuilt = []
        for header, body in sections:
            if header:
                rebuilt.append(header)
            seen = set()
            deduped = []
            for line in reversed(body):
                stripped = line.strip()
                if stripped and stripped.startswith("- "):
                    if stripped in seen:
                        continue
                    seen.add(stripped)
                deduped.append(line)
            deduped.reverse()
            rebuilt.extend(deduped)

        # If still over limit, trim oldest bullet entries from middle sections
        if len(rebuilt) > max_lines:
            excess = len(rebuilt) - max_lines
            trimmed = []
            removed = 0
            for line in rebuilt:
                if removed < excess and line.strip().startswith("- ["):
                    removed += 1
                    continue
                trimmed.append(line)
            rebuilt = trimmed

        with open(self.memory_file, "w", encoding="utf-8") as f:
            f.writelines(rebuilt)

    def compact_if_needed(self, max_lines: int = 300):
        """Compact project memory using local LLM when possible, fallback to heuristic compaction."""
        if not os.path.exists(self.memory_file):
            return False
        with open(self.memory_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if len(lines) <= max_lines:
            return False

        original = "".join(lines)
        summary = self._llm_compact(original)
        if not summary:
            self.auto_compact(max_lines=max_lines)
            return True

        compacted = (
            "# Project Memory\n\n"
            "> Auto-compacted to keep high signal-to-noise ratio.\n\n"
            f"Last compacted: {datetime.now().isoformat()}\n\n" + summary.strip() + "\n"
        )
        with open(self.memory_file, "w", encoding="utf-8") as f:
            f.write(compacted)
        return True

    def _llm_compact(self, content: str) -> str:
        payload = {
            "model": os.environ.get(
                "ZHUANGZHOU_MEMORY_SUMMARY_MODEL", "deepseek-r1:8b"
            ),
            "prompt": (
                "Compress the following project memory into a concise markdown summary. "
                "Keep sections for Learned Patterns, Common Pitfalls, Tool Tips, and Enzyme Notes. "
                "Preserve file paths, commands, and numeric results.\n\n"
                + content[:20000]
            ),
            "stream": False,
        }
        try:
            req = urllib.request.Request(
                "http://localhost:11434/api/generate",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=90) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            return str(result.get("response", "")).strip()
        except Exception:
            return ""

    def _get_local_graph_patterns(self, limit: int = 5) -> List[str]:
        try:
            from zhuangzhou.memory.kg_engine import LocalKGEngine

            return [
                f"{item['head']} --{item['relation']}--> {item['tail']}"
                for item in LocalKGEngine.top_patterns(limit=limit)
            ]
        except Exception:
            return []

    def summary(self) -> dict:
        """Summary of accumulated knowledge."""
        enzymes = self.list_enzymes()
        line_count = 0
        if os.path.exists(self.memory_file):
            with open(self.memory_file, "r", encoding="utf-8") as f:
                line_count = sum(1 for _ in f)
        return {
            "memory_lines": line_count,
            "enzymes_tracked": len(enzymes),
            "enzymes": enzymes,
            "memory_file": self.memory_file,
        }

