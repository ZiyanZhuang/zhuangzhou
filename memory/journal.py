"""
Journal & Memory Layer — Lightweight Event Log
==============================================
Journal remains a fast append-only event log for UI/runtime inspection.
Full-fidelity archival is now normalized into SessionArchive; the legacy
FullFidelityArchive class is kept only as a compatibility adapter.
"""

import json
import os
from zhuangzhou.utils.robustness import redact_secrets
from datetime import datetime
from typing import List

# Base directories
JOURNAL_DIR = os.path.expanduser("~/.zhuangzhou/journal")


class Journal:
    """Persistent, truncated journal for quick UI retrieval and history-aware prompts."""

    def __init__(self, journal_dir: str = None):
        self.journal_dir = journal_dir or JOURNAL_DIR
        os.makedirs(self.journal_dir, exist_ok=True)
        self._today_file = None
        self._today_date = None

    def _get_file(self):
        today = datetime.now().strftime("%Y-%m-%d")
        if today != self._today_date:
            self._today_date = today
            self._today_file = os.path.join(self.journal_dir, f"{today}.jsonl")
        return self._today_file

    def log(self, event_type: str, data: dict):
        """Append a record to the daily journal."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            **data,
        }
        with open(self._get_file(), "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())

    # Truncated logging for quick indexing
    def log_skill(self, skill_name: str, params: dict, success: bool, elapsed: float):
        self.log(
            "skill",
            {
                "skill": skill_name,
                "params": {k: str(v)[:100] for k, v in redact_secrets(params).items()},
                "success": success,
                "elapsed": elapsed,
            },
        )

    def log_command(self, command: str):
        self.log("command", {"cmd": command})


class FullFidelityArchive:
    """Compatibility adapter that now writes into SessionArchive.

    This keeps older call sites working while converging on a single
    durable archive representation under memory/archive.py.
    """

    def record_turn(
        self,
        user_input: str,
        thought: str,
        skills: List[dict],
        output: str,
        errors: List[str] = None,
    ):
        """Record a full conversation turn into the shared SessionArchive."""
        from zhuangzhou.memory.registry import MemoryRegistry

        archive = MemoryRegistry.get().session_archive
        archive.begin_interaction(
            user_input,
            {
                "reasoning_chain_preview": (thought or "")[:1000],
                "final_response_preview": (output or "")[:1000],
                "working_dir": os.getcwd(),
                "errors": errors or [],
                "source": "legacy_full_fidelity_adapter",
            },
        )
        for item in skills or []:
            if not isinstance(item, dict):
                continue
            tool_name = item.get("tool") or item.get("name") or item.get('skill') or "unknown_tool"
            params = item.get("params", {}) if isinstance(item, dict) else {}
            result = item.get("result", {}) if isinstance(item, dict) else {}
            elapsed_ms = item.get("elapsed_ms", item.get('elapsed', 0) * 1000)
            success = item.get("success", True) if isinstance(item, dict) else True
            archive.record_tool_call(tool_name, params, result, elapsed_ms, success)
        if thought:
            archive.record_insight(thought[:1500], category="reasoning_chain")
        if output:
            archive.record_insight(output[:1500], category="final_response")
        for err in errors or []:
            archive.record_insight(str(err)[:500], category="error")
        archive.end_interaction("complete" if not errors else "complete_with_errors")
        return archive.save()


# Global instances
journal = Journal()
archive = FullFidelityArchive()
