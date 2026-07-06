"""
Session Archive 鈥?complete interaction recording
===================================================
Records every tool call with full context, parameters, results, and timing.
This is the "lab notebook" that makes Zhuangzhou smarter over time.

LLM agent runtime pattern: every action is recorded, searchable, and learnable.

Structure:
  ~/.zhuangzhou/archive/
    2026-03-05_143022.json    (one file per session)
    2026-03-04_091500.json
    ...
    index.jsonl               (lightweight index for fast lookup)
"""

import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

ARCHIVE_DIR = os.path.expanduser("~/.zhuangzhou/archive")


class SessionArchive:
    """Records complete sessions 鈥?context, tool calls, results, insights."""

    def __init__(self, archive_dir: str = None):
        self.archive_dir = archive_dir or ARCHIVE_DIR
        os.makedirs(self.archive_dir, exist_ok=True)
        self.session_id = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.interactions: List[dict] = []
        self._current_interaction = None

    def begin_interaction(self, user_input: str, context_used: dict = None):
        """Start recording a new interaction."""
        self._current_interaction = {
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "context_used": context_used or {},
            "tool_calls": [],
            "outcome": None,
            "insights": [],
            "elapsed_ms": 0,
        }
        self._t0 = time.time()

    def record_tool_call(
        self,
        tool_name: str,
        params: dict,
        result: Any,
        elapsed_ms: float = 0,
        success: bool = True,
    ):
        """Record a tool execution within the current interaction."""
        if not self._current_interaction:
            return
        # Truncate large results for storage (increased from 5000 to 10000)
        result_str = json.dumps(result, ensure_ascii=False, default=str)
        if len(result_str) > 10000:
            result_str = result_str[:10000] + "...(truncated)"
            result = {"_truncated": True, "_preview": result_str}

        self._current_interaction["tool_calls"].append(
            {
                "tool": tool_name,
                "params": _safe_serialize(params),
                "result": result,
                "elapsed_ms": elapsed_ms,
                "success": success,
            }
        )

    def record_insight(self, insight: str, category: str = "general"):
        """Record a learned insight from this interaction."""
        if self._current_interaction:
            self._current_interaction["insights"].append(
                {
                    "text": insight,
                    "category": category,
                }
            )

    def end_interaction(self, outcome: str = "complete"):
        """Finish recording the current interaction."""
        if not self._current_interaction:
            return
        self._current_interaction["outcome"] = outcome
        self._current_interaction["elapsed_ms"] = (time.time() - self._t0) * 1000
        self.interactions.append(self._current_interaction)
        self._current_interaction = None

    def save(self):
        """Save session to disk and update the index idempotently."""
        if not self.interactions:
            return None
        data = {
            "session_id": self.session_id,
            "interaction_count": len(self.interactions),
            "interactions": self.interactions,
        }
        path = os.path.join(self.archive_dir, f"{self.session_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

        # Update index
        self._update_index()
        return path

    def _update_index(self):
        """Write session summary to the lightweight index without duplicates."""
        index_path = os.path.join(self.archive_dir, "index.jsonl")
        tools_used = set()
        entities = set()
        for ix in self.interactions:
            for tc in ix.get("tool_calls", []):
                tools_used.add(tc["tool"])
            import re

            mutations = re.findall(r"\b[A-Z]\d{1,4}[A-Z]\b", ix.get("user_input", ""))
            entities.update(mutations)

        entry = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "interactions": len(self.interactions),
            "tools_used": sorted(tools_used),
            "entities": sorted(entities),
            "insights": sum(len(ix.get("insights", [])) for ix in self.interactions),
        }
        entries = []
        if os.path.exists(index_path):
            with open(index_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        existing = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if existing.get("session_id") != self.session_id:
                        entries.append(existing)
        entries.append(entry)
        entries.sort(key=lambda item: item.get("timestamp", ""))
        with open(index_path, "w", encoding="utf-8") as f:
            for item in entries:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 鈹€鈹€ Query archived sessions 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    def search_history(self, query: str, limit: int = 10) -> List[dict]:
        """Search archived sessions by keyword (tools, entities, inputs)."""
        index_path = os.path.join(self.archive_dir, "index.jsonl")
        if not os.path.exists(index_path):
            return []
        query_lower = query.lower()
        matches = []
        with open(index_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # Score by keyword match
                text = json.dumps(entry).lower()
                if query_lower in text:
                    matches.append(entry)
        return matches[-limit:]

    def load_session(self, session_id: str) -> Optional[dict]:
        """Load a full archived session by ID."""
        path = os.path.join(self.archive_dir, f"{session_id}.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def get_recent_sessions(self, limit: int = 5) -> List[dict]:
        """Get summaries of recent sessions from index."""
        index_path = os.path.join(self.archive_dir, "index.jsonl")
        if not os.path.exists(index_path):
            return []
        entries = []
        with open(index_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return entries[-limit:]

    def get_tool_usage_stats(self) -> Dict[str, int]:
        """Get aggregated tool usage counts across all sessions."""
        index_path = os.path.join(self.archive_dir, "index.jsonl")
        if not os.path.exists(index_path):
            return {}
        from collections import Counter

        counts = Counter()
        with open(index_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entry = json.loads(line)
                        for t in entry.get("tools_used", []):
                            counts[t] += 1
                    except json.JSONDecodeError:
                        pass
        return dict(counts)

    def summary(self) -> dict:
        """Get archive summary."""
        files = [
            f
            for f in os.listdir(self.archive_dir)
            if f.endswith(".json") and f != "index.jsonl"
        ]
        return {
            "sessions_archived": len(files),
            "current_interactions": len(self.interactions),
            "archive_dir": self.archive_dir,
        }


def _safe_serialize(obj: Any, max_len: int = 500) -> Any:
    """Safely serialize params for archival, truncating large values."""
    if isinstance(obj, dict):
        return {k: _safe_serialize(v, max_len) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_serialize(v, max_len) for v in obj[:20]]
    if isinstance(obj, str) and len(obj) > max_len:
        return obj[:max_len] + "..."
    if isinstance(obj, set):
        return sorted(obj)[:20]
    return obj

