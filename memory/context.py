"""
Session Context — working memory
==================================
Tracks the current session state:
  - Active protein / PDB
  - Active mutations under analysis
  - Recent results
  - User preferences inferred from behavior

Persisted to disk so sessions can resume.

Structure:
  ~/.zhuangzhou/context/
    current_session.json
    session_history/
      session_20260305_143022.json
"""

import json
import os
import time
import shutil
from datetime import datetime
from typing import Any, Dict, List, Optional


CONTEXT_DIR = os.path.expanduser("~/.zhuangzhou/context")


class SessionContext:
    """Current session state — like working memory."""

    def __init__(self, context_dir: str = None):
        self.context_dir = context_dir or CONTEXT_DIR
        os.makedirs(self.context_dir, exist_ok=True)
        os.makedirs(os.path.join(self.context_dir, "session_history"), exist_ok=True)

        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.started_at = time.time()

        # Working state
        self.active_pdb: Optional[str] = None
        self.active_protein_name: Optional[str] = None
        self.active_binding_site: Optional[set] = None
        self.active_mutations: List[str] = []
        self.active_results: Dict[str, Any] = {}
        self.variables: Dict[str, Any] = {}  # user-set variables
        self.command_count: int = 0
        self._previous_session_id: str = ""

    def set(self, key: str, value: Any):
        """Set a context variable."""
        self.variables[key] = value
        self._auto_track(key, value)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a context variable."""
        return self.variables.get(key, default)

    def _auto_track(self, key: str, value: Any):
        """Auto-detect common patterns and update working state."""
        if key == "pdb" or (isinstance(value, str) and value.endswith(".pdb")):
            self.active_pdb = value
        if key == "protein_name":
            self.active_protein_name = value
        if key == "binding_site":
            self.active_binding_site = value
        if key == "mutations":
            self.active_mutations = value

    def track_result(self, command: str, result: Any):
        """Track a command result in active results."""
        self.active_results[command] = result
        self.command_count += 1
        # Keep only last 20 results
        if len(self.active_results) > 20:
            oldest = list(self.active_results.keys())[0]
            del self.active_results[oldest]

    def save(self):
        """Save current session to disk."""
        data = {
            "session_id": self.session_id,
            "started_at": self.started_at,
            "saved_at": time.time(),
            "active_pdb": self.active_pdb,
            "active_protein_name": self.active_protein_name,
            "active_binding_site": list(self.active_binding_site)
            if self.active_binding_site
            else None,
            "active_mutations": self.active_mutations,
            "variables": {k: str(v) for k, v in self.variables.items()},
            "command_count": self.command_count,
        }
        # Save as current
        current_path = os.path.join(self.context_dir, "current_session.json")
        with open(current_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def archive(self):
        """Archive current session and start fresh."""
        self.save()
        # Copy to history
        src = os.path.join(self.context_dir, "current_session.json")
        if os.path.exists(src):
            dst = os.path.join(
                self.context_dir, "session_history", f"session_{self.session_id}.json"
            )
            shutil.copy2(src, dst)

    def _apply_session_data(self, data: Dict[str, Any]):
        self.active_pdb = data.get("active_pdb")
        self.active_protein_name = data.get("active_protein_name")
        bs = data.get("active_binding_site")
        self.active_binding_site = set(bs) if bs else None
        self.active_mutations = data.get("active_mutations", [])
        self.variables = data.get("variables", {})
        self.command_count = int(data.get("command_count", 0) or 0)
        self._previous_session_id = data.get("session_id", "")

    def resume_current(self) -> bool:
        """Explicitly resume the current saved session state."""
        current_path = os.path.join(self.context_dir, "current_session.json")
        if not os.path.exists(current_path):
            return False
        try:
            with open(current_path, encoding="utf-8", errors="replace") as f:
                data = json.load(f)
            self._apply_session_data(data)
            return True
        except Exception:
            return False

    def list_recent_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """List recent archived sessions for explicit /resume."""
        history_dir = os.path.join(self.context_dir, "session_history")
        if not os.path.isdir(history_dir):
            return []
        sessions = []
        for name in os.listdir(history_dir):
            if not name.startswith("session_") or not name.endswith(".json"):
                continue
            path = os.path.join(history_dir, name)
            try:
                with open(path, encoding="utf-8", errors="replace") as f:
                    data = json.load(f)
                sessions.append(
                    {
                        "session_id": data.get("session_id", ""),
                        "saved_at": data.get("saved_at", 0),
                        "active_pdb": data.get("active_pdb"),
                        "protein": data.get("active_protein_name"),
                        "commands": int(data.get("command_count", 0) or 0),
                        "path": path,
                    }
                )
            except Exception:
                continue
        sessions.sort(key=lambda item: item.get("saved_at", 0), reverse=True)
        return sessions[:limit]

    def resume_session(self, session_id: str = "") -> bool:
        """Resume a specific archived session or the most recent one."""
        candidates = self.list_recent_sessions(limit=50)
        if not candidates:
            return False
        selected = None
        if session_id:
            for item in candidates:
                if item.get("session_id") == session_id:
                    selected = item
                    break
        else:
            selected = candidates[0]
        if not selected:
            return False
        try:
            with open(selected["path"], encoding="utf-8", errors="replace") as f:
                data = json.load(f)
            self._apply_session_data(data)
            return True
        except Exception:
            return False

    def get_summary(self) -> str:
        """Get a brief summary of current context."""
        parts = []
        if self.active_protein_name:
            parts.append(f"Protein: {self.active_protein_name}")
        if self.active_pdb:
            parts.append(f"PDB: {os.path.basename(self.active_pdb)}")
        if self.active_binding_site:
            parts.append(f"Sites: {sorted(self.active_binding_site)}")
        if self.active_mutations:
            parts.append(f"Mutations: {len(self.active_mutations)}")
        parts.append(f"Commands: {self.command_count}")
        return " | ".join(parts) if parts else "No active context"
