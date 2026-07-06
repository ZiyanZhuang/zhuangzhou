"""
Execution History — persistent skill execution log + task checkpoint
=====================================================================
Solves the critical problem: execution_log was IN-MEMORY ONLY in nlp/agent.py,
causing total loss of execution context between sessions.

Three capabilities:
  1. ExecutionHistory — append-only JSONL log of every skill call (persistent)
  2. TaskCheckpoint  — save/resume interrupted tasks (what was being done, what's left)
  3. WorkspaceTracker — track files created/modified during sessions

Structure:
  ~/.zhuangzhou/execution/
    execution_log.jsonl      (append-only, all skill calls ever)
    checkpoint.json          (current task state for resumption)
    workspace_files.json     (tracked file creations/modifications)
"""

import json
import logging
import os
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

EXECUTION_DIR = os.path.expanduser('~/.zhuangzhou/execution')


class ExecutionHistory:
    """Persistent, append-only log of all skill executions across sessions.

    Unlike the in-memory execution_log in AgentExecutor, this survives restarts.
    Used for:
      - Cross-session continuity ("what did I do last time?")
      - Failure pattern analysis with full context
      - Execution statistics and performance tracking
    """

    def __init__(self, execution_dir: str = None):
        self.execution_dir = execution_dir or EXECUTION_DIR
        os.makedirs(self.execution_dir, exist_ok=True)
        self._log_path = os.path.join(self.execution_dir, 'execution_log.jsonl')
        self._write_lock = threading.Lock()  # Thread-safe for parallel skill execution

    def record(self, skill_name: str, params: dict, success: bool,
               elapsed: float, output_files: List[str] = None,
               error: str = '', error_traceback: str = '',
               session_id: str = '', user_context: str = ''):
        """Record a skill execution to persistent storage."""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'unix': time.time(),
            'session_id': session_id,
            'skill': skill_name,
            'params': _safe_params(params),
            'success': success,
            'elapsed': round(elapsed, 2),
            'output_files': output_files or [],
            'user_context': user_context[:200] if user_context else '',
        }
        if not success:
            entry['error'] = error[:3000] if error else ''
            entry['error_traceback'] = error_traceback[:8000] if error_traceback else ''

        with self._write_lock:
            with open(self._log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry, ensure_ascii=False, default=str) + '\n')

    def get_recent(self, limit: int = 50, skill_filter: str = None,
                   only_failures: bool = False) -> List[dict]:
        """Get recent execution entries."""
        if not os.path.exists(self._log_path):
            return []
        entries = []
        with open(self._log_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Malformed JSONL entry in execution log (skipped): %.80s", line)
                    continue
                if skill_filter and entry.get('skill') != skill_filter:
                    continue
                if only_failures and entry.get('success', True):
                    continue
                entries.append(entry)
        return entries[-limit:]

    def get_last_session_log(self, session_id: str) -> List[dict]:
        """Get all executions from a specific session."""
        if not os.path.exists(self._log_path):
            return []
        entries = []
        with open(self._log_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if entry.get('session_id') == session_id:
                        entries.append(entry)
                except json.JSONDecodeError:
                    continue
        return entries

    def get_failure_analysis(self, skill_name: str = None, limit: int = 20) -> List[dict]:
        """Get detailed failure analysis with full context.

        Unlike KnowledgeStore.get_skill_failure_hints which only stores truncated strings,
        this returns the full error + traceback + params that caused the failure.
        """
        failures = self.get_recent(limit=limit * 3, skill_filter=skill_name,
                                   only_failures=True)
        # Group by error pattern
        patterns = {}
        for f in failures:
            err = f.get('error', '')
            # Extract error type (first line or first 80 chars)
            err_key = err.split('\n')[0][:80] if err else 'unknown'
            if err_key not in patterns:
                patterns[err_key] = {
                    'error_type': err_key,
                    'count': 0,
                    'examples': [],
                    'skills': set(),
                }
            patterns[err_key]['count'] += 1
            patterns[err_key]['skills'].add(f.get('skill', ''))
            if len(patterns[err_key]['examples']) < 3:
                patterns[err_key]['examples'].append({
                    'timestamp': f.get('timestamp'),
                    'skill': f.get('skill'),
                    'params': f.get('params'),
                    'traceback': f.get('error_traceback', ''),
                    'user_context': f.get('user_context', ''),
                })

        result = []
        for p in sorted(patterns.values(), key=lambda x: -x['count']):
            p['skills'] = sorted(p['skills'])
            result.append(p)
        return result[:limit]

    def get_session_ids(self, limit: int = 10) -> List[str]:
        """Get recent unique session IDs for resumption."""
        if not os.path.exists(self._log_path):
            return []
        sessions = []
        seen = set()
        # Read backwards for efficiency
        entries = []
        with open(self._log_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        for entry in reversed(entries):
            sid = entry.get('session_id', '')
            if sid and sid not in seen:
                seen.add(sid)
                sessions.append(sid)
                if len(sessions) >= limit:
                    break
        return sessions

    def summary(self) -> dict:
        """Get execution history summary."""
        if not os.path.exists(self._log_path):
            return {'total_executions': 0, 'log_path': self._log_path}
        total = 0
        successes = 0
        failures = 0
        skills = set()
        with open(self._log_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    total += 1
                    if entry.get('success'):
                        successes += 1
                    else:
                        failures += 1
                    skills.add(entry.get('skill', ''))
                except json.JSONDecodeError:
                    pass
        return {
            'total_executions': total,
            'successes': successes,
            'failures': failures,
            'unique_skills': len(skills),
            'log_path': self._log_path,
        }


class TaskCheckpoint:
    """Save and resume interrupted tasks.

    When a ReAct loop is interrupted (timeout, error, user abort),
    saves the current state so the next session can offer to continue.

    Checkpoint includes:
      - Original user request
      - Completed steps (skills + outputs)
      - Remaining plan (if detectable)
      - Key context (active PDB, ligand, mutations)
    """

    def __init__(self, execution_dir: str = None):
        self.execution_dir = execution_dir or EXECUTION_DIR
        os.makedirs(self.execution_dir, exist_ok=True)
        self._path = os.path.join(self.execution_dir, 'checkpoint.json')

    def save(self, user_request: str, completed_steps: List[dict],
             remaining_plan: List[str] = None, context: dict = None,
             session_id: str = ''):
        """Save a task checkpoint for later resumption."""
        data = {
            'timestamp': datetime.now().isoformat(),
            'session_id': session_id,
            'user_request': user_request[:500],
            'completed_steps': [_safe_step(s) for s in completed_steps[-20:]],
            'remaining_plan': remaining_plan or [],
            'context': _safe_params(context or {}),
            'status': 'interrupted',
        }
        with open(self._path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    def load(self) -> Optional[dict]:
        """Load the most recent checkpoint."""
        if not os.path.exists(self._path):
            return None
        try:
            with open(self._path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # Only return if not too old (< 24 hours)
            ts = data.get('timestamp', '')
            if ts:
                try:
                    cp_time = datetime.fromisoformat(ts)
                    age_hours = (datetime.now() - cp_time).total_seconds() / 3600
                    if age_hours > 24:
                        return None
                except (ValueError, TypeError):
                    pass
            return data
        except (json.JSONDecodeError, OSError):
            return None

    def clear(self):
        """Clear the checkpoint (task completed or abandoned)."""
        if os.path.exists(self._path):
            # Don't delete — mark as completed
            try:
                with open(self._path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                data['status'] = 'completed'
                data['completed_at'] = datetime.now().isoformat()
                with open(self._path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            except Exception:
                pass

    def has_pending(self) -> bool:
        """Check if there's an interrupted task to resume."""
        cp = self.load()
        return cp is not None and cp.get('status') == 'interrupted'


class WorkspaceTracker:
    """Track files created and modified during sessions.

    Gives Zhuangzhou "file awareness" — knows what it produced and where.
    Useful for:
      - Resuming work (finding previous outputs)
      - Cleanup suggestions
      - Preventing redundant computation
    """

    def __init__(self, execution_dir: str = None):
        self.execution_dir = execution_dir or EXECUTION_DIR
        os.makedirs(self.execution_dir, exist_ok=True)
        self._path = os.path.join(self.execution_dir, 'workspace_files.json')
        self._data = self._load()

    def _load(self) -> dict:
        if os.path.exists(self._path):
            try:
                with open(self._path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return {'files': {}}

    def _save(self):
        tmp = self._path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False, default=str)
        os.replace(tmp, self._path)

    def track_output(self, file_path: str, skill_name: str,
                     file_type: str = '', session_id: str = ''):
        """Track a file produced by a skill execution."""
        abs_path = os.path.abspath(file_path)
        self._data['files'][abs_path] = {
            'created': datetime.now().isoformat(),
            'skill': skill_name,
            'type': file_type or _guess_file_type(abs_path),
            'session_id': session_id,
            'exists': os.path.exists(abs_path),
        }
        # Limit to last 500 files
        if len(self._data['files']) > 500:
            sorted_files = sorted(self._data['files'].items(),
                                  key=lambda x: x[1].get('created', ''))
            self._data['files'] = dict(sorted_files[-500:])
        self._save()

    def find_outputs(self, skill_name: str = None, file_type: str = None,
                     session_id: str = None) -> List[dict]:
        """Find previously produced files, optionally filtered."""
        results = []
        for path, info in self._data.get('files', {}).items():
            if skill_name and info.get('skill') != skill_name:
                continue
            if file_type and info.get('type') != file_type:
                continue
            if session_id and info.get('session_id') != session_id:
                continue
            # Check if file still exists
            info['path'] = path
            info['still_exists'] = os.path.exists(path)
            results.append(info)
        return results

    def get_recent_outputs(self, limit: int = 20) -> List[dict]:
        """Get most recently created output files."""
        all_files = []
        for path, info in self._data.get('files', {}).items():
            info['path'] = path
            info['still_exists'] = os.path.exists(path)
            all_files.append(info)
        all_files.sort(key=lambda x: x.get('created', ''), reverse=True)
        return all_files[:limit]

    def summary(self) -> dict:
        total = len(self._data.get('files', {}))
        existing = sum(1 for p in self._data.get('files', {}) if os.path.exists(p))
        return {
            'total_tracked': total,
            'still_existing': existing,
            'tracker_path': self._path,
        }


# ── Helpers ─────────────────────────────────────────────────────────

def _safe_params(params: dict, max_val_len: int = 300) -> dict:
    """Safely serialize params for storage."""
    safe = {}
    for k, v in params.items():
        if k == 'config':
            continue
        if isinstance(v, str) and len(v) > max_val_len:
            safe[k] = v[:max_val_len] + '...'
        elif isinstance(v, (list, tuple)) and len(v) > 20:
            safe[k] = list(v[:20]) + ['...(truncated)']
        elif isinstance(v, set):
            safe[k] = sorted(v)[:20]
        else:
            safe[k] = v
    return safe


def _safe_step(step: dict) -> dict:
    """Safely serialize an execution step for checkpoint storage."""
    safe = {
        'skill': step.get('skill', ''),
        'success': step.get('success', False),
        'elapsed': step.get('elapsed', 0),
        'output_files': step.get('output_files', [])[:5],
    }
    params = step.get('params', {})
    safe['params'] = _safe_params(params)
    return safe


def _guess_file_type(path: str) -> str:
    """Guess file type from extension."""
    ext = os.path.splitext(path)[1].lower()
    type_map = {
        '.pdb': 'structure', '.pdbqt': 'docking_input',
        '.sdf': 'molecule', '.mol2': 'molecule',
        '.gro': 'gromacs', '.xtc': 'trajectory', '.trr': 'trajectory',
        '.top': 'topology', '.itp': 'topology', '.mdp': 'md_params',
        '.png': 'image', '.svg': 'image',
        '.md': 'report', '.docx': 'report',
        '.json': 'data', '.csv': 'data',
        '.log': 'log', '.txt': 'text',
        '.fasta': 'sequence', '.fa': 'sequence',
    }
    return type_map.get(ext, 'other')
