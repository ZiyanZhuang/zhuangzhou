"""
Local Knowledge Graph Engine (KG) — Memory Distillation v2
=========================================================
Enhanced with Robust JSON extraction and Error Isolation.
"""

import json
import copy
import logging
import queue
import re
import threading
import time
import os
from datetime import datetime
from typing import List, Dict, Any
import urllib.request

logger = logging.getLogger(__name__)

LOCAL_GRAPH_PATH = os.path.expanduser("~/.zhuangzhou/memory/local_graph.jsonl")
AUTO_EXPORT_THRESHOLD = int(
    os.environ.get("ZHUANGZHOU_KG_AUTO_EXPORT_THRESHOLD", "100")
)

KG_PROMPT = """
You are a Knowledge Extraction Engine for Enzyme Engineering.
Extract entities and relationships from the following turn of conversation.

FOCUS ON:
- Proteins, Mutations, Ligands, Forcefields, Simulation params.
- Success/Failure events and specific error reasons.
- Claims and hypotheses with explicit provenance; do not infer causal validation.
These triples remain unverified model extractions, not established scientific facts.

OUTPUT FORMAT:
Return ONLY a JSON list of triples: [{"head": "...", "relation": "...", "tail": "..."}]
"""


class LocalKGEngine:
    """Asynchronous Knowledge Graph extractor using local Ollama."""

    _storage_lock = threading.RLock()

    def __init__(self, model="deepseek-r1:8b"):
        self.model = model
        self.queue = queue.Queue(maxsize=32)
        self.running = True
        self.worker = None
        self._worker_lock = threading.Lock()

    def add_to_queue(self, turn_data: dict):
        try:
            self.queue.put_nowait(copy.deepcopy(turn_data))
        except queue.Full:
            logger.warning('KG distillation queue is full; original turn remains in session archive')
            return False
        with self._worker_lock:
            if self.worker is None:
                self.worker = threading.Thread(target=self._process_loop, daemon=True)
                self.worker.start()
        return True

    @staticmethod
    def get_db_path() -> str:
        return LOCAL_GRAPH_PATH

    @staticmethod
    def count_records() -> int:
        db_path = LOCAL_GRAPH_PATH
        if not os.path.exists(db_path):
            return 0
        try:
            with open(db_path, encoding="utf-8", errors="replace") as f:
                return sum(1 for line in f if line.strip())
        except OSError:
            return 0

    @staticmethod
    def _iter_records():
        db_path = LOCAL_GRAPH_PATH
        if not os.path.exists(db_path):
            return
        try:
            with open(db_path, encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(record, dict):
                        yield record
        except OSError:
            return

    @staticmethod
    def _write_records(records: List[dict]):
        db_path = LOCAL_GRAPH_PATH
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        tmp_path = db_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        os.replace(tmp_path, db_path)

    @classmethod
    def iter_triples(cls):
        for record in cls._iter_records() or []:
            timestamp = str(record.get("timestamp", ""))
            triples = record.get("triples", [])
            if not isinstance(triples, list):
                continue
            for triple in triples:
                if not isinstance(triple, dict):
                    continue
                head = str(triple.get("head", "")).strip()
                relation = str(triple.get("relation", "")).strip()
                tail = str(triple.get("tail", "")).strip()
                if head and relation and tail:
                    yield {
                        "head": head,
                        "relation": relation,
                        "tail": tail,
                        "timestamp": timestamp,
                        "access_count": int(triple.get("access_count", 0) or 0),
                        "last_accessed": str(triple.get("last_accessed", "")),
                        'verification_status': record.get('verification_status', 'legacy_unverified'),
                        'source_type': record.get('source_type', 'legacy_unknown'),
                        'source_model': record.get('source_model', ''),
                    }

    @classmethod
    def count_triples(cls) -> int:
        return sum(1 for _ in cls.iter_triples())

    @classmethod
    def stats(cls) -> dict:
        entities = set()
        relations = set()
        triple_count = 0
        for triple in cls.iter_triples():
            triple_count += 1
            entities.add(triple["head"])
            entities.add(triple["tail"])
            relations.add(triple["relation"])
        return {
            "path": LOCAL_GRAPH_PATH,
            "records": cls.count_records(),
            "triples": triple_count,
            "entities": len(entities),
            "relations": len(relations),
        }

    @classmethod
    def recent_triples(cls, limit: int = 10) -> List[dict]:
        triples = list(cls.iter_triples())
        triples.sort(
            key=lambda item: (
                item.get("timestamp", ""),
                item["head"],
                item["relation"],
                item["tail"],
            ),
            reverse=True,
        )
        return triples[:limit]

    @classmethod
    def query_triples(cls, query: str, limit: int = 10) -> List[dict]:
        query = str(query or "").strip().lower()
        if not query:
            return cls.recent_triples(limit=limit)
        matched = []
        for triple in cls.iter_triples():
            haystacks = [
                triple["head"].lower(),
                triple["relation"].lower(),
                triple["tail"].lower(),
            ]
            score = sum(1 for hay in haystacks if query in hay)
            if score:
                item = dict(triple)
                item["score"] = score
                matched.append(item)
        matched.sort(
            key=lambda item: (
                item.get("score", 0),
                item.get("timestamp", ""),
                item["head"],
            ),
            reverse=True,
        )
        return matched[:limit]

    @classmethod
    def query(cls, entity: str, limit: int = 5) -> List[dict]:
        with cls._storage_lock:
            query = str(entity or "").strip().lower()
            if not query:
                return cls.recent_triples(limit=limit)
            records = list(cls._iter_records() or [])
            matched = []
            touched = False
            now = datetime.now().isoformat()
            for record in records:
                triples = record.get("triples", [])
                if not isinstance(triples, list):
                    continue
                for triple in triples:
                    if not isinstance(triple, dict):
                        continue
                    head = str(triple.get("head", "")).strip()
                    relation = str(triple.get("relation", "")).strip()
                    tail = str(triple.get("tail", "")).strip()
                    haystacks = [head.lower(), relation.lower(), tail.lower()]
                    score = sum(1 for hay in haystacks if query in hay)
                    if score:
                        triple["access_count"] = int(triple.get("access_count", 0) or 0) + 1
                        triple["last_accessed"] = now
                        touched = True
                        matched.append(
                            {
                                "head": head,
                                "relation": relation,
                                "tail": tail,
                                "timestamp": str(record.get("timestamp", "")),
                                "access_count": triple["access_count"],
                                "last_accessed": triple["last_accessed"],
                                "score": score,
                                'verification_status': record.get('verification_status', 'legacy_unverified'),
                                'source_type': record.get('source_type', 'legacy_unknown'),
                                'source_model': record.get('source_model', ''),
                            }
                        )
            if touched:
                cls._write_records(records)
            matched.sort(
                key=lambda item: (
                    item.get("score", 0),
                    item.get("access_count", 0),
                    item.get("timestamp", ""),
                ),
                reverse=True,
            )
            return matched[:limit]

    @classmethod
    def top_patterns(cls, limit: int = 5) -> List[dict]:
        triples = list(cls.iter_triples())
        triples.sort(
            key=lambda item: (item.get("access_count", 0), item.get("timestamp", "")),
            reverse=True,
        )
        seen = set()
        top = []
        for item in triples:
            key = (item["head"].lower(), item["relation"].lower(), item["tail"].lower())
            if key in seen:
                continue
            seen.add(key)
            top.append(item)
            if len(top) >= limit:
                break
        return top

    @classmethod
    def export_overlay_for_graphrag(cls, export_dir: str) -> dict:
        os.makedirs(export_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_base = os.path.join(export_dir, f"overlay_export_{timestamp}")
        jsonl_path = export_base + ".jsonl"
        manifest_path = export_base + ".manifest.json"

        triples = list(cls.iter_triples())
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for triple in triples:
                f.write(json.dumps(triple, ensure_ascii=False) + "\n")

        stats = cls.stats()
        manifest = {
            "created_at": datetime.now().isoformat(),
            "source": LOCAL_GRAPH_PATH,
            "export_jsonl": jsonl_path,
            "record_count": stats["records"],
            "triple_count": stats["triples"],
            "entity_count": stats["entities"],
            "relation_count": stats["relations"],
            "next_step": "Use this normalized JSONL as the monthly input for parquet rebuild.",
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        return {
            "status": "ok",
            "export_jsonl": jsonl_path,
            "manifest": manifest_path,
            **stats,
        }

    @classmethod
    def _default_export_dir(cls) -> str:
        return os.path.join(os.getcwd(), "storage", "graphrag", "_overlay_exports")

    @classmethod
    def _maybe_auto_export(cls):
        if AUTO_EXPORT_THRESHOLD <= 0:
            return None
        stats = cls.stats()
        if stats["records"] < AUTO_EXPORT_THRESHOLD:
            return None
        export_dir = cls._default_export_dir()
        os.makedirs(export_dir, exist_ok=True)
        state_path = os.path.join(export_dir, "auto_export_state.json")
        last_export_records = 0
        if os.path.exists(state_path):
            try:
                with open(state_path, encoding="utf-8", errors="replace") as f:
                    last_export_records = int(
                        json.load(f).get("last_export_records", 0) or 0
                    )
            except Exception:
                last_export_records = 0
        if stats["records"] < last_export_records + AUTO_EXPORT_THRESHOLD:
            return None
        result = cls.export_overlay_for_graphrag(export_dir)
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "last_export_records": stats["records"],
                    "last_export_at": datetime.now().isoformat(),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        return result

    def _process_loop(self):
        while self.running:
            try:
                turn = self.queue.get(timeout=5)
                try:
                    self._distill(turn)
                finally:
                    self.queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error("KG Background Worker error: %s", e)

    def _distill(self, turn: dict):
        """Invoke local Ollama with extreme JSON robustness."""
        text = f"User: {turn.get('input')}\nReasoning: {str(turn.get('reasoning_chain'))[:2000]}\nOutput: {str(turn.get('final_response'))[:2000]}"
        prompt = f"{KG_PROMPT}\n\nCONVERSATION:\n{text}"

        payload = {"model": self.model, "prompt": prompt, "stream": False}

        try:
            req = urllib.request.Request(
                "http://localhost:11434/api/generate",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=180) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                raw_res = result.get("response", "")

                # 1. Strip Thought and Markdown
                clean = re.sub(r"<think>.*?</think>", "", raw_res, flags=re.DOTALL)
                clean = re.sub(r"```json\s*|```", "", clean)

                # 2. Slice to JSON array
                start, end = clean.find("["), clean.rfind("]")
                if start != -1 and end != -1:
                    json_str = clean[start : end + 1]
                    try:
                        data = json.loads(json_str)
                        if isinstance(data, list):
                            # 3. Triple Validation
                            valid = []
                            for t in data:
                                if (
                                    isinstance(t, dict)
                                    and "head" in t
                                    and "relation" in t
                                    and "tail" in t
                                ):
                                    valid.append(
                                        {
                                            k: str(t[k])
                                            for k in ("head", "relation", "tail")
                                        }
                                    )

                            if valid:
                                self._save_triples(valid, turn.get("timestamp"))
                    except:
                        pass
        except Exception as e:
            logger.debug("Ollama KG link failed: %s", e)

    def _save_triples(self, triples: List[dict], timestamp: str):
        db_path = LOCAL_GRAPH_PATH
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        enriched = []
        for triple in triples:
            item = dict(triple)
            item.setdefault("access_count", 0)
            item.setdefault("last_accessed", "")
            enriched.append(item)
        record = {"timestamp": timestamp, "triples": enriched,
                  'source_type': 'llm_distillation', 'source_model': self.model,
                  'verification_status': 'unverified'}
        try:
            with self._storage_lock:
                with open(db_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
            self._maybe_auto_export()
        except OSError:
            logger.exception('Failed to persist extracted KG triples')
            raise


kg_engine = LocalKGEngine()
