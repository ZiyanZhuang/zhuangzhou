#!/usr/bin/env python
"""
GraphRAG Bridge — runs inside conda 'graphrag' environment.

Reads JSON command from stdin, executes against parquet files, writes JSON to stdout.

Commands:
  search_entities:     keyword search in entities.parquet
  search_relationships: find relationships for an entity
  community_reports:   search community report summaries
  kg_stats:           get statistics for a KG
  list_kgs:           list available knowledge graphs
"""

import json
import os
import sys

import pandas as pd


LOCAL_OVERLAY_KEY = "local_overlay"


def _graphrag_root():
    return os.environ.get(
        "GRAPHRAG_ROOT",
        os.path.normpath(
            os.path.join(
                os.path.dirname(__file__), "..", "..", "..", "storage", "graphrag"
            )
        ),
    )


def _load_parquet(kg_id, filename):
    path = os.path.join(_graphrag_root(), kg_id, filename)
    if os.path.exists(path):
        try:
            return pd.read_parquet(path)
        except Exception:
            return None
    return None


def _local_graph_path():
    return os.environ.get(
        "ZHUANGZHOU_LOCAL_GRAPH",
        os.path.expanduser("~/.zhuangzhou/memory/local_graph.jsonl"),
    )


def _iter_local_triples():
    path = _local_graph_path()
    if not os.path.exists(path):
        return
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
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
                        }
    except OSError:
        return


def _dedupe_records(records, key_fields):
    seen = set()
    deduped = []
    for record in records:
        key = tuple(str(record.get(field, "")).strip().lower() for field in key_fields)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)
    return deduped


def _search_local_overlay_entities(keywords, limit=20):
    normalized = [kw.lower().strip() for kw in keywords if str(kw).strip()]
    if not normalized:
        return []
    matched = []
    for triple in _iter_local_triples() or []:
        haystacks = [
            triple["head"].lower(),
            triple["tail"].lower(),
            triple["relation"].lower(),
        ]
        if any(kw in hay for kw in normalized for hay in haystacks):
            matched.append(
                {
                    "id": f"overlay:{triple['head']}:{triple['relation']}:{triple['tail']}",
                    "title": triple["head"],
                    "type": "LOCAL_TRIPLE",
                    "description": f"{triple['head']} --{triple['relation']}--> {triple['tail']}",
                    "relation": triple["relation"],
                    "tail": triple["tail"],
                    "timestamp": triple["timestamp"],
                    "source_kg": LOCAL_OVERLAY_KEY,
                }
            )
    matched.sort(key=lambda x: (x.get("timestamp", ""), x["title"]), reverse=True)
    return _dedupe_records(matched, ("title", "relation", "tail"))[:limit]


def _search_local_overlay_relationships(entity_name, limit=20):
    entity_lower = entity_name.lower().strip()
    if not entity_lower:
        return []
    matched = []
    for triple in _iter_local_triples() or []:
        src = triple["head"].lower()
        tgt = triple["tail"].lower()
        rel = triple["relation"].lower()
        if entity_lower in src or entity_lower in tgt or entity_lower in rel:
            matched.append(
                {
                    "source": triple["head"],
                    "target": triple["tail"],
                    "description": f"{triple['head']} --{triple['relation']}--> {triple['tail']}",
                    "relation": triple["relation"],
                    "weight": 1.0,
                    "timestamp": triple["timestamp"],
                    "source_kg": LOCAL_OVERLAY_KEY,
                }
            )
    matched.sort(
        key=lambda x: (x.get("timestamp", ""), x["source"], x["target"]), reverse=True
    )
    return _dedupe_records(matched, ("source", "relation", "target"))[:limit]


def _local_overlay_stats():
    triple_count = 0
    entity_names = set()
    relation_names = set()
    for triple in _iter_local_triples() or []:
        triple_count += 1
        entity_names.add(triple["head"])
        entity_names.add(triple["tail"])
        relation_names.add(triple["relation"])
    return {
        "overlay_records": triple_count,
        "overlay_entities": len(entity_names),
        "overlay_relations": len(relation_names),
        "overlay_path": _local_graph_path(),
    }


def _list_kg_ids():
    root = _graphrag_root()
    if not os.path.exists(root):
        return []
    return [
        d
        for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
        and os.path.exists(os.path.join(root, d, "entities.parquet"))
    ]


def search_entities(keywords, kg_id="all", limit=20):
    kg_dirs = [kg_id] if kg_id != "all" else _list_kg_ids()
    results = {}
    overlay_matches = _search_local_overlay_entities(keywords, limit=limit)
    if overlay_matches:
        results[LOCAL_OVERLAY_KEY] = overlay_matches
    for kid in kg_dirs:
        df = _load_parquet(kid, "entities.parquet")
        if df is None:
            continue
        title_col = "title" if "title" in df.columns else "name"
        if title_col not in df.columns:
            continue
        matched = []
        for _, row in df.iterrows():
            title = str(row.get(title_col, "")).lower()
            desc = str(row.get("description", "")).lower()
            for kw in keywords:
                if kw.lower() in title or kw.lower() in desc:
                    matched.append(
                        {
                            "id": str(row.get("id", row.get("human_readable_id", ""))),
                            "title": row.get(title_col, ""),
                            "type": row.get("type", "UNKNOWN"),
                            "description": str(row.get("description", ""))[:500],
                            "source_kg": kid,
                        }
                    )
                    break
        if matched:
            results[kid] = matched[:limit]
    return results


def search_relationships(entity_name, kg_id="all", limit=20):
    kg_dirs = [kg_id] if kg_id != "all" else _list_kg_ids()
    results = {}
    entity_lower = entity_name.lower()
    overlay_matches = _search_local_overlay_relationships(entity_name, limit=limit)
    if overlay_matches:
        results[LOCAL_OVERLAY_KEY] = overlay_matches
    for kid in kg_dirs:
        df = _load_parquet(kid, "relationships.parquet")
        if df is None:
            continue
        matched = []
        for _, row in df.iterrows():
            src = str(row.get("source", "")).lower()
            tgt = str(row.get("target", "")).lower()
            if entity_lower in src or entity_lower in tgt:
                matched.append(
                    {
                        "source": row.get("source", ""),
                        "target": row.get("target", ""),
                        "description": str(row.get("description", ""))[:500],
                        "weight": float(row.get("weight", 1.0)),
                        "source_kg": kid,
                    }
                )
        if matched:
            results[kid] = matched[:limit]
    return results


def community_reports(query, kg_id="all", limit=5):
    kg_dirs = [kg_id] if kg_id != "all" else _list_kg_ids()
    results = {}
    query_words = [w.lower() for w in query.split() if len(w) > 2]
    for kid in kg_dirs:
        df = _load_parquet(kid, "community_reports.parquet")
        if df is None:
            continue
        matched = []
        for _, row in df.iterrows():
            title = str(row.get("title", "")).lower()
            summary = str(row.get("summary", "")).lower()
            full = str(row.get("full_content", "")).lower()
            score = sum(
                1 for w in query_words if w in title or w in summary or w in full
            )
            if score > 0:
                matched.append(
                    {
                        "id": str(row.get("id", row.get("community", ""))),
                        "title": row.get("title", ""),
                        "summary": str(row.get("summary", ""))[:800],
                        "relevance_score": score,
                        "source_kg": kid,
                    }
                )
        matched.sort(key=lambda x: x["relevance_score"], reverse=True)
        if matched:
            results[kid] = matched[:limit]
    return results


def kg_stats(kg_id):
    stats = {"kg_id": kg_id}
    for name in [
        "entities",
        "relationships",
        "communities",
        "community_reports",
        "text_units",
        "documents",
    ]:
        df = _load_parquet(kg_id, f"{name}.parquet")
        stats[name] = len(df) if df is not None else 0
    stats.update(_local_overlay_stats())
    return stats


def main():
    payload = json.loads(sys.stdin.read())
    cmd = payload.get("command", "")
    args = payload.get("args", {})

    try:
        if cmd == "search_entities":
            result = search_entities(**args)
        elif cmd == "search_relationships":
            result = search_relationships(**args)
        elif cmd == "community_reports":
            result = community_reports(**args)
        elif cmd == "kg_stats":
            result = kg_stats(**args)
        elif cmd == "list_kgs":
            result = _list_kg_ids()
        else:
            result = {"error": f"Unknown command: {cmd}"}

        sys.stdout.reconfigure(encoding="utf-8")
        print(
            json.dumps(
                {"status": "ok", "result": result}, ensure_ascii=False, default=str
            )
        )
    except Exception as e:
        print(json.dumps({"status": "error", "error": str(e)}))


if __name__ == "__main__":
    main()
