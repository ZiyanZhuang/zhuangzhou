#!/usr/bin/env python3
"""Export local_graph.jsonl into a normalized JSONL bundle for later GraphRAG parquet rebuilds."""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_PARENT = os.path.dirname(ROOT)
if PROJECT_PARENT not in sys.path:
    sys.path.insert(0, PROJECT_PARENT)

from zhuangzhou.memory.kg_engine import LocalKGEngine


def main():
    parser = argparse.ArgumentParser(
        description="Export Zhuangzhou local KG overlay for monthly GraphRAG rebuilds"
    )
    parser.add_argument(
        "--export-dir",
        default=os.path.join(ROOT, "storage", "graphrag", "_overlay_exports"),
        help="Directory where normalized overlay export files will be written",
    )
    args = parser.parse_args()

    result = LocalKGEngine.export_overlay_for_graphrag(args.export_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
