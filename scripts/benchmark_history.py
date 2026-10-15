"""Compare history tail reads against a fixed Git baseline using synthetic JSONL.

Run from the repository: python scripts/benchmark_history.py --entries 100000
No model, network, scientific software or user history is used.
"""

import argparse
import json
from pathlib import Path
import platform
import subprocess
import tempfile
import time
import tracemalloc
import types

from zhuangzhou.memory.execution_history import ExecutionHistory


def measure(history, limit):
    tracemalloc.start()
    start = time.perf_counter()
    result = history.get_recent(limit=limit)
    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, {'seconds': round(elapsed, 6), 'peak_python_bytes': peak}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--entries', type=int, default=100000)
    parser.add_argument('--limit', type=int, default=50)
    parser.add_argument('--baseline-ref', default='b55b88c')
    args = parser.parse_args()
    if args.entries < 1 or args.limit < 1:
        parser.error('entries and limit must be positive')
    root = Path(__file__).resolve().parents[1]
    source = subprocess.check_output(['git', 'show', f'{args.baseline_ref}:memory/execution_history.py'], cwd=root)
    baseline = types.ModuleType('history_baseline')
    exec(compile(source, '<git-baseline-history>', 'exec'), baseline.__dict__)
    with tempfile.TemporaryDirectory(prefix='zz-history-benchmark-') as directory:
        log = Path(directory) / 'execution_log.jsonl'
        with log.open('w', encoding='utf-8') as handle:
            for index in range(args.entries):
                handle.write(json.dumps({'skill': 'synthetic_fixture', 'success': True, 'index': index,
                                         'params': {'input': 'synthetic input' * 8}}) + '\n')
        old_result, before = measure(baseline.ExecutionHistory(directory), args.limit)
        new_result, after = measure(ExecutionHistory(directory), args.limit)
        assert old_result == new_result
        assert len(new_result) == min(args.entries, args.limit)
        print(json.dumps({'python': platform.python_version(), 'platform': platform.system(),
                          'entries': args.entries, 'limit': args.limit, 'fixture_bytes': log.stat().st_size,
                          'baseline_ref': args.baseline_ref, 'before': before, 'after': after,
                          'same_results': True, 'note': 'One synthetic local run; tracemalloc excludes native allocations.'}))


if __name__ == '__main__':
    main()
