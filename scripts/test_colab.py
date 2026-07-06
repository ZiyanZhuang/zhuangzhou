"""
Test Colab GPU Server connection (v0.9.0 — HTTP tunnel mode)
=============================================================
Usage:
    python scripts/test_colab.py                                         # Run all tests
    python scripts/test_colab.py --url https://xxx.trycloudflare.com     # Test specific URL
    python scripts/test_colab.py --set-url https://xxx.trycloudflare.com # Save URL + test
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8')


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Test Colab GPU server connection')
    parser.add_argument('--url', help='Tunnel URL to test')
    parser.add_argument('--set-url', help='Set tunnel URL in config and test')
    args = parser.parse_args()

    from envs.colab_sandbox import ColabSandbox
    from envs.colab_config import set_tunnel_url

    # Set URL if requested
    if args.set_url:
        set_tunnel_url(args.set_url)
        print(f'Saved tunnel URL: {args.set_url}')

    sandbox = ColabSandbox()
    if args.url:
        sandbox.tunnel_url = args.url

    if not sandbox.available:
        print('ERROR: No tunnel URL configured.')
        print('Set ZZ_COLAB_TUNNEL env var or use --url / --set-url')
        print('\nSetup steps:')
        print('  1. Open Google Colab, select T4 GPU')
        print('  2. Run scripts/colab_gpu_server.ipynb (or .py)')
        print('  3. Copy the printed tunnel URL')
        print('  4. Run: python scripts/test_colab.py --set-url <URL>')
        sys.exit(1)

    print(f'Tunnel URL: {sandbox.tunnel_url}\n')

    # Test 1: Health check
    print('=== Test 1: Health Check ===')
    r = sandbox.connect(timeout=10)
    if r.success:
        data = r.data if isinstance(r.data, dict) else {}
        print(f'  OK: gpu={data.get("gpu_name", "?")}, '
              f'uptime={data.get("uptime_seconds", 0):.0f}s')
    else:
        print(f'  FAILED: {r.error}')
        sys.exit(1)

    # Test 2: GPU info
    print('\n=== Test 2: GPU Info ===')
    r = sandbox.gpu_info(timeout=10)
    if r.success:
        print(json.dumps(r.data, indent=2) if isinstance(r.data, dict) else r.stdout)
    else:
        print(f'  FAILED: {r.error}')

    # Test 3: Execute code
    print('\n=== Test 3: Execute Code ===')
    r = sandbox.execute_code('print("Hello from Zhuangzhou on Colab!")')
    print(f'  success={r.success}  stdout=[{r.stdout}]')

    # Test 4: JSON bridge (backward compat)
    print('\n=== Test 4: JSON Bridge (backward compat) ===')
    r = sandbox.run_json_bridge(
        'import json\nprint(json.dumps({"len": len(_payload["seq"]), "ok": True}))',
        {'seq': 'MKTLLILAVVAAALA'}
    )
    print(f'  success={r.success}  data={r.data}')

    # Test 5: Packages
    print('\n=== Test 5: Package Check ===')
    pkgs = sandbox.check_packages(['torch', 'fair-esm', 'transformers'])
    print(f'  packages: {pkgs}')

    # Test 6: Error handling
    print('\n=== Test 6: Error Handling ===')
    r = sandbox.execute_code('1/0')
    print(f'  success={r.success}  error={r.error[:80] if r.error else "none"}')

    # Test 7: ESM scan (short)
    print('\n=== Test 7: ESM Scan (short) ===')
    r = sandbox.esm_scan(sequence='MKTLLILAVFLCLAQG', top_k=3, timeout=120)
    if r.success:
        data = r.data if isinstance(r.data, dict) else {}
        muts = data.get('mutations', [])
        print(f'  Found {len(muts)} mutations ({data.get("elapsed_seconds", "?")}s)')
        for m in muts[:3]:
            print(f'    {m.get("mutation", "?")}: score={m.get("score", 0):.4f}')
    else:
        print(f'  FAILED: {r.error}')
        print('  (OK if fair-esm not installed on Colab yet)')

    print('\n=== All tests complete! Server still running on Colab. ===')


if __name__ == '__main__':
    main()
