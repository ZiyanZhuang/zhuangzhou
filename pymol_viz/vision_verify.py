"""
PyMOL Render Vision Verifier
============================
Inspired by PyMolAI's `capture_viewer_snapshot` pattern.

After any headless PyMOL render, sends the output PNG to the LLM backend
as a multimodal vision input to verify render quality and detect issues
(empty scene, black image, missing interactions, wrong ligand displayed, etc.)

Usage:
    from zhuangzhou.pymol_viz.vision_verify import verify_render

    result = verify_render("path/to/output.png")
    if not result['ok']:
        print(result['issues'])   # list of detected problems
        print(result['fix_hints'])  # suggested PyMOL command corrections
"""

from __future__ import annotations

import base64
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Verification criteria injected into the LLM prompt
_VERIFY_SYSTEM = (
    "You are a structural biology image quality reviewer. "
    "A PyMOL headless render was just produced. Analyze the image and report any issues. "
    "Be concise and technical. Respond ONLY in valid JSON."
)

_VERIFY_PROMPT = """Analyze this PyMOL molecular visualization image for quality issues.

Check for:
1. EMPTY/BLACK: Is the scene empty or all-black? (render failed)
2. LIGAND_MISSING: Is the ligand visible? (should be colored green sticks)
3. INTERACTIONS_MISSING: Are interaction dashes visible? (H-bonds = yellow dashes)
4. PROTEIN_MISSING: Is the protein cartoon/ribbon visible?
5. ORIENTATION: Is the molecule facing the camera (not behind/inside the protein)?
6. CLIPPING: Are molecules cut off at image edges?
7. OVERLAPPING_LABELS: Are distance labels overlapping or unreadable?

Respond with JSON:
{
  "ok": true/false,
  "quality": "good" | "acceptable" | "poor" | "failed",
  "issues": ["list of detected issues"],
  "fix_hints": ["suggested PyMOL commands to fix each issue"],
  "summary": "one sentence description of what is shown"
}"""


def _load_png_as_base64(png_path: str) -> Optional[str]:
    """Load PNG file and return base64-encoded string."""
    try:
        with open(png_path, 'rb') as f:
            data = f.read()
        if len(data) < 1000:
            # File too small — likely empty or failed render
            return None
        return base64.standard_b64encode(data).decode('ascii')
    except Exception as e:
        logger.debug("Failed to load PNG %s: %s", png_path, e)
        return None


def _is_black_image(png_path: str) -> bool:
    """Quick check: is the image mostly black (< 5% non-black pixels)?"""
    try:
        # Try PIL first (fast)
        try:
            from PIL import Image
            import numpy as np
            img = Image.open(png_path).convert('RGB')
            arr = np.array(img)
            # Count pixels where all channels < 20 (near-black)
            black_mask = (arr < 20).all(axis=2)
            return black_mask.mean() > 0.95
        except ImportError:
            pass
        # Fallback: file size heuristic (< 5KB for 2400x1800 is suspicious)
        size = os.path.getsize(png_path)
        return size < 5_000
    except Exception:
        return False


def verify_render(
    png_path: str,
    backend=None,
    context: str = "",
    timeout: int = 30,
) -> dict:
    """
    Verify a PyMOL render PNG using LLM vision.

    Args:
        png_path:  Path to the rendered PNG file.
        backend:   LLM backend instance (must support multimodal chat).
                   If None, attempts to use a lightweight API call.
        context:   Optional context (ligand name, structure name) for better diagnosis.
        timeout:   Max seconds to wait for LLM response.

    Returns:
        {
            'ok': bool,
            'quality': str,       # 'good' | 'acceptable' | 'poor' | 'failed'
            'issues': list,       # detected problems
            'fix_hints': list,    # suggested corrections
            'summary': str,       # one-line description
            'skipped': bool,      # True if verification was skipped (no backend)
        }
    """
    if not png_path or not os.path.exists(png_path):
        return {
            'ok': False,
            'quality': 'failed',
            'issues': [f'PNG file not found: {png_path}'],
            'fix_hints': ['Check that PyMOL rendered successfully'],
            'summary': 'File not found',
            'skipped': False,
        }

    # Quick pre-check: black/empty image
    if _is_black_image(png_path):
        return {
            'ok': False,
            'quality': 'failed',
            'issues': ['Image is empty or all-black — render likely failed'],
            'fix_hints': [
                'Check PyMOL Python path in pymol_viz/config.py',
                'Try: cmd.bg_color("white"); cmd.ray(2400,1800)',
                'Verify PDB file loads correctly with: cmd.load("file.pdb")',
            ],
            'summary': 'Black/empty image — render failed',
            'skipped': False,
        }

    b64 = _load_png_as_base64(png_path)
    if not b64:
        return {
            'ok': False,
            'quality': 'failed',
            'issues': ['PNG file is empty or unreadable'],
            'fix_hints': ['Re-run the render'],
            'summary': 'Unreadable PNG',
            'skipped': False,
        }

    if backend is None:
        # Try to load a backend automatically
        try:
            from zhuangzhou.nlp.backends import get_best_available_backend
            backend = get_best_available_backend()
        except Exception:
            pass

    if backend is None:
        logger.debug("No LLM backend available — skipping vision verification")
        return {
            'ok': True,
            'quality': 'unknown',
            'issues': [],
            'fix_hints': [],
            'summary': 'Verification skipped (no LLM backend)',
            'skipped': True,
        }

    # Build multimodal message — standard Anthropic vision format
    prompt_text = _VERIFY_PROMPT
    if context:
        prompt_text = f"Context: {context}\n\n" + prompt_text

    messages = [
        {
            'role': 'user',
            'content': [
                {
                    'type': 'image',
                    'source': {
                        'type': 'base64',
                        'media_type': 'image/png',
                        'data': b64,
                    },
                },
                {
                    'type': 'text',
                    'text': prompt_text,
                },
            ],
        }
    ]

    try:
        response = backend.chat(_VERIFY_SYSTEM, messages)
        raw_text = response.text if hasattr(response, 'text') else str(response)

        # Parse JSON response
        import json as _json
        import re
        # Use JSONDecoder to find the first valid JSON object
        json_match = None
        _result_data = None
        for _start in range(len(raw_text)):
            if raw_text[_start] == '{':
                try:
                    _result_data, _ = _json.JSONDecoder().raw_decode(raw_text, _start)
                    break
                except _json.JSONDecodeError:
                    continue
        if _result_data is not None:
            result = _result_data
            result.setdefault('skipped', False)
            return result
        else:
            # LLM returned non-JSON — treat as OK but warn
            return {
                'ok': True,
                'quality': 'unknown',
                'issues': [],
                'fix_hints': [],
                'summary': raw_text[:200],
                'skipped': False,
            }

    except Exception as e:
        logger.debug("Vision verification failed: %s", e)
        return {
            'ok': True,
            'quality': 'unknown',
            'issues': [],
            'fix_hints': [],
            'summary': f'Verification error: {e}',
            'skipped': True,
        }


def verify_and_patch(
    png_path: str,
    pymol_script_path: str,
    backend=None,
    max_retries: int = 1,
) -> dict:
    """
    Verify a render and optionally apply fix commands if issues found.

    For each fix hint that looks like a PyMOL command, appends it to the
    existing render script and re-runs it.

    Returns the final verify result after up to `max_retries` fix attempts.
    """
    result = verify_render(png_path, backend=backend)

    if result.get('ok') or result.get('skipped') or not result.get('fix_hints'):
        return result

    if max_retries <= 0 or not pymol_script_path or not os.path.exists(pymol_script_path):
        return result

    # Attempt one auto-fix pass
    hints = result.get('fix_hints', [])
    pymol_commands = [h for h in hints if h.strip().startswith('cmd.')]

    if not pymol_commands:
        return result

    try:
        with open(pymol_script_path, 'r', encoding='utf-8') as f:
            original = f.read()

        # Inject fix commands before the final png() call
        fix_block = '\n'.join(pymol_commands)
        patched = original.replace(
            '# === END RENDER ===',
            f'# === AUTO-FIX ===\n{fix_block}\n# === END RENDER ===',
        )

        if patched == original:
            # Marker not found — append before last cmd.png
            lines = original.rsplit('cmd.png', 1)
            patched = (lines[0] + fix_block + '\ncmd.png' + lines[1]) if len(lines) == 2 else original

        _base, _ext = os.path.splitext(pymol_script_path)
        patched_path = _base + '_fixed' + (_ext or '.py')
        with open(patched_path, 'w', encoding='utf-8') as f:
            f.write(patched)

        from .config import PYMOL_PYTHON
        import subprocess
        proc = subprocess.run(
            [PYMOL_PYTHON, patched_path],
            capture_output=True, text=True, timeout=90,
        )

        if os.path.exists(png_path):
            result = verify_render(png_path, backend=backend)
            result['auto_fixed'] = True

    except Exception as e:
        logger.debug("Auto-fix attempt failed: %s", e)

    return result
