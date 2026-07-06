"""
ChatMol Bridge — PLIP-to-ChatMol agentic visualization.

Architecture:
  1. PLIP results (structured JSON) → rich prompt
  2. ChatMol v2 agent loop (DeepSeek LLM) → PyMOL commands
  3. PyMOL headless execution → publication-quality PNG

This bridge runs OUTSIDE PyMOL and spawns a PyMOL headless subprocess
that executes the ChatMol v2 agent internally.
"""

import json
import logging
import os
import subprocess
import textwrap
import tempfile

from .config import PYMOL_PYTHON

logger = logging.getLogger(__name__)

# Path to bundled ChatMol v2 plugin
_HERE = os.path.dirname(os.path.abspath(__file__))
CHATMOL_V2_PATH = os.path.join(_HERE, "chatmol", "chatmol_plugin_v2.py")


def _redact_known_secrets(text: str) -> str:
    """Redact API keys from subprocess logs before writing to the parent logger."""
    if not text:
        return text
    redacted = text
    for env_var in (
        "DEEPSEEK_API_KEY",
        "OPENROUTER_API_KEY",
        "MOONSHOT_API_KEY",
        "QWEN_API_KEY",
        "OPENAI_API_KEY",
    ):
        value = os.environ.get(env_var, "").strip()
        if value:
            redacted = redacted.replace(value, f"<redacted:{env_var}>")
    return redacted


def build_plip_prompt(
    complex_pdb: str,
    plip_result: dict,
    ligand_resname: str = None,
    affinity: float = None,
    rank: int = None,
    output_png: str = None,
) -> str:
    """
    Build a detailed prompt for ChatMol v2 from PLIP interaction data.
    The prompt tells the LLM exactly what to visualize.
    """
    pdb_name = os.path.basename(complex_pdb)

    # Extract ligand name
    if not ligand_resname:
        detected = plip_result.get("ligands_detected", [])
        if detected:
            ligand_resname = detected[0]
        else:
            # Fallback: parse from interaction site keys (e.g. "MOL:B:1")
            for key in plip_result.get("interactions", {}):
                parts = str(key).split(":")
                if parts:
                    ligand_resname = parts[0]
                    break
            else:
                ligand_resname = "UNK"

    # Extract interactions into readable format
    interaction_text = []
    interactions = plip_result.get("interactions", {})
    for site_key, site_data in interactions.items():
        if not isinstance(site_data, dict):
            continue
        for itype, ilist in site_data.items():
            if not isinstance(ilist, list) or not ilist:
                continue
            for ix in ilist:
                if not isinstance(ix, dict):
                    continue
                res = ix.get("residue", "?")
                chain = ix.get("reschain", "A")
                dist_info = ""
                if "dist_h_a" in ix:
                    dist_info = f", H-A dist={ix['dist_h_a']}A, D-A dist={ix.get('dist_d_a', '?')}A"
                elif "dist" in ix:
                    dist_info = f", dist={ix['dist']}A"
                sc = (
                    " (sidechain)"
                    if ix.get("sidechain") in ("True", True)
                    else " (backbone)"
                )
                interaction_text.append(
                    f"  - {itype}: {res} chain {chain}{sc}{dist_info}"
                )

    ix_block = (
        "\n".join(interaction_text)
        if interaction_text
        else "  (no interactions detected)"
    )

    affinity_line = ""
    if affinity is not None:
        affinity_line = (
            f"\nDocking affinity: {affinity} kcal/mol (rank {rank})."
            if rank
            else f"\nDocking affinity: {affinity} kcal/mol."
        )

    output_line = ""
    if output_png:
        png_fwd = output_png.replace("\\", "/")
        output_line = f"\n\nRender the FINAL image to: {png_fwd}\nUse purpose=final, width=3600, height=2700, dpi=600, ray=true, transparent_bg=false."

    prompt = f"""The PDB file "{pdb_name}" is already loaded as object "complex".
Ligand residue name: {ligand_resname}{affinity_line}

PLIP interaction analysis results:
{ix_block}

Please create a publication-quality figure showing ALL these protein-ligand interactions. Requirements:

STYLE (Nature Structural Biology / Science level):
- White background, clean composition, no clutter
- Protein: light gray cartoon, semi-transparent (transparency 0.4)
- Ligand ({ligand_resname}): sticks, green carbons, prominent
- Interacting residues: sticks, slate/steel-blue carbons, show sidechains
- Use util.cnc for standard element coloring on N/O/S atoms

INTERACTIONS — create a SEPARATE distance object for EACH residue interaction:
- H-bonds: use "distance hb_RES, (resn LIGNAME) and (elem N+O), (resi X) and (elem N+O+S), mode=2, cutoff=4.0"
  Then: color yellow, hb_RES; set dash_gap, 0.15, hb_RES; set dash_radius, 0.07, hb_RES
  Do NOT hide labels — let PyMOL show the measured distance automatically.
- Hydrophobic: use "distance hp_RES, (resn LIGNAME) and (elem C), (resi X) and (elem C), mode=2, cutoff=4.5"
  Then: color gray60, hp_RES; set dash_gap, 0.3, hp_RES; set dash_radius, 0.04, hp_RES
- Salt bridges: orange dashed lines if present
- Pi-stacking: green dashed lines if present
- Set "set label_distance_digits, 1" globally so distances show 1 decimal place
- Set "set label_color, yellow, hb_*" and "set label_color, gray60, hp_*"
- Set "set label_size, -0.4" globally for distance labels

RESIDUE LABELS — for EACH interacting residue, add a label:
- Use exactly: label (chain A and resi X and name CA), "D10"
- Then: set label_position, [dx, dy, 2.5], (chain A and resi X and name CA)
  where dx,dy vary per residue to avoid overlap (e.g., [0,0,2.5], [2,1.5,2], [-2,1.5,2])
- set label_color, black
- set label_outline_color, white
- set label_font_id, 7
- set label_size, -0.6
- CRITICAL: Do NOT use set_property or label_offset — only use "set label_position"

CAMERA:
- Zoom to ligand + binding site with buffer=6
- Orient for best view, slight Y rotation (15 degrees)

RENDERING:
- ray_trace_mode 1 (fast, with black outlines)
- Soft lighting: ambient=0.40, direct=0.45, spec_reflect=0.08
- cartoon_fancy_helices=1, cartoon_smooth_loops=1
- antialias=2{output_line}"""

    return prompt


def run_chatmol_viz(
    complex_pdb: str,
    plip_result: dict,
    output_dir: str = None,
    ligand_resname: str = None,
    affinity: float = None,
    rank: int = None,
    provider: str = "deepseek",
    model: str = "deepseek-chat",
    timeout: int = 180,
) -> dict:
    """
    Run ChatMol v2 agent to generate publication-quality visualization.

    This spawns PyMOL headless, loads the ChatMol v2 plugin, and runs
    the agentic loop with a PLIP-enriched prompt.

    Args:
        complex_pdb: path to complex PDB
        plip_result: dict from run_plip_analysis()
        output_dir: where to save images
        ligand_resname: ligand 3-letter code
        affinity: docking affinity kcal/mol
        rank: docking rank number
        provider: LLM provider (deepseek, openrouter, etc.)
        model: LLM model name
        timeout: max seconds for the whole process

    Returns:
        dict with png_path, chatmol_log, etc.
    """
    complex_pdb = os.path.abspath(complex_pdb)
    if not os.path.exists(complex_pdb):
        return {"error": f"PDB not found: {complex_pdb}"}

    stem = os.path.splitext(os.path.basename(complex_pdb))[0]
    if not output_dir:
        output_dir = os.path.join(os.getcwd(), f"{stem}_chatmol_viz")
    os.makedirs(output_dir, exist_ok=True)

    output_png = os.path.join(output_dir, f"{stem}_publication.png")

    # Build prompt
    prompt = build_plip_prompt(
        complex_pdb=complex_pdb,
        plip_result=plip_result,
        ligand_resname=ligand_resname,
        affinity=affinity,
        rank=rank,
        output_png=output_png,
    )

    # Save prompt for debugging
    prompt_path = os.path.join(output_dir, "chatmol_prompt.txt")
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(prompt)

    # Generate the PyMOL runner script
    pdb_fwd = complex_pdb.replace("\\", "/")
    v2_fwd = CHATMOL_V2_PATH.replace("\\", "/")
    prompt_fwd = prompt_path.replace("\\", "/")
    log_path = os.path.join(output_dir, "chatmol_log.txt").replace("\\", "/")

    runner_script = textwrap.dedent(f'''\
        import sys, os, json, time
        sys.stdout.reconfigure(encoding="utf-8")

        # Start PyMOL headless
        import pymol
        pymol.pymol_argv = ["pymol", "-cq"]
        pymol.finish_launching()
        from pymol import cmd

        # Load the complex
        cmd.load(r"{pdb_fwd}", "complex")
        print(f"Loaded: {{cmd.count_atoms('complex')}} atoms")

        # Load ChatMol v2 plugin
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location("chatmol_v2", r"{v2_fwd}")
        _mod = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        agent = _mod.ChatMolAgent()
        _api_keys = {{}}
        for _env_var, _prov_name in [
            ("DEEPSEEK_API_KEY", "deepseek"),
            ("OPENROUTER_API_KEY", "openrouter"),
            ("MOONSHOT_API_KEY", "kimi"),
            ("QWEN_API_KEY", "qwen"),
            ("OPENAI_API_KEY", "openai"),
        ]:
            _v = os.environ.get(_env_var, "").strip()
            if _v:
                _api_keys[_prov_name] = _v
        agent.config["provider"] = "{provider}"
        if "{provider}" in _api_keys:
            agent.config["_session_api_key"] = _api_keys["{provider}"]
        agent.config["text_model"] = "{model}"
        agent._reinit_client()

        # Read prompt
        with open(r"{prompt_fwd}", encoding="utf-8") as f:
            prompt = f.read()

        # Run agent loop
        t0 = time.time()
        log_lines = []
        try:
            response = agent._run_agent_loop(prompt)
            log_lines.append(f"Agent response: {{response}}")
            log_lines.append(f"Time: {{time.time()-t0:.1f}}s")
        except Exception as e:
            log_lines.append(f"Agent error: {{e}}")

        # Save log
        with open(r"{log_path}", "w", encoding="utf-8") as f:
            f.write("\\n".join(log_lines))

        print("\\n".join(log_lines))
        cmd.quit()
    ''')

    runner_path = os.path.join(output_dir, "_chatmol_runner.py")
    with open(runner_path, "w", encoding="utf-8") as f:
        f.write(runner_script)

    # Execute via PyMOL's Python
    logger.info("Running ChatMol agent via: %s", PYMOL_PYTHON)
    try:
        proc = subprocess.run(
            [PYMOL_PYTHON, runner_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=output_dir,
        )
        stdout = _redact_known_secrets(proc.stdout)
        stderr = _redact_known_secrets(proc.stderr)
        logger.info("ChatMol stdout:\n%s", stdout[-1000:] if stdout else "")
        if stderr:
            logger.warning("ChatMol stderr:\n%s", stderr[-500:])
    except subprocess.TimeoutExpired:
        return {
            "error": f"ChatMol timed out after {timeout}s",
            "hint": "Try increasing timeout or simplifying the prompt",
            "prompt_path": prompt_path,
        }
    except FileNotFoundError:
        return {"error": f"PyMOL python not found: {PYMOL_PYTHON}"}

    # Check output
    result = {
        "output_dir": output_dir,
        "prompt_path": prompt_path,
        "log_path": log_path,
        "provider": provider,
        "model": model,
    }

    if os.path.exists(output_png):
        result["png_path"] = output_png
        result["png_size"] = os.path.getsize(output_png)
        result["success"] = True
    else:
        result["success"] = False
        result["warning"] = "ChatMol agent ran but PNG not found"
        result["stdout"] = stdout[-500:] if stdout else ""

    # Cleanup runner
    try:
        os.unlink(runner_path)
    except OSError:
        pass

    return result
