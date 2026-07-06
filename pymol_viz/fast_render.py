"""
Fast one-shot PLIP visualization renderer.

Generates a single Python script that PyMOL executes directly.
No .pml intermediary, no cmd.do() per-line overhead.
Target: < 30s end-to-end.

Style: Nature Chemical Biology / JACS / Angew Chem conventions.
"""

import json
import logging
import os
import re
import subprocess
import textwrap
import time
from collections import defaultdict
from pathlib import Path

from .config import PYMOL_PYTHON

logger = logging.getLogger(__name__)

AA3 = {
    "ALA": "Ala",
    "ARG": "Arg",
    "ASN": "Asn",
    "ASP": "Asp",
    "CYS": "Cys",
    "GLN": "Gln",
    "GLU": "Glu",
    "GLY": "Gly",
    "HIS": "His",
    "ILE": "Ile",
    "LEU": "Leu",
    "LYS": "Lys",
    "MET": "Met",
    "PHE": "Phe",
    "PRO": "Pro",
    "SER": "Ser",
    "THR": "Thr",
    "TRP": "Trp",
    "TYR": "Tyr",
    "VAL": "Val",
}


def fast_visualize(
    complex_pdb: str,
    plip_result: dict,
    output_dir: str = None,
    ligand_resname: str = None,
    affinity: float = None,
    rank: int = None,
    width: int = 2400,
    height: int = 1800,
    dpi: int = 300,
    timeout: int = 90,
) -> dict:
    """
    Fast publication-quality PLIP visualization.

    Generates a Python script -> runs via PyMOL headless -> PNG.
    All PyMOL commands use cmd.* API directly (no .pml parsing overhead).
    """
    complex_pdb = os.path.abspath(complex_pdb)
    if not os.path.exists(complex_pdb):
        return {"error": f"PDB not found: {complex_pdb}"}

    stem = os.path.splitext(os.path.basename(complex_pdb))[0]
    # Convention: output defaults to CWD (project workspace), not input file dir
    if not output_dir:
        output_dir = os.path.join(os.getcwd(), f"{stem}_viz")
    os.makedirs(output_dir, exist_ok=True)

    # Detect ligand
    if not ligand_resname:
        for key in plip_result.get("interactions", {}):
            ligand_resname = str(key).split(":")[0]
            break
        if not ligand_resname:
            ligand_resname = _detect_ligand(complex_pdb)

    # Extract interactions
    interactions = _extract_interactions(plip_result.get("interactions", {}))
    residues = set((ix["resn"], ix["resi"], ix["chain"]) for ix in interactions)

    output_png = str(Path(output_dir) / f"{stem}.png")
    # Ensure forward slashes and no embedded quotes that could break PyMOL syntax
    pdb_fwd = complex_pdb.replace("\\", "/").replace('"', '\\"').replace("'", "\\'")

    # Sanitize ligand name for PyMOL command embedding
    safe_ligand = (ligand_resname or '').replace('"', '\\"').replace("'", "\\'").replace(';', '').strip()

    # Build the complete PyMOL Python script
    script = _build_render_script(
        pdb_path=pdb_fwd,
        ligand=safe_ligand,
        interactions=interactions,
        residues=residues,
        output_png=output_png,
        width=width,
        height=height,
        dpi=dpi,
        affinity=affinity,
        rank=rank,
    )

    # Also save .pml for manual use
    pml_path = os.path.join(output_dir, f"{stem}.pml")
    pml_script = _build_pml(
        pdb_path=pdb_fwd,
        ligand=safe_ligand,
        interactions=interactions,
        residues=residues,
        output_png=output_png,
        width=width,
        height=height,
        dpi=dpi,
    )
    with open(pml_path, "w", encoding="utf-8") as f:
        f.write(pml_script)

    # Execute with guaranteed cleanup of the temp render script
    script_path = os.path.join(output_dir, f"{stem}_render.py")
    try:
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script)
        t0 = time.time()
        try:
            proc = subprocess.run(
                [PYMOL_PYTHON, script_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=output_dir,
            )
            elapsed = time.time() - t0
            stdout = proc.stdout.strip()
            if proc.returncode != 0:
                result = {"error": f"PyMOL exited with code {proc.returncode}", "stderr": proc.stderr[-300:], "script_path": pml_path}
            elif os.path.exists(output_png):
                result = {"png_path": output_png, "pml_path": pml_path, "elapsed": elapsed, "stdout": stdout}
            else:
                result = {"error": "PNG not generated — check PyMOL stdout", "stdout": stdout, "pml_path": pml_path}
        except subprocess.TimeoutExpired:
            return {"error": f"Render timed out ({timeout}s)", "script_path": pml_path}
        except FileNotFoundError:
            return {"error": f"PyMOL not found: {PYMOL_PYTHON}", "script_path": pml_path}
    finally:
        try:
            if os.path.exists(script_path):
                os.unlink(script_path)
        except OSError as e:
            logger.debug("Could not clean up render script %s: %s", script_path, e)

    return result


def _build_render_script(
    pdb_path,
    ligand,
    interactions,
    residues,
    output_png,
    width,
    height,
    dpi,
    affinity=None,
    rank=None,
) -> str:
    """Build a Python script using PyMOL cmd.* API — Nature/JACS style."""

    resi_list = "+".join(
        sorted(set(r[1] for r in residues), key=lambda x: int(x) if x.isdigit() else 0)
    )

    # Group interactions by type and residue
    hbonds = defaultdict(list)
    hydrophobic = defaultdict(list)
    saltbridges = defaultdict(list)
    pistacking = defaultdict(list)
    for ix in interactions:
        key = f"{ix['resn']}{ix['resi']}"
        if ix["type"] == "hydrogen_bonds":
            hbonds[key].append(ix)
        elif ix["type"] == "hydrophobic":
            hydrophobic[key].append(ix)
        elif ix["type"] == "salt_bridges":
            saltbridges[key].append(ix)
        elif ix["type"] in ("pi_stacking", "pi_cation"):
            pistacking[key].append(ix)

    # Residue labels — 3-letter convention (Asp10, Phe15, Leu151)
    label_cmds = []
    offsets = [
        (0, 4.0, 2.5),
        (4.0, 2.0, 2.5),
        (-4.0, 2.0, 2.5),
        (4.0, -2.0, 2.5),
        (-4.0, -2.0, 2.5),
        (0, -4.0, 2.5),
    ]
    sorted_res = sorted(residues, key=lambda r: int(r[1]) if r[1].isdigit() else 0)
    for i, (resn, resi, chain) in enumerate(sorted_res):
        lbl = f"{AA3.get(resn, resn)}{resi}"
        ox, oy, oz = offsets[i % len(offsets)]
        cs = f"chain {chain} and " if chain else ""
        sel = f"({cs}resi {resi}) and name CA"
        label_cmds.append(f'cmd.label("{sel}", "\\"{lbl}\\"")')
        label_cmds.append(f'cmd.set("label_position", [{ox},{oy},{oz}], "{sel}")')
        label_cmds.append(f'cmd.set("label_size", -0.8, "{sel}")')
    labels_block = "\n".join(label_cmds)

    # H-bond distance objects — BLACK dashes (Nature convention)
    hb_cmds = []
    for key, ixlist in hbonds.items():
        resn = ixlist[0]["resn"]
        resi = ixlist[0]["resi"]
        chain = ixlist[0]["chain"]
        cs = f"chain {chain} and " if chain else ""
        has_sc = any(ix.get("sidechain") for ix in ixlist)
        if has_sc:
            res_atoms = f"({cs}resi {resi}) and (name OD1+OD2+OE1+OE2+NE2+NH1+NH2+NZ+OG+OG1+OH+ND1+ND2+NE1+SD+SG+N+O)"
        else:
            res_atoms = f"({cs}resi {resi}) and (name N+O)"
        obj = f"hb_{AA3.get(resn, resn)}{resi}"
        hb_cmds.append(
            f'cmd.distance("{obj}", "(resn {ligand}) and (elem N+O+S)", "{res_atoms}", mode=2, cutoff=4.5)'
        )
        hb_cmds.append(f'cmd.color("black", "{obj}")')
        hb_cmds.append(f'cmd.set("dash_radius", 0.08, "{obj}")')
        hb_cmds.append(f'cmd.set("dash_gap", 0.15, "{obj}")')
        hb_cmds.append(f'cmd.set("dash_length", 0.15, "{obj}")')
        hb_cmds.append(f'cmd.set("dash_round_ends", 0, "{obj}")')
        hb_cmds.append(f'cmd.set("label_size", -0.6, "{obj}")')
    hb_block = "\n".join(hb_cmds)

    # Salt bridge distance objects — orange dashes
    sb_cmds = []
    for key, ixlist in saltbridges.items():
        resn = ixlist[0]["resn"]
        resi = ixlist[0]["resi"]
        chain = ixlist[0]["chain"]
        cs = f"chain {chain} and " if chain else ""
        obj = f"sb_{AA3.get(resn, resn)}{resi}"
        sb_cmds.append(
            f'cmd.distance("{obj}", "(resn {ligand}) and (elem N+O)", "({cs}resi {resi}) and (elem N+O)", mode=2, cutoff=5.0)'
        )
        sb_cmds.append(f'cmd.color("orange", "{obj}")')
        sb_cmds.append(f'cmd.set("dash_radius", 0.06, "{obj}")')
        sb_cmds.append(f'cmd.set("dash_gap", 0.15, "{obj}")')
    sb_block = "\n".join(sb_cmds)

    return textwrap.dedent(f'''\
import sys, time
sys.stdout.reconfigure(encoding="utf-8")
t0 = time.time()

import pymol
pymol.pymol_argv = ["pymol", "-cq"]
pymol.finish_launching()
from pymol import cmd, util

# Load
cmd.load(r"{pdb_path}", "complex")

# === Global rendering — Nature Chemical Biology style ===
cmd.bg_color("white")
cmd.set("ray_opaque_background", 1)
cmd.set("orthoscopic", 1)
cmd.set("antialias", 2)
cmd.set("ray_trace_mode", 0)
cmd.set("ray_shadows", 0)
cmd.set("ambient", 0.50)
cmd.set("direct", 0.35)
cmd.set("reflect", 0.2)
cmd.set("spec_reflect", 0.02)
cmd.set("spec_power", 200)
cmd.set("depth_cue", 0)

# Cartoon quality
cmd.set("cartoon_fancy_helices", 1)
cmd.set("cartoon_smooth_loops", 1)
cmd.set("cartoon_sampling", 14)
cmd.set("cartoon_loop_radius", 0.2)
cmd.set("cartoon_oval_width", 0.25)
cmd.set("cartoon_oval_length", 1.2)

# Stick quality — NO stick_ball (Nature convention)
cmd.set("stick_ball", "off")
cmd.set("valence", 0)

# Distance label style
cmd.set("label_distance_digits", 1)
cmd.set("label_font_id", 7)
cmd.set("label_size", -0.5)
cmd.set("label_color", "black")
cmd.set("label_outline_color", "white")
cmd.set("label_bg_color", "white")
cmd.set("float_labels", 1)

# Dash defaults
cmd.set("dash_gap", 0.15)
cmd.set("dash_length", 0.15)
cmd.set("dash_radius", 0.08)
cmd.set("dash_color", "black")

# Extend H-bond detection cutoff to match PLIP (D-A up to 4.0A)
cmd.set("h_bond_cutoff_center", 4.2)
cmd.set("h_bond_cutoff_edge", 0.0)

# === Hide everything, then build up ===
cmd.hide("everything")
# Keep H atoms in model (needed for H-bond detection) but hide them
cmd.hide("everything", "elem H")

# Protein cartoon — opaque, light gray
cmd.show("cartoon", "polymer")
cmd.color("gray85", "polymer")
cmd.set("cartoon_transparency", 0.15, "polymer")

# Ligand — cyan sticks, prominent (no H shown)
cmd.select("lig", "resn {ligand}")
cmd.show("sticks", "lig and not elem H")
cmd.color("cyan", "lig and elem C")
util.cnc("lig")
cmd.set("stick_radius", 0.18, "lig")

# Binding site residues — wheat/tan sticks, sidechains only (no H)
cmd.select("bs", "resi {resi_list} and polymer")
cmd.show("sticks", "(bs and sidechain) and not elem H")
cmd.color("wheat", "bs and elem C")
util.cnc("bs")
cmd.set("stick_radius", 0.13, "bs")
# Also show backbone for backbone-interacting residues (needed for H-bond viz)
cmd.show("sticks", "(bs and name CA+C+N+O) and not elem H")
cmd.set("stick_radius", 0.08, "(bs and name CA+C+N+O)")

# === Residue labels (3-letter, black, offset) ===
cmd.set("label_color", "black")
cmd.set("label_outline_color", "white")
cmd.set("label_size", -0.8)
cmd.set("label_font_id", 7)
{labels_block}

# === H-bonds — BLACK dashed lines with distance ===
{hb_block}

# === Salt bridges — orange dashed ===
{sb_block}

# Note: Hydrophobic contacts shown by proximity (no explicit dashes — Nature convention)

# === Camera ===
# Center on ligand, frame binding site with protein context
cmd.center("lig")
cmd.zoom("lig or bs", 2)
cmd.turn("y", 30)
cmd.turn("x", -10)
cmd.deselect()

# === Render ===
t1 = time.time()
cmd.ray({width}, {height})
print(f"Ray: {{time.time()-t1:.1f}}s")
cmd.png(r"{output_png}", dpi={dpi})
print(f"Total: {{time.time()-t0:.1f}}s")
cmd.quit()
''')


def _build_pml(
    pdb_path, ligand, interactions, residues, output_png, width, height, dpi
) -> str:
    """Build a .pml for manual PyMOL use (same visuals, human-readable)."""
    resi_list = "+".join(
        sorted(set(r[1] for r in residues), key=lambda x: int(x) if x.isdigit() else 0)
    )

    # Group by type
    by_type = defaultdict(list)
    for ix in interactions:
        by_type[ix["type"]].append(ix)

    lines = [
        f"load {pdb_path}, complex",
        "",
        "# === Nature Chemical Biology / JACS style ===",
        "bg_color white",
        "set ray_opaque_background, 1",
        "set orthoscopic, 1",
        "set antialias, 2",
        "set ray_trace_mode, 0",
        "set ray_shadows, 0",
        "set ambient, 0.50",
        "set direct, 0.35",
        "set reflect, 0.2",
        "set spec_reflect, 0.02",
        "set spec_power, 200",
        "set depth_cue, 0",
        "",
        "# Cartoon quality",
        "set cartoon_fancy_helices, 1",
        "set cartoon_smooth_loops, 1",
        "set cartoon_sampling, 14",
        "set cartoon_loop_radius, 0.2",
        "",
        "# Sticks — no ball",
        "set stick_ball, off",
        "set valence, 0",
        "",
        "# Distance labels",
        "set label_distance_digits, 1",
        "set label_font_id, 7",
        "set label_size, -0.5",
        "set label_color, black",
        "set label_outline_color, white",
        "set label_bg_color, white",
        "set float_labels, 1",
        "",
        "# Dash defaults",
        "set dash_gap, 0.15",
        "set dash_length, 0.15",
        "set dash_radius, 0.08",
        "set dash_color, black",
        "",
        "# H-bond detection cutoff (match PLIP)",
        "set h_bond_cutoff_center, 4.2",
        "set h_bond_cutoff_edge, 0.0",
        "",
        "hide everything",
        "# Keep H atoms for H-bond detection, just hide them",
        "hide everything, elem H",
        "",
        "# Protein — light gray cartoon",
        "show cartoon, polymer",
        "color gray85, polymer",
        "set cartoon_transparency, 0.15, polymer",
        "",
        "# Ligand — cyan sticks",
        f"select lig, resn {ligand}",
        "show sticks, lig and not elem H",
        "color cyan, lig and elem C",
        "util.cnc lig",
        "set stick_radius, 0.18, lig",
        "",
        "# Binding site — wheat sticks",
        f"select bs, resi {resi_list} and polymer",
        "show sticks, (bs and sidechain) and not elem H",
        "color wheat, bs and elem C",
        "util.cnc bs",
        "set stick_radius, 0.13, bs",
        "show sticks, (bs and name CA+C+N+O) and not elem H",
        "set stick_radius, 0.08, (bs and name CA+C+N+O)",
        "",
    ]

    # Labels
    sorted_res = sorted(residues, key=lambda r: int(r[1]) if r[1].isdigit() else 0)
    offsets = [
        (0, 4.0, 2.5),
        (4.0, 2.0, 2.5),
        (-4.0, 2.0, 2.5),
        (4.0, -2.0, 2.5),
        (-4.0, -2.0, 2.5),
        (0, -4.0, 2.5),
    ]
    lines.append("# Residue labels")
    for i, (resn, resi, chain) in enumerate(sorted_res):
        lbl = f"{AA3.get(resn, resn)}{resi}"
        ox, oy, oz = offsets[i % len(offsets)]
        cs = f"chain {chain} and " if chain else ""
        sel = f"({cs}resi {resi}) and name CA"
        lines.append(f'label {sel}, "{lbl}"')
        lines.append(f"set label_position, [{ox},{oy},{oz}], {sel}")
        lines.append(f"set label_size, -0.8, {sel}")
        lines.append(f"set label_color, black, {sel}")
    lines.append("")

    # H-bonds
    lines.append("# H-bonds — black dashes")
    for key_resi in set(
        f"{ix['resn']}{ix['resi']}" for ix in by_type.get("hydrogen_bonds", [])
    ):
        ixlist = [
            ix
            for ix in by_type["hydrogen_bonds"]
            if f"{ix['resn']}{ix['resi']}" == key_resi
        ]
        ix0 = ixlist[0]
        cs = f"chain {ix0['chain']} and " if ix0["chain"] else ""
        has_sc = any(ix.get("sidechain") for ix in ixlist)
        if has_sc:
            res_sel = f"({cs}resi {ix0['resi']}) and (name OD1+OD2+OE1+OE2+NE2+NH1+NH2+NZ+OG+OG1+OH+ND1+ND2+NE1+SD+SG+N+O)"
        else:
            res_sel = f"({cs}resi {ix0['resi']}) and (name N+O)"
        lbl = AA3.get(ix0["resn"], ix0["resn"])
        obj = f"hb_{lbl}{ix0['resi']}"
        lines.append(
            f"distance {obj}, (resn {ligand}) and (elem N+O+S), {res_sel}, mode=2, cutoff=4.5"
        )
        lines.append(f"color black, {obj}")
        lines.append(f"set dash_radius, 0.08, {obj}")
        lines.append(f"set dash_gap, 0.15, {obj}")
        lines.append(f"set dash_length, 0.15, {obj}")
        lines.append(f"set label_size, -0.6, {obj}")

    # Salt bridges
    if by_type.get("salt_bridges"):
        lines.append("")
        lines.append("# Salt bridges — orange dashes")
        for key_resi in set(
            f"{ix['resn']}{ix['resi']}" for ix in by_type["salt_bridges"]
        ):
            ix0 = [
                ix
                for ix in by_type["salt_bridges"]
                if f"{ix['resn']}{ix['resi']}" == key_resi
            ][0]
            cs = f"chain {ix0['chain']} and " if ix0["chain"] else ""
            lbl = AA3.get(ix0["resn"], ix0["resn"])
            obj = f"sb_{lbl}{ix0['resi']}"
            lines.append(
                f"distance {obj}, (resn {ligand}) and (elem N+O), ({cs}resi {ix0['resi']}) and (elem N+O), mode=2, cutoff=5.0"
            )
            lines.append(f"color orange, {obj}")

    lines.extend(
        [
            "",
            "# Camera",
            "center lig",
            "zoom lig or bs, 2",
            "turn y, 30",
            "turn x, -10",
            "deselect",
            "",
            f"ray {width}, {height}",
            f"png {output_png}, dpi={dpi}",
        ]
    )
    return "\n".join(lines)


def _extract_interactions(interactions_dict: dict) -> list:
    result = []
    for site_key, site_data in interactions_dict.items():
        if not isinstance(site_data, dict):
            continue
        for itype, ilist in site_data.items():
            if not isinstance(ilist, list):
                continue
            for ix in ilist:
                if not isinstance(ix, dict):
                    continue
                res = ix.get("residue", "")
                m = re.match(r"([A-Z]+)(\d+)", res)
                if not m:
                    continue
                result.append(
                    {
                        "type": itype,
                        "resn": m.group(1),
                        "resi": m.group(2),
                        "chain": ix.get("reschain", ""),
                        "sidechain": ix.get("sidechain") in ("True", True),
                    }
                )
    return result


def _detect_ligand(pdb_path: str) -> str:
    SKIP = {
        "HOH",
        "WAT",
        "SOL",
        "NA",
        "CL",
        "K",
        "MG",
        "CA",
        "ZN",
        "FE",
        "MN",
        "SO4",
        "PO4",
        "GOL",
        "EDO",
    }
    counts = {}
    with open(pdb_path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("HETATM"):
                rn = line[17:20].strip()
                if rn and rn not in SKIP:
                    counts[rn] = counts.get(rn, 0) + 1
    return max(counts, key=counts.get) if counts else "UNK"
