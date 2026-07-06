"""
ChimeraX Protein-Ligand Interaction Visualizer
===============================================
Generates publication-quality interaction figures using UCSF ChimeraX.
Much faster than PyMOL/ChatMol: typically 5-15s vs 3-6 minutes.

Key advantages over PyMOL:
  - No LLM agent loop required
  - ChimeraX starts fast and exits cleanly
  - Built-in hbonds/clashes commands are more accurate than PyMOL mode=2
  - GPU-accelerated ray-quality rendering with supersample

Usage:
    from zhuangzhou.chimerax_viz.interactions import visualize_plip_interactions
    result = visualize_plip_interactions("complex.pdb", plip_result, output_dir="./figures")
"""

import logging
import os
import re
from typing import Optional

from .config import CHIMERAX_EXE, RENDER_DEFAULTS, TIMEOUT_RENDER, COLORS
from .render import _run_chimerax

logger = logging.getLogger(__name__)


# ── Interaction type colors (ChimeraX hex) ──
_IX_COLORS = {
    "hydrogen_bonds":      "#FFCC00",   # golden yellow dashes
    "hydrophobic_contacts": "#999999",  # gray dashes
    "pi_stacking":         "#AA44FF",   # purple
    "salt_bridges":        "#FF7700",   # orange
    "water_bridges":       "#55AAFF",   # light blue
    "metal_coordination":  "#FF3333",   # red
}

_IX_LABELS = {
    "hydrogen_bonds":      "H-bond",
    "hydrophobic_contacts": "Hydrophobic",
    "pi_stacking":         "π-stack",
    "salt_bridges":        "Salt bridge",
    "water_bridges":       "Water bridge",
    "metal_coordination":  "Metal coord.",
}


def _parse_plip_interactions(plip_result: dict) -> list:
    """
    Extract a flat list of interactions from PLIP result dict.
    Returns: [{type, residue, resnum, reschain, ligand_atom, dist, sidechain}]
    """
    entries = []

    # Support both flat plip_result and nested {"interactions": {...}} format
    interactions = plip_result.get("interactions", plip_result)

    for site_key, site_data in interactions.items():
        if not isinstance(site_data, dict):
            continue
        for itype, ilist in site_data.items():
            if not isinstance(ilist, list):
                continue
            for ix in ilist:
                if not isinstance(ix, dict):
                    continue
                resnum = ix.get("resnr", ix.get("resnum", 0))
                try:
                    resnum = int(resnum)
                except (ValueError, TypeError):
                    resnum = 0
                dist = (
                    ix.get("dist_h_a", ix.get("dist_ha", ix.get("dist", 3.5)))
                )
                try:
                    dist = float(dist)
                except (ValueError, TypeError):
                    dist = 3.5
                entries.append({
                    "type": itype,
                    "residue": ix.get("restype", ix.get("resname", "UNK")),
                    "resnum": resnum,
                    "reschain": ix.get("reschain", "A"),
                    "dist": round(dist, 2),
                    "sidechain": ix.get("sidechain") in (True, "True", "true", 1),
                })

    return entries


def _build_interaction_cxc(
    complex_pdb: str,
    output_png: str,
    plip_result: dict,
    ligand_resname: str,
    width: int = 3600,
    height: int = 2700,
    supersample: int = 3,
    show_hbonds_native: bool = True,
    show_contacts_native: bool = True,
) -> list:
    """
    Build ChimeraX command lines (.cxc) for interaction visualization.
    Returns list of command strings.
    """
    pdb_fwd = complex_pdb.replace("\\", "/")
    png_fwd = output_png.replace("\\", "/")
    lig = ligand_resname or "LIG"

    lines = [
        "# Zhuangzhou — ChimeraX interaction figure",
        f'open "{pdb_fwd}"',
        "set bgColor white",
        "",
        "# ── Protein: light gray cartoon ──",
        "cartoon",
        "hide atoms",
        "color #C8C8C8 cartoons",
        "graphics silhouettes true width 1.5",
        "",
        "# ── Ligand: green sticks ──",
        f"show :{lig} atoms",
        f"style :{lig} stick",
        f"color :{lig} byhetero",
        f"color :{lig}@C #44BB66",   # green carbons
        f"size :{lig} stickRadius 0.18",
        "",
    ]

    # Parse PLIP interactions
    interactions = _parse_plip_interactions(plip_result)

    # Collect unique interacting residues
    interacting_residues = {}  # (chain, resnum) -> {types}
    for ix in interactions:
        if ix["resnum"] > 0:
            key = (ix["reschain"], ix["resnum"])
            if key not in interacting_residues:
                interacting_residues[key] = {"residue": ix["residue"], "types": set()}
            interacting_residues[key]["types"].add(ix["type"])

    if interacting_residues:
        resnums = ",".join(str(r) for _, r in interacting_residues)
        lines += [
            "# ── Interacting residues: slate sticks ──",
            f"show :{resnums} atoms",
            f"style :{resnums} stick",
            f"color :{resnums}@C #4477BB",   # slate blue carbons
            f"color :{resnums} byhetero",    # N/O/S by element
            "",
        ]

    # H-bonds via ChimeraX native (most accurate, handles donor/acceptor properly)
    if show_hbonds_native:
        lines += [
            "# ── H-bonds (ChimeraX native detection) ──",
            f"hbonds :{lig} bothSides true color {_IX_COLORS['hydrogen_bonds']} "
            f"radius 0.07 dashes 8 reveal true",
            "",
        ]
    # Note: ChimeraX 'contacts' command does not support 'dashes'/'radius' styling.
    # Hydrophobic contacts are indicated by coloring interacting residues in slate-blue.

    # PLIP-derived distance pseudobonds for salt bridges / pi-stacking
    plip_pb_lines = []
    for ix in interactions:
        if ix["type"] in ("salt_bridges", "pi_stacking", "water_bridges") and ix["resnum"] > 0:
            color = _IX_COLORS.get(ix["type"], "#AAAAAA")
            chain = ix["reschain"]
            resn = ix["resnum"]
            dist_label = f"{ix['dist']:.1f}Å"
            # Show distance between residue Cα and nearest ligand atom
            plip_pb_lines.append(
                f"distance /{chain}:{resn}@CA :{lig} color {color} "
                f"; label '{_IX_LABELS.get(ix['type'], ix['type'])} {dist_label}'"
            )
    if plip_pb_lines:
        lines.append("# ── PLIP-derived contacts (salt bridge / π-stack) ──")
        lines.extend(plip_pb_lines)
        lines.append("")

    # Residue labels
    if interacting_residues:
        lines.append("# ── Residue labels ──")
        for (chain, resnum), info in interacting_residues.items():
            resname = info["residue"]
            lines.append(
                f"label /{chain}:{resnum}@CA text '{resname}{resnum}' "
                f"color black offset 0,0,1.5 size 14"
            )
        lines.append("")

    # Camera: zoom to ligand + binding site
    lines += [
        "# ── Camera ──",
        f"view :{lig} pad 0.5",
        "turn y 10",   # slight rotation for depth
        "",
        "# ── Lighting ──",
        "lighting soft",
        "lighting ambientIntensity 0.45",
        "lighting fillIntensity 0.35",
        "lighting directIntensity 0.6",
        "",
        f"# ── Save {width}x{height} PNG ──",
        f'save "{png_fwd}" width {width} height {height} supersample {supersample}',
        "exit",
    ]

    return lines


def visualize_plip_interactions(
    complex_pdb: str,
    plip_result: dict,
    output_dir: str = None,
    ligand_resname: str = None,
    affinity: float = None,
    rank: int = None,
    width: int = 3600,
    height: int = 2700,
    style: str = "publication",   # publication | presentation | minimal
    execute: bool = True,
    timeout: int = 90,
) -> dict:
    """
    Generate publication-quality protein-ligand interaction figure with ChimeraX.

    Uses ChimeraX's native hbond/contacts detection supplemented by PLIP data.
    Typical runtime: 5-15 seconds (vs 3-6 minutes for ChatMol/PyMOL).

    Args:
        complex_pdb:    Path to protein-ligand complex PDB
        plip_result:    Dict from run_plip_analysis() — used for interaction residues
                        and non-hbond contacts (salt bridge, pi-stack).
                        Pass {} to use ChimeraX-only detection.
        output_dir:     Output directory (default: <stem>_chimerax_plip/ in cwd)
        ligand_resname: Ligand 3-letter code (auto-detected from PDB if omitted)
        affinity:       Docking affinity for figure caption metadata
        rank:           Docking rank number for caption
        width/height:   Output PNG dimensions (default 3600x2700, 300 DPI at 12x9 in)
        style:          'publication' (3600px, silhouettes) | 'presentation' (2400px)
                        | 'minimal' (1200px, fast)
        execute:        Run ChimeraX immediately (True) or just write .cxc script
        timeout:        ChimeraX subprocess timeout in seconds

    Returns:
        dict with keys: png_path, cxc_path, output_dir, interactions_found,
                        success, chimerax_log (on error)
    """
    complex_pdb = os.path.abspath(complex_pdb)
    if not os.path.exists(complex_pdb):
        return {"error": f"PDB not found: {complex_pdb}"}

    if not CHIMERAX_EXE or not os.path.isfile(CHIMERAX_EXE):
        return {
            "error": "ChimeraX not found",
            "hint": f"Set CHIMERAX_EXE env var or install at {CHIMERAX_EXE!r}",
        }

    stem = os.path.splitext(os.path.basename(complex_pdb))[0]
    if not output_dir:
        output_dir = os.path.join(os.getcwd(), f"{stem}_chimerax_plip")
    os.makedirs(output_dir, exist_ok=True)

    # Auto-detect ligand residue name from PDB HETATM records
    if not ligand_resname:
        ligand_resname = _detect_ligand_resname(complex_pdb)

    # Style presets
    style_presets = {
        "publication":   {"width": 3600, "height": 2700, "supersample": 3},
        "presentation":  {"width": 2400, "height": 1800, "supersample": 2},
        "minimal":       {"width": 1200, "height":  900, "supersample": 1},
    }
    preset = style_presets.get(style, style_presets["publication"])
    w = width if width != 3600 else preset["width"]
    h = height if height != 2700 else preset["height"]
    ss = preset["supersample"]

    output_png = os.path.join(output_dir, f"{stem}_interactions.png")
    cxc_path = os.path.join(output_dir, f"{stem}_interactions.cxc")

    # Build .cxc commands
    cmd_lines = _build_interaction_cxc(
        complex_pdb=complex_pdb,
        output_png=output_png,
        plip_result=plip_result or {},
        ligand_resname=ligand_resname,
        width=w,
        height=h,
        supersample=ss,
    )

    # Write script
    with open(cxc_path, "w", encoding="utf-8") as f:
        f.write("\n".join(cmd_lines) + "\n")

    # Count interactions parsed
    interactions = _parse_plip_interactions(plip_result or {})
    ix_summary = {}
    for ix in interactions:
        ix_summary[ix["type"]] = ix_summary.get(ix["type"], 0) + 1

    result = {
        "output_dir": output_dir,
        "cxc_path": cxc_path,
        "png_path": output_png,
        "ligand_resname": ligand_resname,
        "interactions_found": ix_summary,
        "chimerax_exe": CHIMERAX_EXE,
    }
    if affinity is not None:
        result["affinity"] = affinity
    if rank is not None:
        result["rank"] = rank

    if not execute:
        result["success"] = False
        result["status"] = "script_only"
        result["note"] = f"ChimeraX script written. Open manually: {cxc_path}"
        return result

    logger.info("Running ChimeraX interaction viz (%s)...", stem)
    chimerax_out, ok = _run_chimerax(cxc_path, timeout=timeout)
    log_path = cxc_path.replace('.cxc', '_chimerax.log')

    if os.path.exists(output_png) and os.path.getsize(output_png) > 1000:
        result["success"] = True
        result["status"] = "rendered"
        result["file_size_kb"] = os.path.getsize(output_png) // 1024
        result["log_path"] = log_path
        logger.info("ChimeraX interaction PNG: %s (%d KB)",
                    output_png, result["file_size_kb"])
    else:
        result["success"] = False
        result["status"] = "script_only"
        result["warning"] = (
            "ChimeraX ran but PNG not produced. "
            "On Windows, check that paths have no spaces and ChimeraX is found. "
            f"Full log at: {log_path}  |  CXC script: {cxc_path}"
        )
        # Provide full log for debugging (truncated only for display)
        result["chimerax_log"] = chimerax_out[-2000:] if chimerax_out else ""
        result["log_path"] = log_path

    return result


def _detect_ligand_resname(pdb_path: str) -> Optional[str]:
    """
    Auto-detect ligand residue name from PDB HETATM records.
    Skips water (HOH/WAT/H2O) and common cofactors (HEM/MG/ZN/CA/FE).
    Returns 3-letter code or None.
    """
    _SKIP = {"HOH", "WAT", "H2O", "HEM", "HEC", "HEB", "MG", "ZN", "CA",
              "FE", "NA", "CL", "SO4", "PO4", "GOL", "EDO", "PEG", "ACT"}
    resname_counts: dict = {}
    try:
        with open(pdb_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("HETATM"):
                    rn = line[17:20].strip()
                    if rn and rn not in _SKIP:
                        resname_counts[rn] = resname_counts.get(rn, 0) + 1
    except OSError:
        return None

    if not resname_counts:
        return None
    # Return the most common HETATM residue name
    return max(resname_counts, key=lambda k: resname_counts[k])


def generate_interaction_overview(
    complex_pdb: str,
    plip_result: dict,
    output_dir: str = None,
    ligand_resname: str = None,
    execute: bool = True,
    timeout: int = 90,
) -> dict:
    """
    Generate both detail (binding site zoom) and global (full protein) views.

    Returns dict with:
        detail_png: zoomed binding site view (3600x2700)
        global_png: full protein context view (3600x2700)
        cxc_path: combined .cxc script
    """
    complex_pdb = os.path.abspath(complex_pdb)
    stem = os.path.splitext(os.path.basename(complex_pdb))[0]

    if not output_dir:
        output_dir = os.path.join(os.getcwd(), f"{stem}_chimerax_plip")
    os.makedirs(output_dir, exist_ok=True)

    if not ligand_resname:
        ligand_resname = _detect_ligand_resname(complex_pdb)

    lig = ligand_resname or "LIG"
    pdb_fwd = complex_pdb.replace("\\", "/")
    detail_png = os.path.join(output_dir, f"{stem}_detail.png")
    global_png = os.path.join(output_dir, f"{stem}_global.png")
    detail_fwd = detail_png.replace("\\", "/")
    global_fwd = global_png.replace("\\", "/")

    # Parse interactions for residue highlights
    interactions = _parse_plip_interactions(plip_result or {})
    ix_residues = list({ix["resnum"] for ix in interactions if ix["resnum"] > 0})
    res_spec = ",".join(str(r) for r in ix_residues) if ix_residues else ""

    lines = [
        "# Zhuangzhou — ChimeraX interaction overview (detail + global)",
        f'open "{pdb_fwd}"',
        "set bgColor white",
        "",
        "# Base styling",
        "cartoon",
        "hide atoms",
        "color #C8C8C8 cartoons",
        "graphics silhouettes true width 1.5",
        "",
        f"show :{lig} atoms",
        f"style :{lig} stick",
        f"color :{lig} byhetero",
        f"color :{lig}@C #44BB66",
        "",
    ]

    if res_spec:
        lines += [
            f"show :{res_spec} atoms",
            f"style :{res_spec} stick",
            f"color :{res_spec}@C #4477BB",
            f"color :{res_spec} byhetero",
            "",
        ]

    lines += [
        f"hbonds :{lig} bothSides true color {_IX_COLORS['hydrogen_bonds']} "
        f"radius 0.07 dashes 8 reveal true",
        "",
    ]

    if ix_residues:
        lines.append("# Residue labels")
        for ix in interactions:
            if ix["resnum"] > 0:
                lines.append(
                    f"label /{ix['reschain']}:{ix['resnum']}@CA "
                    f"text '{ix['residue']}{ix['resnum']}' color black offset 0,0,1.5 size 14"
                )
        lines.append("")

    lines += [
        "# ── Detail view: zoom to binding site ──",
        "lighting soft",
        "lighting ambientIntensity 0.45",
        "lighting fillIntensity 0.35",
        f"view :{lig} pad 0.4",
        "turn y 10",
        f'save "{detail_fwd}" width 3600 height 2700 supersample 3',
        "",
        "# ── Global view: show full protein ──",
        "view all",
        f'save "{global_fwd}" width 3600 height 2700 supersample 3',
        "",
        "exit",
    ]

    cxc_path = os.path.join(output_dir, f"{stem}_overview.cxc")
    with open(cxc_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    result = {
        "output_dir": output_dir,
        "cxc_path": cxc_path,
        "detail_png": detail_png,
        "global_png": global_png,
        "ligand_resname": lig,
    }

    if not execute:
        result["success"] = False
        result["status"] = "script_only"
        return result

    chimerax_out, ok = _run_chimerax(cxc_path, timeout=timeout)
    detail_ok = os.path.exists(detail_png) and os.path.getsize(detail_png) > 1000
    global_ok = os.path.exists(global_png) and os.path.getsize(global_png) > 1000

    result["success"] = detail_ok or global_ok
    result["detail_rendered"] = detail_ok
    result["global_rendered"] = global_ok
    if not (detail_ok or global_ok):
        result["chimerax_log"] = chimerax_out[-600:] if chimerax_out else ""

    return result
