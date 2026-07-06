from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from cypforge_core.io import parse_frcmod_sections


# ---------------------------------------------------------------------------
# Heme state constants (Shahrokh et al. J. Comput. Chem. 2012, 33, 119-133)
# Numbers in IC-labels denote spin multiplicity (2S+1), NOT coordination number
# ---------------------------------------------------------------------------
HEME_STATE_IC6   = "IC6"    # Fe鲁鈦? high-spin S=5/2, penta-coordinate, no distal ligand
HEME_STATE_CPDI  = "CPDI"   # Fe鈦粹伜=O, Compound I, hexa-coordinate, distal oxo (~1.65 脜)
HEME_STATE_DIOXY = "DIOXY"  # Fe虏鈦?O-O, ferrous-oxy, hexa-coordinate, distal dioxygen
HEME_STATE_CUSTOM = "CUSTOM"  # user-supplied mol2 + frcmod, bypasses auto-detection

VALID_HEME_STATES = (HEME_STATE_IC6, HEME_STATE_CPDI, HEME_STATE_DIOXY, HEME_STATE_CUSTOM)
ALGORITHM_VERSION = "v1.1"

# Detection thresholds (脜)
CPDI_O_CUTOFF  = 1.70   # Fe=O bond in Compound I is ~1.65 脜
WATER_O_CUTOFF = 2.50   # upper bound for any Fe-O coordination

# Core porphyrin ring atom names
CORE_RING_NAMES = {
    "FE", "NA", "NB", "NC", "ND",
    "CHA", "CHB", "CHC", "CHD",
    "C1A", "C2A", "C3A", "C4A",
    "C1B", "C2B", "C3B", "C4B",
    "C1C", "C2C", "C3C", "C4C",
    "C1D", "C2D", "C3D", "C4D",
}

PROPIONATE_A = {"CAA", "CBA", "CGA", "O1A", "O2A"}
PROPIONATE_D = {"CAD", "CBD", "CGD", "O1D", "O2D"}
CLASS_RANK = {
    "axial_o": 0,
    "macrocycle": 1,
    "prop_a": 2,
    "prop_d": 3,
    "peripheral": 4,
    "other": 5,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AtomRecord:
    name: str
    coord: np.ndarray
    serial: int = 0
    element: str = ""
    chain: str = ""
    resid: int = 0
    resname: str = "HEM"
    altloc: str = ""
    occupancy: float = 0.0


@dataclass(frozen=True)
class LocalFrame:
    origin: np.ndarray
    x_axis: np.ndarray
    y_axis: np.ndarray
    z_axis: np.ndarray

    @property
    def basis(self) -> np.ndarray:
        return np.column_stack((self.x_axis, self.y_axis, self.z_axis))

    def to_local(self, point: np.ndarray) -> np.ndarray:
        return self.basis.T @ (point - self.origin)

    def to_global(self, local_point: np.ndarray) -> np.ndarray:
        return self.origin + self.basis @ local_point


@dataclass
class MappingResultV2:
    source_atoms: List[AtomRecord]
    template_atoms: List[AtomRecord]
    source_frame: LocalFrame
    template_frame: LocalFrame
    direct_matches: Dict[str, str]
    slot_assignments: Dict[str, str]
    completed_heavy_atoms: Dict[str, np.ndarray]
    heme_state: str
    diagnostics: Dict[str, object]


# ---------------------------------------------------------------------------
# State detection
# ---------------------------------------------------------------------------

@dataclass
class StateDetectionResult:
    state: str
    method: str          # "auto" | "manual"
    distal_o_count: int
    distal_o_distances: List[float]
    warning: Optional[str]


KNOWN_GAFF_TYPES = {
    "c", "c1", "c2", "c3", "ca", "cb", "cc", "cd", "ce", "cf", "cg", "ch",
    "cp", "cq", "ct", "cu", "cv", "cx", "cy",
    "h", "h1", "h2", "h3", "h4", "h5", "ha", "hc", "hn", "ho", "hp", "hs", "hw",
    "n", "n1", "n2", "n3", "n4", "na", "nb", "nc", "nd", "ne", "nf", "nh", "no",
    "o", "oh", "os", "ow",
    "f", "cl", "br", "i",
    "p", "p2", "p3", "p4", "p5",
    "s", "s2", "s4", "s6", "sh", "ss",
    "fe",
}


def validate_custom_heme_params(
    *,
    heme_mol2: str | Path,
    cyp_mol2: str | Path,
    frcmod: str | Path,
    state_label: str,
) -> dict[str, object]:
    """Validate user-supplied heme/CYP mol2 and frcmod files.

    Minimal checks: FE in HEM.mol2, SG in CYP.mol2, fe-SH bond in frcmod,
    atom types are known GAFF/GAFF2, charge sanity (~-2.0 for Cys-thiolate).
    """
    errors: list[str] = []
    warnings: list[str] = []

    heme_atoms = load_template_atoms_from_mol2(Path(heme_mol2))
    cyp_atoms = load_template_atoms_from_mol2(Path(cyp_mol2))

    # FE presence
    fe_atoms = [a for a in heme_atoms if a.name.upper() == "FE"]
    if not fe_atoms:
        errors.append("HEM.mol2 has no FE atom")

    # SG presence
    if not any(a.name.upper() == "SG" for a in cyp_atoms):
        errors.append("CYP.mol2 has no SG atom")

    # atom type scan
    def _raw_mol2_types(path: Path) -> set[str]:
        types: set[str] = set()
        in_atoms = False
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("@<TRIPOS>ATOM"):
                in_atoms = True
                continue
            if in_atoms and line.startswith("@<TRIPOS>"):
                break
            if in_atoms and line.strip():
                parts = line.split()
                if len(parts) >= 6:
                    types.add(parts[5].lower())
        return types

    all_types = _raw_mol2_types(Path(heme_mol2)) | _raw_mol2_types(Path(cyp_mol2))
    unknown = all_types - KNOWN_GAFF_TYPES
    if unknown:
        warnings.append(f"Atom types not in known GAFF/GAFF2 set: {sorted(unknown)}")

    # fe-SH bond parameter
    frcmod_sections = parse_frcmod_sections(Path(frcmod))
    fe_sh_found = any(
        line.strip().lower().startswith("fe-sh")
        for line in frcmod_sections.get("BOND", [])
    )
    if not fe_sh_found:
        errors.append("frcmod has no fe-SH bond parameter")

    # charge sanity
    def _mol2_charge(path: Path) -> float:
        total = 0.0
        in_atoms = False
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("@<TRIPOS>ATOM"):
                in_atoms = True
                continue
            if in_atoms and line.startswith("@<TRIPOS>"):
                break
            if in_atoms and line.strip():
                parts = line.split()
                if len(parts) >= 9:
                    total += float(parts[8])
        return total

    heme_q = _mol2_charge(Path(heme_mol2))
    cyp_q = _mol2_charge(Path(cyp_mol2))
    total_q = heme_q + cyp_q
    if abs(total_q - (-2.0)) > 0.5:
        warnings.append(
            f"HEM({heme_q:.4f}) + CYP({cyp_q:.4f}) = {total_q:.4f}; "
            f"expected ~-2.0 for standard Cys-thiolate"
        )

    status = "FAIL" if errors else "WARN" if warnings else "PASS"
    return {
        "status": status,
        "state_label": state_label,
        "heme_mol2": str(heme_mol2),
        "cyp_mol2": str(cyp_mol2),
        "frcmod": str(frcmod),
        "heme_charge": round(heme_q, 6),
        "cyp_charge": round(cyp_q, 6),
        "heme_plus_cyp_charge": round(total_q, 6),
        "fe_count": len(fe_atoms),
        "sg_present": any(a.name.upper() == "SG" for a in cyp_atoms),
        "fe_sh_bond_in_frcmod": fe_sh_found,
        "unknown_atom_types": sorted(unknown) if unknown else [],
        "errors": errors,
        "warnings": warnings,
    }




def detect_heme_state(
    heme_atoms: Sequence[AtomRecord],
    manual_state: Optional[str] = None,
    all_atoms: Optional[Sequence[AtomRecord]] = None,
) -> StateDetectionResult:
    """
    Determine the heme iron spin/coordination state.

    Auto-detection logic (from PDB coordinates):
      - 0 O atoms within 2.5 脜 of Fe 鈫?IC6  (penta-coordinate, resting)
      - 1 O atom, distance < 1.70 脜   鈫?CPDI (Compound I, Fe=O)
      - 2 O atoms within 2.5 脜 of Fe  鈫?DIOXY (ferrous-oxy, Fe-O-O)
      - 1 O atom, 1.70鈥?.50 脜         鈫?water-bound (not parameterised),
                                          fallback to IC6 with a warning

    Note: distal O ligands are often in a separate residue (e.g. OXY, O) rather
    than the HEM residue itself. Pass all_atoms (all ATOM/HETATM records from
    the PDB) to enable cross-residue detection.

    Manual override:
      Pass manual_state="IC6" / "CPDI" / "DIOXY" to bypass auto-detection.
      Useful when researchers want to study a specific spin state regardless
      of what is present in the crystal structure.
    """
    if manual_state is not None:
        if manual_state not in VALID_HEME_STATES:
            raise ValueError(
                f"Unknown heme state '{manual_state}'. "
                f"Valid options: {VALID_HEME_STATES}"
            )
        if manual_state == HEME_STATE_CUSTOM:
            return StateDetectionResult(
                state=HEME_STATE_CUSTOM,
                method="custom",
                distal_o_count=0,
                distal_o_distances=[],
                warning=(
                    "Custom heme state 鈥?auto-detection of distal ligands is "
                    "bypassed. The user is responsible for providing correct "
                    "mol2 and frcmod files."
                ),
            )
        return StateDetectionResult(
            state=manual_state,
            method="manual",
            distal_o_count=0,
            distal_o_distances=[],
            warning=None,
        )

    by_name = {atom.name: atom for atom in heme_atoms}
    if "FE" not in by_name:
        raise ValueError("Cannot detect heme state: no FE atom found.")
    fe_coord = by_name["FE"].coord

    # Search pool: HEM atoms + any nearby non-protein atoms from other residues
    # (handles OXY/O/HOH residues that carry the distal ligand)
    porphyrin_o_names = PROPIONATE_A | PROPIONATE_D
    protein_resnames = {
        "ALA","ARG","ASN","ASP","CYS","CYM","CYX","CYP",
        "GLN","GLU","GLY","HIS","HID","HIE","HIP","ILE",
        "LEU","LYS","MET","PHE","PRO","SER","THR","TRP",
        "TYR","VAL","HEM",
    }

    search_atoms: List[AtomRecord] = list(heme_atoms)
    if all_atoms is not None:
        for atom in all_atoms:
            if atom.resname.upper() in protein_resnames:
                continue
            if atom.element != "O":
                continue
            dist = float(np.linalg.norm(atom.coord - fe_coord))
            if dist <= WATER_O_CUTOFF:
                search_atoms.append(atom)

    # Collect O atoms within WATER_O_CUTOFF of Fe, excluding propionate oxygens
    distal_o: List[Tuple[float, str]] = []
    for atom in search_atoms:
        if atom.element != "O":
            continue
        if atom.name in porphyrin_o_names and atom.resname.upper() == "HEM":
            continue
        dist = float(np.linalg.norm(atom.coord - fe_coord))
        if dist <= WATER_O_CUTOFF:
            distal_o.append((dist, atom.name))

    distal_o.sort()
    o_count = len(distal_o)
    o_dists = [d for d, _ in distal_o]

    if o_count == 0:
        return StateDetectionResult(
            state=HEME_STATE_IC6,
            method="auto",
            distal_o_count=0,
            distal_o_distances=[],
            warning=None,
        )

    # Check for DIOXY: Fe-O1 within 2.5 A, and O1-O2 within 1.6 A (O-O bond ~1.25 A)
    # In ferrous-oxy, O2 is at ~2.9 A from Fe (beyond WATER_O_CUTOFF), but within
    # ~1.3 A of O1. So we extend the search pool to 4 A from Fe for O-O detection.
    extended_pool: List[AtomRecord] = list(search_atoms)
    if all_atoms is not None:
        for atom in all_atoms:
            if atom.resname.upper() in protein_resnames:
                continue
            if atom.element != "O":
                continue
            dist = float(np.linalg.norm(atom.coord - fe_coord))
            if WATER_O_CUTOFF < dist <= 4.0:   # beyond primary cutoff, up to 4 A
                extended_pool.append(atom)
    pool_for_oo = extended_pool
    for i, (d1, n1) in enumerate(distal_o):
        o1_coord = next(a.coord for a in pool_for_oo if a.name == n1 and a.element == "O")
        for atom in pool_for_oo:
            if atom.element != "O" or atom.name == n1:
                continue
            if atom.name in porphyrin_o_names and atom.resname.upper() == "HEM":
                continue
            oo_dist = float(np.linalg.norm(atom.coord - o1_coord))
            if 1.0 <= oo_dist <= 1.6:   # O-O bond distance range
                # Found dioxygen pair
                all_dists = o_dists + [round(oo_dist, 3)]
                return StateDetectionResult(
                    state=HEME_STATE_DIOXY,
                    method="auto",
                    distal_o_count=2,
                    distal_o_distances=sorted(all_dists),
                    warning=None,
                )

    if o_count >= 2:
        return StateDetectionResult(
            state=HEME_STATE_DIOXY,
            method="auto",
            distal_o_count=o_count,
            distal_o_distances=o_dists,
            warning=None,
        )

    # Exactly 1 O within 2.5 A
    closest_dist = distal_o[0][0]
    if closest_dist < CPDI_O_CUTOFF:
        return StateDetectionResult(
            state=HEME_STATE_CPDI,
            method="auto",
            distal_o_count=1,
            distal_o_distances=o_dists,
            warning=None,
        )

    # 1 O at 1.70-2.50 A: water-bound hexacoordinate (Shahrokh not parameterised)
    return StateDetectionResult(
        state=HEME_STATE_IC6,
        method="auto",
        distal_o_count=1,
        distal_o_distances=o_dists,
        warning=(
            f"One distal O detected at {closest_dist:.2f} A -- consistent with a "
            "water-bound hexacoordinate ferric state. Shahrokh 2012 does not provide "
            "parameters for this state. Falling back to IC6 (penta-coordinate). "
            "Use --heme-state to override."
        ),
    )


# ---------------------------------------------------------------------------
# Template resolution (state-aware)
# ---------------------------------------------------------------------------

_MODULE_ROOT = Path(__file__).resolve().parent
_PACKAGE_ROOT = _MODULE_ROOT.parent
_PARAMS_ROOT = _PACKAGE_ROOT / "data" / "heme_params"

# 鍛藉悕瑙勮寖锛歿state}-HEM.mol2锛屽 IC6-HEM.mol2 / CPDI-HEM.mol2 / DIOXY-HEM.mol2
_STATE_TEMPLATE_NAMES: Dict[str, List[str]] = {
    HEME_STATE_IC6: ["HEM.mol2", "IC6-HEM.mol2"],
    HEME_STATE_CPDI: ["HEM.mol2", "CPDI-HEM.mol2"],
    HEME_STATE_DIOXY: ["HEM.mol2", "DIOXY-HEM.mol2"],
}


def resolve_template_mol2_path(
    state: str,
    template_mol2_path: Optional[str] = None,
) -> Path:
    """Return the mol2 template path for *state*.

    Priority:
      1. Explicit path supplied by caller.
      2. State-specific file in sibling params/{state}/ directory.
    """
    if template_mol2_path:
        path = Path(template_mol2_path)
        if not path.exists():
            raise FileNotFoundError(f"Template mol2 not found: {path}")
        return path

    if state == HEME_STATE_CUSTOM:
        raise FileNotFoundError(
            "CUSTOM heme state requires an explicit --custom-heme-mol2 path; "
            "no built-in template exists for CUSTOM."
        )

    for candidate in _STATE_TEMPLATE_NAMES.get(state, []):
        path = _PARAMS_ROOT / state / candidate
        if path.exists():
            return path

    raise FileNotFoundError(
        f"No template mol2 found for state '{state}'. "
        f"Expected: {_PARAMS_ROOT / state / _STATE_TEMPLATE_NAMES.get(state, ['?'])[0]}. "
        "Provide a custom path via --template-mol2."
    )


# ---------------------------------------------------------------------------
# Parsing helpers (unchanged from v1)
# ---------------------------------------------------------------------------

def infer_element(atom_name: str) -> str:
    atom_name = atom_name.strip().upper()
    if atom_name.startswith("FE"):
        return "FE"
    for ch in atom_name:
        if ch.isalpha():
            return ch
    return ""


def infer_element_from_pdb_line(line: str) -> str:
    element = line[76:78].strip().upper()
    if element:
        return element
    return infer_element(line[12:16])


def parse_pdb_atoms(pdb_path: str, chain_id: Optional[str] = None) -> List[AtomRecord]:
    atoms: List[AtomRecord] = []
    with open(pdb_path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            if chain_id and line[21].strip() != chain_id:
                continue
            atoms.append(AtomRecord(
                name=line[12:16].strip(),
                coord=np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])], dtype=float),
                serial=int(line[6:11]),
                element=infer_element_from_pdb_line(line),
                chain=line[21].strip(),
                resid=int(line[22:26]),
                resname=line[17:20].strip(),
                altloc=line[16].strip(),
                occupancy=float(line[54:60] or 0.0),
            ))
    return atoms


def parse_pdb_heme_atoms(
    pdb_path: str,
    resname: str = "HEM",
    chain_id: Optional[str] = None,
) -> List[AtomRecord]:
    return [a for a in parse_pdb_atoms(pdb_path, chain_id=chain_id) if a.resname == resname]


def find_proximal_anchor(
    pdb_path: str,
    fe_coord: np.ndarray,
    anchor_chain: Optional[str] = None,
    axial_cys_resid: Optional[int] = None,
    anchor_atom_name: str = "SG",
    allowed_resnames: Sequence[str] = ("CYS", "CYM", "CYX", "CYP"),
) -> Optional[AtomRecord]:
    candidates: List[Tuple[float, AtomRecord]] = []
    allowed = {n.upper() for n in allowed_resnames}
    for atom in parse_pdb_atoms(pdb_path, chain_id=anchor_chain):
        if atom.name.upper() != anchor_atom_name.upper():
            continue
        if atom.resname.upper() not in allowed:
            continue
        if axial_cys_resid is not None and atom.resid != axial_cys_resid:
            continue
        distance = float(np.linalg.norm(atom.coord - fe_coord))
        candidates.append((distance, atom))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    for dist, atom in candidates:
        if 1.5 <= dist <= 3.2:
            return atom
    return candidates[0][1]


def dedupe_heme_atoms(atoms: Sequence[AtomRecord]) -> Tuple[List[AtomRecord], Dict[str, List[AtomRecord]]]:
    grouped: Dict[Tuple[str, str, int, str], List[AtomRecord]] = {}
    for atom in atoms:
        key = (atom.name, atom.chain, atom.resid, atom.resname)
        grouped.setdefault(key, []).append(atom)

    def score(a: AtomRecord) -> Tuple[int, float, int]:
        return (0 if a.altloc == "" else 1 if a.altloc == "A" else 2, -a.occupancy, a.serial)

    chosen: List[AtomRecord] = []
    duplicates: Dict[str, List[AtomRecord]] = {}
    for key, records in grouped.items():
        records = sorted(records, key=score)
        chosen.append(records[0])
        if len(records) > 1:
            duplicates[key[0]] = records
    chosen.sort(key=lambda a: a.serial)
    return chosen, duplicates


# ---------------------------------------------------------------------------
# Geometry helpers (unchanged from v1)
# ---------------------------------------------------------------------------

def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("Cannot normalize a zero-length vector.")
    return v / n


def project_to_plane(v: np.ndarray, normal: np.ndarray) -> np.ndarray:
    return v - np.dot(v, normal) * normal


def fit_plane(points: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    coords = np.asarray(points, dtype=float)
    centroid = coords.mean(axis=0)
    _, _, vh = np.linalg.svd(coords - centroid, full_matrices=False)
    return centroid, normalize(vh[-1])


def detect_carboxyl_carbons(atoms: Sequence[AtomRecord]) -> List[Tuple[AtomRecord, List[AtomRecord]]]:
    oxygens = [a for a in atoms if a.element == "O"]
    carbons = [a for a in atoms if a.element == "C"]
    groups = []
    for c in carbons:
        bonded = [o for o in oxygens if 1.15 <= np.linalg.norm(c.coord - o.coord) <= 1.45]
        if len(bonded) >= 2:
            groups.append((c, sorted(bonded, key=lambda a: a.serial)[:2]))
    return groups


def pick_propionate_pair(
    fe: np.ndarray,
    groups: Sequence[Tuple[AtomRecord, List[AtomRecord]]],
) -> Tuple[AtomRecord, AtomRecord]:
    if len(groups) < 2:
        raise ValueError("Could not detect two carboxyl carbons.")
    ranked = sorted(groups, key=lambda x: np.linalg.norm(x[0].coord - fe), reverse=True)
    return ranked[0][0], ranked[1][0]


def template_class(name: str) -> str:
    if name in CORE_RING_NAMES:
        return "macrocycle"
    if name in PROPIONATE_A:
        return "prop_a"
    if name in PROPIONATE_D:
        return "prop_d"
    if name in ("O1", "O2"):
        return "axial_o"
    return "peripheral"


def source_class(atom: AtomRecord, carboxyl_lookup: Mapping[str, str], frame: LocalFrame) -> str:
    if atom.name in carboxyl_lookup:
        return carboxyl_lookup[atom.name]
    local = frame.to_local(atom.coord)
    radius = math.hypot(local[0], local[1])
    if atom.element == "O" and abs(local[2]) > 0.8:
        return "axial_o"
    if radius <= 4.8:
        return "macrocycle"
    return "peripheral"


def build_source_carboxyl_lookup(atoms: Sequence[AtomRecord], frame: LocalFrame) -> Dict[str, str]:
    groups = detect_carboxyl_carbons(atoms)
    if len(groups) < 2:
        return {}
    chosen = list(pick_propionate_pair(frame.origin, groups))
    chosen.sort(key=lambda a: frame.to_local(a.coord)[1])
    labels: Dict[str, str] = {chosen[0].name: "prop_d", chosen[1].name: "prop_a"}
    by_name = {a.name: a for a in atoms}
    carbons = [a for a in atoms if a.element == "C"]
    for cname, label in list(labels.items()):
        carbon = by_name[cname]
        for o in [a for a in atoms if a.element == "O"]:
            if 1.15 <= np.linalg.norm(carbon.coord - o.coord) <= 1.45:
                labels[o.name] = label
        first_hop = []
        for other in carbons:
            if other.name == carbon.name:
                continue
            if 1.20 <= np.linalg.norm(carbon.coord - other.coord) <= 1.85:
                first_hop.append(other)
                labels[other.name] = label
        for hop in first_hop:
            for other in carbons:
                if other.name not in labels and 1.20 <= np.linalg.norm(hop.coord - other.coord) <= 1.85:
                    labels[other.name] = label
    return labels


def local_descriptor(atom: AtomRecord, frame: LocalFrame) -> Tuple[float, float, float]:
    local = frame.to_local(atom.coord)
    return math.hypot(local[0], local[1]), math.atan2(local[1], local[0]), local[2]


def template_slot_key(atom: AtomRecord, frame: LocalFrame) -> Tuple[int, str, float, float, float]:
    r, a, z = local_descriptor(atom, frame)
    return (CLASS_RANK.get(template_class(atom.name), 99), atom.element, a, r, z)


def source_slot_key(atom: AtomRecord, frame: LocalFrame, labels: Mapping[str, str]) -> Tuple[int, str, float, float, float]:
    r, a, z = local_descriptor(atom, frame)
    return (CLASS_RANK.get(labels.get(atom.name, "other"), 99), atom.element, a, r, z)


def assign_source_atoms_to_template_slots(
    source_atoms: Sequence[AtomRecord],
    template_atoms: Sequence[AtomRecord],
    source_frame: LocalFrame,
    template_frame: LocalFrame,
    source_labels: Mapping[str, str],
) -> Dict[str, str]:
    assignments: Dict[str, str] = {}
    grouped_src: Dict[Tuple[str, str], List[AtomRecord]] = {}
    grouped_tpl: Dict[Tuple[str, str], List[AtomRecord]] = {}
    for a in source_atoms:
        grouped_src.setdefault((source_labels.get(a.name, "other"), a.element), []).append(a)
    for a in template_atoms:
        grouped_tpl.setdefault((template_class(a.name), a.element), []).append(a)
    for key, tpl_group in grouped_tpl.items():
        src_group = grouped_src.get(key, [])
        src_sorted = sorted(src_group, key=lambda a: source_slot_key(a, source_frame, source_labels))
        tpl_sorted = sorted(tpl_group, key=lambda a: template_slot_key(a, template_frame))
        for sa, ta in zip(src_sorted, tpl_sorted):
            assignments[sa.name] = ta.name
    return assignments


def load_template_atoms_from_mol2(path: Path) -> List[AtomRecord]:
    atoms: List[AtomRecord] = []
    in_atoms = False
    serial = 1
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("@<TRIPOS>ATOM"):
            in_atoms = True
            continue
        if line.startswith("@<TRIPOS>") and not line.startswith("@<TRIPOS>ATOM"):
            in_atoms = False
        if not in_atoms:
            continue
        parts = line.split()
        if len(parts) < 6:
            continue
        atoms.append(AtomRecord(
            name=parts[1],
            coord=np.array([float(parts[2]), float(parts[3]), float(parts[4])], dtype=float),
            serial=serial,
            element=infer_element(parts[1]),
            chain="T", resid=1, resname="HEM", altloc="", occupancy=1.0,
        ))
        serial += 1
    return atoms


def build_template_frame_from_atoms(template_atoms: Sequence[AtomRecord]) -> LocalFrame:
    by_name = {a.name: a for a in template_atoms}
    fe = by_name["FE"].coord
    ring_pts = [by_name[n].coord for n in CORE_RING_NAMES if n in by_name and n != "FE"]
    _, z_axis = fit_plane(ring_pts)
    # Orient z toward distal side (away from proximal CYS).
    # If O1 present (CPDI/DIOXY), z points toward the distal ligand.
    if "O1" in by_name:
        if np.dot(z_axis, by_name["O1"].coord - fe) < 0:
            z_axis = -z_axis
    else:
        # IC6 has no distal O marker. The two propionate carboxyl carbons
        # belong on the proximal/thiolate side, so their midpoint must have
        # negative local z when +z is the distal face.
        cga, cgd = by_name["CGA"].coord, by_name["CGD"].coord
        propionate_midpoint = (cga + cgd) / 2.0
        if np.dot(z_axis, propionate_midpoint - fe) > 0:
            z_axis = -z_axis
    cga, cgd = by_name["CGA"].coord, by_name["CGD"].coord
    midpoint = (cga + cgd) / 2.0
    x_axis = normalize(project_to_plane(midpoint - fe, z_axis))
    y_axis = normalize(np.cross(z_axis, x_axis))
    x_axis = normalize(np.cross(y_axis, z_axis))
    return LocalFrame(origin=fe, x_axis=x_axis, y_axis=y_axis, z_axis=z_axis)


def mapped_propionate_side_diagnostics(
    completed: Mapping[str, np.ndarray],
    proximal_anchor: Optional[np.ndarray],
    tolerance: float = 0.05,
) -> Dict[str, object]:
    """Check mapped heme face orientation using SG/CGA/CGD signed distances."""
    diag: Dict[str, object] = {"propionate_side_qc": "not-run"}
    if proximal_anchor is None:
        diag["propionate_side_qc_reason"] = "proximal_anchor_missing"
        return diag
    required = ("FE", "NA", "NB", "NC", "ND", "CGA", "CGD")
    missing = [name for name in required if name not in completed]
    if missing:
        diag["propionate_side_qc"] = "not-run"
        diag["propionate_side_qc_reason"] = "missing_atoms"
        diag["propionate_side_qc_missing_atoms"] = missing
        return diag

    fe = completed["FE"]
    _, normal = fit_plane([completed[name] for name in ("NA", "NB", "NC", "ND")])

    def signed_distance(point: np.ndarray) -> float:
        return float(np.dot(normal, point - fe))

    sg_d = signed_distance(np.asarray(proximal_anchor, dtype=float))
    cga_d = signed_distance(completed["CGA"])
    cgd_d = signed_distance(completed["CGD"])
    prop_mid_d = 0.5 * (cga_d + cgd_d)
    same_side_products = {
        "SGxCGA": sg_d * cga_d,
        "SGxCGD": sg_d * cgd_d,
        "SGxPropionateMidpoint": sg_d * prop_mid_d,
    }
    pass_qc = all(value > tolerance for value in same_side_products.values())
    diag.update({
        "propionate_side_qc": "pass" if pass_qc else "fail",
        "propionate_side_signed_distances_A": {
            "SG": sg_d,
            "CGA": cga_d,
            "CGD": cgd_d,
            "propionate_midpoint": prop_mid_d,
        },
        "propionate_side_same_side_products": same_side_products,
        "propionate_side_tolerance_A2": tolerance,
        "propionate_side_rule": "SG must be on the same heme-plane side as CGA and CGD.",
    })
    if not pass_qc:
        raise ValueError(
            "Mapped heme propionate-side QC failed: proximal SG is not on the "
            "same heme-plane side as CGA/CGD."
        )
    return diag


def build_source_frame(
    atoms: Sequence[AtomRecord],
    proximal_anchor: Optional[np.ndarray] = None,
) -> Tuple[LocalFrame, Dict[str, object]]:
    by_name = {a.name: a for a in atoms}
    if "FE" not in by_name:
        raise ValueError("Source heme does not contain FE.")
    fe = by_name["FE"].coord
    core = [a for a in atoms if a.name in CORE_RING_NAMES]
    if len(core) < 8:
        core = [a for a in atoms if a.element != "O" and np.linalg.norm(a.coord - fe) <= 4.5]
    if len(core) < 6:
        raise ValueError("Not enough core atoms to fit heme macrocycle plane.")
    _, z_axis = fit_plane([a.coord for a in core])
    z_source = "plane-only"
    if proximal_anchor is not None:
        # The source-frame +z axis must point to the distal side so that
        # template distal oxygens (O1/O2) are mapped away from the proximal
        # thiolate. The previous implementation oriented +z toward the proximal
        # anchor, which inverted CPDI/DIOXY oxygen placement onto the SG side.
        if np.dot(z_axis, proximal_anchor - fe) > 0:
            z_axis = -z_axis
        z_source = "proximal-anchor-distal-oriented"
    groups = detect_carboxyl_carbons(atoms)
    ca, cd = pick_propionate_pair(fe, groups)
    midpoint = (ca.coord + cd.coord) / 2.0
    x_axis = project_to_plane(midpoint - fe, z_axis)
    if np.linalg.norm(x_axis) < 1e-6:
        x_axis = project_to_plane(ca.coord - fe, z_axis)
    x_axis = normalize(x_axis)
    y_axis = normalize(np.cross(z_axis, x_axis))
    x_axis = normalize(np.cross(y_axis, z_axis))
    diag = {
        "core_atom_names": [a.name for a in core],
        "carboxyl_candidates": [g[0].name for g in groups],
        "selected_carboxyl_carbons": [ca.name, cd.name],
        "z_axis_source": z_source,
    }
    return LocalFrame(origin=fe, x_axis=x_axis, y_axis=y_axis, z_axis=z_axis), diag


def rewrite_all_template_coordinates(
    source_frame: LocalFrame,
    template_frame: LocalFrame,
    template_atoms: Sequence[AtomRecord],
) -> Dict[str, np.ndarray]:
    return {
        atom.name: source_frame.to_global(template_frame.to_local(np.asarray(atom.coord, dtype=float)))
        for atom in template_atoms
    }


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def map_heme_template(
    pdb_path: str,
    resname: str = "HEM",
    chain_id: Optional[str] = None,
    proximal_anchor: Optional[Sequence[float]] = None,
    anchor_pdb_path: Optional[str] = None,
    anchor_chain: Optional[str] = None,
    axial_cys_resid: Optional[int] = None,
    template_mol2_path: Optional[str] = None,
    heme_state: Optional[str] = None,          # None = auto-detect
) -> MappingResultV2:
    """
    Map heme atoms from *pdb_path* onto the IC6/CPDI/DIOXY standard template
    and rewrite all coordinates using the LocalFrame transform.

    Parameters
    ----------
    heme_state : str or None
        If None, the state is auto-detected from the PDB coordinates.
        Pass "IC6", "CPDI", or "DIOXY" to override (manual research mode).
    """
    raw_atoms = parse_pdb_heme_atoms(pdb_path, resname=resname, chain_id=chain_id)
    if not raw_atoms:
        raise ValueError(f"No {resname} atoms found in {pdb_path}.")

    source_atoms, duplicates = dedupe_heme_atoms(raw_atoms)

    # 鈹€鈹€ State detection / override 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    # Parse all atoms for cross-residue distal O detection (e.g. OXY residue)
    all_atoms = parse_pdb_atoms(pdb_path)
    state_result = detect_heme_state(source_atoms, manual_state=heme_state,
                                     all_atoms=all_atoms)
    if state_result.warning:
        import warnings
        warnings.warn(state_result.warning, UserWarning, stacklevel=2)

    # 鈹€鈹€ Template loading 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    template_path = resolve_template_mol2_path(state_result.state, template_mol2_path)
    template_atoms = load_template_atoms_from_mol2(template_path)
    template_frame = build_template_frame_from_atoms(template_atoms)

    # 鈹€鈹€ Proximal anchor (Cys SG) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    proximal_vec = np.asarray(proximal_anchor, dtype=float) if proximal_anchor is not None else None
    proximal_record: Optional[AtomRecord] = None
    if proximal_vec is None:
        fe_coord = next(a.coord for a in source_atoms if a.name == "FE")
        proximal_record = find_proximal_anchor(
            anchor_pdb_path or pdb_path,
            fe_coord,
            anchor_chain=anchor_chain,
            axial_cys_resid=axial_cys_resid,
        )
        if proximal_record is not None:
            proximal_vec = proximal_record.coord

    source_frame, frame_diag = build_source_frame(source_atoms, proximal_vec)

    # 鈹€鈹€ Atom matching + coordinate rewrite 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    template_names = {a.name for a in template_atoms}
    direct_matches = {a.name: a.name for a in source_atoms if a.name in template_names}
    unmatched_src = [a for a in source_atoms if a.name not in direct_matches]
    unmatched_tpl = [a for a in template_atoms if a.name not in direct_matches.values()]
    carboxyl_lookup = build_source_carboxyl_lookup(unmatched_src, source_frame)
    src_labels = {a.name: source_class(a, carboxyl_lookup, source_frame) for a in unmatched_src}
    slot_assignments = assign_source_atoms_to_template_slots(
        unmatched_src, unmatched_tpl, source_frame, template_frame, src_labels
    )
    completed = rewrite_all_template_coordinates(source_frame, template_frame, template_atoms)

    diagnostics: Dict[str, object] = {
        "algorithm_version": ALGORITHM_VERSION,
        "algorithm_mode": "template-rewrite-geometry",
        "heme_state": state_result.state,
        "state_detection_method": state_result.method,
        "state_distal_o_count": state_result.distal_o_count,
        "state_distal_o_distances_A": state_result.distal_o_distances,
        "state_detection_warning": state_result.warning,
        "template_mol2_path": str(template_path),
        "raw_atom_count": len(raw_atoms),
        "deduped_atom_count": len(source_atoms),
        "duplicate_input_atom_names": sorted(duplicates.keys()),
        "direct_match_count": len(direct_matches),
        "slot_assign_count": len(slot_assignments),
        "template_geometry_rewrite_count": len(template_atoms),
        "proximal_anchor_found": proximal_record is not None or proximal_anchor is not None,
        "proximal_anchor_atom": (
            {"name": proximal_record.name, "resname": proximal_record.resname,
             "chain": proximal_record.chain, "resid": proximal_record.resid}
            if proximal_record is not None else None
        ),
    }
    diagnostics.update(frame_diag)
    diagnostics.update(mapped_propionate_side_diagnostics(completed, proximal_vec))

    return MappingResultV2(
        source_atoms=source_atoms,
        template_atoms=template_atoms,
        source_frame=source_frame,
        template_frame=template_frame,
        direct_matches=direct_matches,
        slot_assignments=slot_assignments,
        completed_heavy_atoms=completed,
        heme_state=state_result.state,
        diagnostics=diagnostics,
    )


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _pdb_atom_line(serial, atom_name, resname, chain, resid, coord, element) -> str:
    return (
        f"HETATM{serial:5d} {atom_name:>4s} {resname:>3s} {chain:1s}{resid:4d}    "
        f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}"
        f"{1.00:6.2f}{0.00:6.2f}          {element:>2s}\n"
    )


def write_completed_heme_pdb(result: MappingResultV2, output_path: str, resname: str = "HEM") -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exemplar = result.source_atoms[0]
    chain, resid = exemplar.chain or "A", exemplar.resid or 1
    lines = []
    for serial, atom in enumerate(result.template_atoms, start=1):
        coord = result.completed_heavy_atoms[atom.name]
        lines.append(_pdb_atom_line(serial, atom.name, resname, chain, resid, coord, infer_element(atom.name)))
    lines.append("END\n")
    path.write_text("".join(lines), encoding="utf-8")


def _json_ready(result: MappingResultV2) -> Dict[str, object]:
    return {
        "heme_state": result.heme_state,
        "direct_matches": result.direct_matches,
        "slot_assignments": result.slot_assignments,
        "diagnostics": result.diagnostics,
        "completed_heavy_atoms": {k: [float(x) for x in v.tolist()] for k, v in result.completed_heavy_atoms.items()},
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "CYPForge heme mapping: multi-state CYP450 heme geometry template-rewrite. "
            "Auto-detects IC6 / CPDI / DIOXY from PDB coordinates; "
            "supports manual override for spin-state research."
        )
    )
    parser.add_argument("pdb_path", help="Input PDB file.")
    parser.add_argument("--resname", default="HEM", help="Heme residue name (default: HEM).")
    parser.add_argument("--chain", default=None, help="Chain selector.")
    parser.add_argument("--proximal-anchor", nargs=3, type=float, default=None, metavar=("X", "Y", "Z"),
                        help="Proximal Cys SG coordinate (optional).")
    parser.add_argument("--anchor-pdb", default=None, help="Full PDB for proximal Cys auto-detection.")
    parser.add_argument("--anchor-chain", default=None)
    parser.add_argument("--axial-cys-resid", type=int, default=None)
    parser.add_argument("--template-mol2", default=None,
                        help="Custom mol2 template path (overrides state-based selection).")
    parser.add_argument(
        "--heme-state",
        choices=list(VALID_HEME_STATES),
        default=None,
        help=(
            "Manually specify heme iron state instead of auto-detecting. "
            "IC6=Fe(III) high-spin penta-coord (resting); "
            "CPDI=Compound I Fe(IV)=O; "
            "DIOXY=Ferrous-oxy Fe(II)-O2."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    parser.add_argument("--write-pdb", default=None, help="Write completed heme PDB to this path.")
    parser.add_argument("--write-json", default=None, help="Write JSON report to this path.")
    args = parser.parse_args(argv)

    result = map_heme_template(
        args.pdb_path,
        resname=args.resname,
        chain_id=args.chain,
        proximal_anchor=args.proximal_anchor,
        anchor_pdb_path=args.anchor_pdb,
        anchor_chain=args.anchor_chain,
        axial_cys_resid=args.axial_cys_resid,
        template_mol2_path=args.template_mol2,
        heme_state=args.heme_state,
    )

    if args.write_pdb:
        write_completed_heme_pdb(result, args.write_pdb, resname=args.resname)
    if args.write_json:
        Path(args.write_json).write_text(
            json.dumps(_json_ready(result), indent=2, ensure_ascii=False), encoding="utf-8"
        )
    if args.json:
        sys.stdout.buffer.write(
            json.dumps(_json_ready(result), indent=2, ensure_ascii=True).encode("ascii")
        )
        sys.stdout.buffer.write(b"\n")
    else:
        print(f"Detected heme state : {result.heme_state}")
        print(f"Direct matches      : {len(result.direct_matches)}")
        print(f"Slot assignments    : {len(result.slot_assignments)}")
        if result.diagnostics.get("state_detection_warning"):
            warn = result.diagnostics["state_detection_warning"]
            print(f"WARNING: {warn}".encode("ascii", errors="replace").decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
