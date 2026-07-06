"""
Structural Context Enricher
============================
Enriches each mutation with structural context:
  - Secondary structure type and RSA (from DSSP)
  - Distance to active site
  - Substrate channel proximity
  - Buried/surface classification
  - Hydrophobic core detection
"""

import logging
import math
from typing import Dict, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

HYDROPHOBIC_AA = set('ACFILMVWY')

# One-letter -> three-letter for annotation strings
AA_1TO3 = {
    'A': 'Ala', 'R': 'Arg', 'N': 'Asn', 'D': 'Asp', 'C': 'Cys',
    'E': 'Glu', 'Q': 'Gln', 'G': 'Gly', 'H': 'His', 'I': 'Ile',
    'L': 'Leu', 'K': 'Lys', 'M': 'Met', 'F': 'Phe', 'P': 'Pro',
    'S': 'Ser', 'T': 'Thr', 'W': 'Trp', 'Y': 'Tyr', 'V': 'Val',
}

# Zone thresholds
_ACTIVE_SITE_SHELL_DIST = 8.0    # Å
_CHANNEL_INNER_DIST     = 8.0    # Å  (must be > active site shell)
_CHANNEL_OUTER_DIST     = 12.0   # Å  (default)
_RSA_BURIED             = 0.2
_RSA_SURFACE            = 0.4
_RSA_CHANNEL            = 0.3


# ── Coordinate loading ────────────────────────────────────────────────────────


def load_ca_coords(pdb_path: str) -> Dict[int, Tuple[float, float, float]]:
    """Returns {resnum: (x, y, z)} from ATOM/CA records."""
    coords: Dict[int, Tuple[float, float, float]] = {}
    try:
        with open(pdb_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not (line.startswith('ATOM') or line.startswith('HETATM')):
                    continue
                atom_name = line[12:16].strip()
                if atom_name != 'CA':
                    continue
                try:
                    resnum = int(line[22:26].strip())
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    # Keep first occurrence (handles insertion codes)
                    if resnum not in coords:
                        coords[resnum] = (x, y, z)
                except (ValueError, IndexError) as e:
                    logger.warning('CA coord parse error at line %r: %s', line.rstrip(), e)
    except Exception as e:
        logger.warning('load_ca_coords failed for %s: %s', pdb_path, e)
    return coords


# ── Distance utilities ────────────────────────────────────────────────────────


def _dist3d(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    """Euclidean distance between two 3D points."""
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def compute_active_site_distances(
    ca_coords: Dict[int, Tuple[float, float, float]],
    active_site: Set[int],
) -> Dict[int, float]:
    """Returns {resnum: float} minimum distance to any active site residue in Å."""
    if not active_site or not ca_coords:
        return {}

    active_site_coords = [
        ca_coords[r] for r in active_site if r in ca_coords
    ]

    if not active_site_coords:
        logger.warning('No active site residues found in ca_coords')
        return {}

    distances: Dict[int, float] = {}
    for resnum, coord in ca_coords.items():
        if resnum in active_site:
            distances[resnum] = 0.0
            continue
        min_dist = min(_dist3d(coord, as_coord) for as_coord in active_site_coords)
        distances[resnum] = round(min_dist, 2)

    return distances


# ── Zone classification ───────────────────────────────────────────────────────


def classify_structural_zone(
    pos: int,
    ca_coords: Dict[int, Tuple[float, float, float]],
    dssp_data: dict,
    active_site: Set[int],
    catalytic: Set[int] = None,
) -> str:
    """
    Classify a residue into one of:
      catalytic | active_site | active_site_shell | buried_hydrophobic |
      buried_polar | surface | intermediate
    """
    if catalytic and pos in catalytic:
        return 'catalytic'

    if pos in active_site:
        return 'active_site'

    # Compute distance to active site
    active_site_coords = [ca_coords[r] for r in active_site if r in ca_coords]
    dist_to_active = None
    if pos in ca_coords and active_site_coords:
        dist_to_active = min(_dist3d(ca_coords[pos], c) for c in active_site_coords)

    if dist_to_active is not None and dist_to_active <= _ACTIVE_SITE_SHELL_DIST:
        return 'active_site_shell'

    # RSA-based classification (skip when RSA is unavailable, e.g. pydssp_no_rsa)
    rsa = None
    aa = None
    if pos in dssp_data:
        entry = dssp_data[pos]
        if entry.get('rsa_source') != 'pydssp_no_rsa':
            rsa = entry.get('rsa')
        aa = entry.get('aa', '')

    if rsa is not None:
        if rsa < _RSA_BURIED:
            if aa and aa.upper() in HYDROPHOBIC_AA:
                return 'buried_hydrophobic'
            return 'buried_polar'
        if rsa > _RSA_SURFACE:
            return 'surface'

    return 'intermediate'


# ── Channel detection ─────────────────────────────────────────────────────────


def identify_channel_residues(
    ca_coords: Dict[int, Tuple[float, float, float]],
    dssp_data: dict,
    active_site: Set[int],
    distance_threshold: float = _CHANNEL_OUTER_DIST,
    rsa_threshold: float = _RSA_CHANNEL,
) -> Set[int]:
    """Returns set of residue numbers likely lining the substrate access channel.

    Criteria: buried (RSA < rsa_threshold) AND within distance_threshold of
    active site AND NOT in the inner active site shell (> 8 Å).
    """
    if not active_site or not ca_coords:
        return set()

    active_site_coords = [ca_coords[r] for r in active_site if r in ca_coords]
    if not active_site_coords:
        return set()

    channel: Set[int] = set()
    for resnum, coord in ca_coords.items():
        if resnum in active_site:
            continue

        rsa = dssp_data.get(resnum, {}).get('rsa')
        if rsa is None or rsa >= rsa_threshold:
            continue

        dist = min(_dist3d(coord, asc) for asc in active_site_coords)
        if _CHANNEL_INNER_DIST < dist <= distance_threshold:
            channel.add(resnum)

    return channel


# ── Main enrichment function ──────────────────────────────────────────────────


def enrich_structural_context(
    mutation: dict,
    ca_coords: Dict[int, Tuple[float, float, float]],
    dssp_data: dict,
    active_site: Set[int],
    catalytic: Set[int] = None,
    channel_residues: Set[int] = None,
) -> dict:
    """
    Returns new fields to merge into the mutation dict.

    Keys added:
      ss_type, rsa, buried, zone, dist_active_site, in_channel,
      structural_annotation
    """
    pos = mutation.get('position')
    wt = mutation.get('wt', '')

    # Defaults
    ss_type = 'unknown'
    rsa_val = None
    buried = False
    zone = 'unknown'
    dist_active_site = None
    in_channel = False

    # --- Secondary structure & RSA from DSSP ---
    if dssp_data and pos in dssp_data:
        entry = dssp_data[pos]
        ss_type = entry.get('ss_label', 'coil/loop')
        rsa_val = entry.get('rsa')
        # rsa_source='pydssp_no_rsa' means RSA is 0.0 placeholder, not real
        if entry.get('rsa_source') == 'pydssp_no_rsa':
            rsa_val = None   # treat as missing so zone uses distance only
        if rsa_val is not None:
            buried = rsa_val < _RSA_BURIED
    elif not dssp_data:
        # No DSSP data at all — SS unknown; zone computed from CA coords
        ss_type = 'N/A'
        buried = False

    # --- Distance to active site ---
    if pos in ca_coords and active_site:
        active_site_coords = [ca_coords[r] for r in active_site if r in ca_coords]
        if active_site_coords:
            dist_active_site = round(
                min(_dist3d(ca_coords[pos], c) for c in active_site_coords), 2
            )

    # --- Zone classification ---
    if ca_coords and active_site:
        zone = classify_structural_zone(
            pos, ca_coords, dssp_data, active_site, catalytic
        )
    elif dssp_data and pos in dssp_data:
        # No coordinate context; classify from RSA alone
        if rsa_val is not None:
            aa = dssp_data[pos].get('aa', '')
            if rsa_val < _RSA_BURIED:
                zone = 'buried_hydrophobic' if aa.upper() in HYDROPHOBIC_AA else 'buried_polar'
            elif rsa_val > _RSA_SURFACE:
                zone = 'surface'
            else:
                zone = 'intermediate'

    # --- Channel membership ---
    if channel_residues is not None:
        in_channel = pos in channel_residues
    elif ca_coords and dssp_data and active_site:
        # Compute on the fly if channel_residues not pre-computed
        channel_residues = identify_channel_residues(ca_coords, dssp_data, active_site)
        in_channel = pos in channel_residues

    # --- Build human-readable annotation ---
    structural_annotation = _build_annotation(
        pos, wt, ss_type, zone, dist_active_site, in_channel, active_site
    )

    enriched: dict = {
        'ss_type': ss_type,
        'rsa': round(rsa_val, 4) if rsa_val is not None else 0.0,
        'buried': buried,
        'zone': zone,
        'dist_active_site': dist_active_site,
        'in_channel': in_channel,
        'structural_annotation': structural_annotation,
    }

    return enriched


# ── Annotation string builder ─────────────────────────────────────────────────


def _build_annotation(
    pos: int,
    wt: str,
    ss_type: str,
    zone: str,
    dist_active_site: Optional[float],
    in_channel: bool,
    active_site: Set[int],
) -> str:
    """Build a concise human-readable structural annotation sentence."""
    aa_name = AA_1TO3.get(wt.upper(), wt) if wt else 'residue'

    parts = [f'Position {pos} ({aa_name}) in {ss_type}']

    zone_phrases = {
        'catalytic':          'catalytic residue',
        'active_site':        'active site',
        'active_site_shell':  'active site shell',
        'buried_hydrophobic': 'buried hydrophobic core',
        'buried_polar':       'buried polar core',
        'surface':            'solvent-exposed surface',
        'intermediate':       'intermediate burial',
        'unknown':            '',
    }
    zone_phrase = zone_phrases.get(zone, zone)
    if zone_phrase:
        parts.append(zone_phrase)

    if in_channel:
        parts.append('substrate access channel')

    if dist_active_site is not None:
        # Identify the nearest active site residue label for clarity
        if active_site:
            sorted_as = sorted(active_site)
            as_label = f'C{sorted_as[0]}' if sorted_as else 'active site'
        else:
            as_label = 'active site'
        parts.append(f'{dist_active_site} A from {as_label}')

    return ', '.join(parts) + '.'
