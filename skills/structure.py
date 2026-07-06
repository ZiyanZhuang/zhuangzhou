"""Structure analysis skills — PLIP, cofactor detection, mutation classification."""

import os
import sys

from .registry import Skill, SkillParam, SkillRegistry


def plip_analysis(complex_pdb: str = '', output_dir: str = None, config: dict = None,
                  pdb_path: str = '',   # alias for complex_pdb
                  **kw) -> dict:
    complex_pdb = complex_pdb or pdb_path
    if not complex_pdb:
        return {'error': 'complex_pdb (or pdb_path) is required',
                'hint': 'Pass the protein-ligand complex PDB file path'}
    from zhuangzhou.tools import run_plip_analysis
    return run_plip_analysis(complex_pdb, output_dir, config)


def cofactor_detection(pdb_path: str, output_dir: str = None, config: dict = None, **kw) -> dict:
    from zhuangzhou.tools import detect_cofactors
    return detect_cofactors(pdb_path, output_dir, config)


def mutation_classification(pdb_path: str, mutations: list,
                            binding_site: set = None, config: dict = None, **kw) -> dict:
    """Classify mutations by structural role. Returns dict (classifications key = list)."""
    from zhuangzhou.tools import classify_mutations
    result = classify_mutations(pdb_path, mutations, binding_site, config)
    # classify_mutations returns a list; wrap as dict so agent receives consistent format
    if isinstance(result, list):
        return {'classifications': result, 'n_classified': len(result)}
    return result  # already a dict (error case)


def _parse_cif_sequences(cif_path: str, aa3to1: dict,
                         target_chains: set = None,
                         source: str = 'seqres') -> tuple:
    """Parse mmCIF (.cif) file to extract protein sequences.

    Returns (sequences_dict, source_used_str).
    """
    import re
    sequences = {}

    # Strategy: parse _entity_poly_seq for SEQRES-equivalent,
    # or _atom_site for ATOM-equivalent.
    entity_poly = {}  # entity_id -> sequence
    entity_chain_map = {}  # entity_id -> [chain_ids]
    atom_seq = {}  # chain -> {resnum: one_letter}

    with open(cif_path, encoding='utf-8', errors='replace') as fh:
        content = fh.read()

    # --- Parse _entity_poly.pdbx_seq_one_letter_code (best source) ---
    # This gives the full sequence like SEQRES
    poly_match = re.findall(
        r'_entity_poly\.entity_id\s+(\d+).*?'
        r'_entity_poly\.pdbx_seq_one_letter_code\s*\n?;?\s*([A-Za-z\s\n]+)',
        content, re.DOTALL
    )
    # Alternative: look for loop_ format
    if not poly_match:
        # Try loop format: _entity_poly table
        in_entity_poly = False
        entity_poly_cols = []
        for line in content.split('\n'):
            stripped = line.strip()
            if stripped == '_entity_poly.entity_id':
                in_entity_poly = True
                entity_poly_cols = ['entity_id']
                continue
            if in_entity_poly and stripped.startswith('_entity_poly.'):
                entity_poly_cols.append(stripped.split('.')[1])
                continue
            if in_entity_poly and not stripped.startswith('_') and stripped and stripped != '#':
                # Data line
                if stripped.startswith(';'):
                    continue  # multi-line value, skip for now
                parts = stripped.split()
                if len(parts) >= len(entity_poly_cols):
                    eid_idx = entity_poly_cols.index('entity_id') if 'entity_id' in entity_poly_cols else 0
                    entity_id = parts[eid_idx]
                    if 'pdbx_strand_id' in entity_poly_cols:
                        sid_idx = entity_poly_cols.index('pdbx_strand_id')
                        chain_ids = parts[sid_idx].split(',')
                        entity_chain_map[entity_id] = [c.strip() for c in chain_ids]
                continue
            if in_entity_poly and (stripped.startswith('_') or stripped == '#' or stripped.startswith('loop_')):
                in_entity_poly = False

    # --- Parse _entity_poly_seq (residue-by-residue sequence) ---
    # More reliable for extracting sequence from CIF
    in_poly_seq = False
    poly_seq_cols = []
    entity_residues = {}  # entity_id -> [(num, mon_id)]
    for line in content.split('\n'):
        stripped = line.strip()
        if stripped == '_entity_poly_seq.entity_id':
            in_poly_seq = True
            poly_seq_cols = ['entity_id']
            continue
        if in_poly_seq and stripped.startswith('_entity_poly_seq.'):
            poly_seq_cols.append(stripped.split('.')[1])
            continue
        if in_poly_seq and not stripped.startswith('_') and stripped and stripped != '#':
            parts = stripped.split()
            if len(parts) >= len(poly_seq_cols):
                eid_idx = poly_seq_cols.index('entity_id') if 'entity_id' in poly_seq_cols else 0
                num_idx = poly_seq_cols.index('num') if 'num' in poly_seq_cols else 1
                mon_idx = poly_seq_cols.index('mon_id') if 'mon_id' in poly_seq_cols else 2
                eid = parts[eid_idx]
                try:
                    num = int(parts[num_idx])
                except (ValueError, IndexError):
                    continue
                mon = parts[mon_idx].upper() if mon_idx < len(parts) else ''
                entity_residues.setdefault(eid, []).append((num, mon))
            continue
        if in_poly_seq and (stripped.startswith('_') or stripped == '#' or stripped.startswith('loop_')):
            in_poly_seq = False

    # Build sequences from _entity_poly_seq
    if entity_residues:
        for eid, residues in entity_residues.items():
            residues.sort(key=lambda x: x[0])
            seq = ''.join(aa3to1.get(mon, 'X') for _, mon in residues)
            # Filter out non-protein (all X)
            if seq.replace('X', ''):
                chains_for_entity = entity_chain_map.get(eid, [eid])
                for ch in chains_for_entity:
                    if target_chains and ch.upper() not in target_chains:
                        continue
                    sequences[ch] = seq
        if sequences:
            return sequences, 'cif_entity_poly_seq'

    # --- Fallback: parse _atom_site for CA atoms ---
    in_atom_site = False
    atom_cols = []
    for line in content.split('\n'):
        stripped = line.strip()
        if stripped == '_atom_site.group_PDB':
            in_atom_site = True
            atom_cols = ['group_PDB']
            continue
        if in_atom_site and stripped.startswith('_atom_site.'):
            atom_cols.append(stripped.split('.')[1])
            continue
        if in_atom_site and not stripped.startswith('_') and stripped and stripped != '#':
            parts = stripped.split()
            if len(parts) < len(atom_cols):
                continue
            col_map = {c: parts[i] if i < len(parts) else '' for i, c in enumerate(atom_cols)}
            group = col_map.get('group_PDB', '')
            if group not in ('ATOM', 'HETATM'):
                continue
            atom_name = col_map.get('label_atom_id', '')
            if atom_name != 'CA':
                continue
            res_name = col_map.get('label_comp_id', '').upper()
            if res_name not in aa3to1:
                continue
            chain = col_map.get('label_asym_id', col_map.get('auth_asym_id', ''))
            if target_chains and chain.upper() not in target_chains:
                continue
            try:
                resnum = int(col_map.get('label_seq_id', col_map.get('auth_seq_id', '0')))
            except ValueError:
                continue
            atom_seq.setdefault(chain, {})[resnum] = aa3to1[res_name]
            continue
        if in_atom_site and (stripped.startswith('_') or stripped == '#' or stripped.startswith('loop_')):
            in_atom_site = False

    for chain, resmap in sorted(atom_seq.items()):
        seq = ''.join(resmap[k] for k in sorted(resmap))
        if seq:
            sequences[chain] = seq

    return sequences, 'cif_atom_site'


def pdb_to_fasta(pdb_path: str, output_path: str = None,
                 chains: str = None, source: str = 'seqres', **kw) -> dict:
    """Extract protein sequence from a PDB file and save as FASTA.

    Args:
        pdb_path:    Path to the input PDB file.
        output_path: Output .fasta path. Defaults to <stem>.fasta in same dir.
        chains:      Comma-separated chain IDs to extract, e.g. 'A,B'.
                     None = all protein chains.
        source:      'seqres' (prefer SEQRES records, fall back to ATOM/CA) or
                     'atom' (always use ATOM/CA coordinate records).

    Returns:
        {'fasta_path': str, 'sequences': {chain: seq}, 'n_chains': int,
         'n_residues': int, 'source': str}
    """
    # Standard 3-letter → 1-letter amino acid mapping
    AA3TO1 = {
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
        'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
        'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
        # Non-standard aliases
        'MSE': 'M', 'SEC': 'U', 'PYL': 'O',
        'HSD': 'H', 'HSE': 'H', 'HSP': 'H',  # CHARMM HIS variants
        'CYX': 'C', 'CYM': 'C',               # disulfide / deprotonated Cys
        'GLH': 'E', 'ASH': 'D',               # protonated Glu/Asp
        'LYN': 'K', 'ARN': 'R',
    }

    try:
        if not os.path.isfile(pdb_path):
            return {'error': f'PDB file not found: {pdb_path}',
                    'hint': 'Check the file path'}

        target_chains = None
        if chains:
            target_chains = {c.strip().upper() for c in chains.split(',')}

        is_cif = pdb_path.lower().endswith('.cif')

        seqres: dict = {}     # chain -> list of 3-letter codes
        atom_seq: dict = {}   # chain -> {resnum: one-letter}

        if is_cif:
            # ── Parse mmCIF format ───────────────────────────────────────────
            sequences, used_source = _parse_cif_sequences(pdb_path, AA3TO1, target_chains, source)
        else:
            # ── Try SEQRES records first (PDB format) ────────────────────────
            with open(pdb_path, encoding='utf-8', errors='replace') as fh:
                for line in fh:
                    rec = line[:6].strip()

                    if rec == 'SEQRES':
                        chain = line[11].strip()
                        if not chain:
                            continue
                        if target_chains and chain not in target_chains:
                            continue
                        residues = line[19:].split()
                        seqres.setdefault(chain, []).extend(residues)

                    elif rec in ('ATOM', 'HETATM'):
                        if len(line) < 27:
                            continue
                        res_name = line[17:20].strip()
                        chain = line[21].strip()
                        if not chain:
                            continue
                        if target_chains and chain not in target_chains:
                            continue
                        atom_name = line[12:16].strip()
                        if atom_name != 'CA':
                            continue
                        if res_name not in AA3TO1:
                            continue
                        try:
                            resnum = int(line[22:26].strip())
                        except ValueError:
                            continue
                        atom_seq.setdefault(chain, {})[resnum] = AA3TO1[res_name]

            # ── Build sequences per chain ────────────────────────────────────
            use_seqres = (source == 'seqres') and bool(seqres)
            sequences = {}
            used_source = 'seqres' if use_seqres else 'atom'

            if use_seqres:
                for chain, residues in sorted(seqres.items()):
                    seq = ''.join(AA3TO1.get(r.upper(), 'X') for r in residues)
                    if seq:
                        sequences[chain] = seq
            else:
                for chain, resmap in sorted(atom_seq.items()):
                    seq = ''.join(resmap[k] for k in sorted(resmap))
                    if seq:
                        sequences[chain] = seq

        if not sequences:
            return {'error': 'No protein sequences found in PDB',
                    'hint': 'For CIF files, ensure _atom_site records contain protein atoms. '
                            'Check chains argument or verify file contains ATOM records'}

        # ── Write FASTA ──────────────────────────────────────────────────────
        if not output_path:
            stem = os.path.splitext(os.path.basename(pdb_path))[0]
            output_path = os.path.join(
                os.path.dirname(pdb_path) or os.getcwd(),
                f'{stem}.fasta')

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        stem = os.path.splitext(os.path.basename(pdb_path))[0]
        with open(output_path, 'w', encoding='utf-8') as fh:
            for chain, seq in sequences.items():
                fh.write(f'>{stem}_chain_{chain}\n')
                # Wrap at 70 chars (standard FASTA)
                for i in range(0, len(seq), 70):
                    fh.write(seq[i:i + 70] + '\n')

        total_res = sum(len(s) for s in sequences.values())
        return {
            'fasta_path': output_path,
            'sequences': sequences,
            'n_chains': len(sequences),
            'n_residues': total_res,
            'source': used_source,
        }

    except Exception as e:
        return {'error': str(e), 'hint': 'Check PDB format and file encoding'}


def _win_to_wsl(path: str) -> str:
    """Convert a Windows path to WSL /mnt/X/... path."""
    import re
    p = path.replace('\\', '/')
    return re.sub(r'^([A-Za-z]):', lambda m: f'/mnt/{m.group(1).lower()}', p)


def pocket_detection(pdb_path: str, output_dir: str = None, config: dict = None, **kw) -> dict:
    """Run fpocket pocket detection via WSLSandbox.

    fpocket is passed the WSL-translated path so it never sees Windows paths,
    which avoids the case-mangling issue where 'WT.pdb' becomes 'wt.pdb' inside WSL.
    """
    import re

    # Translate Windows path → WSL path for the fpocket command
    pdb_wsl = _win_to_wsl(pdb_path) if pdb_path else pdb_path

    try:
        from zhuangzhou.envs.sandbox import WSLSandbox
        sandbox = WSLSandbox()
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        result = sandbox.run(f'fpocket -f "{pdb_wsl}"', timeout=300)
        return {
            'success': result.returncode == 0,
            'stdout': result.stdout or '',
            'stderr': result.stderr or '',
        }
    except ImportError:
        # WSLSandbox unavailable — try direct subprocess as last resort
        import subprocess
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        result = subprocess.run(
            ['wsl', 'fpocket', '-f', pdb_wsl],
            capture_output=True, text=True, timeout=300, errors='replace',
        )
        return {'success': result.returncode == 0,
                'stdout': result.stdout, 'stderr': result.stderr}


def enrich_mutations(pdb_path: str, mutations: list,
                     uniprot_id: str = None, active_site: list = None,
                     catalytic: list = None, protein_name: str = '',
                     pfam_id: str = None, run_literature: bool = True,
                     sequence: str = '', fasta_header: str = '',
                     config: dict = None, **kw) -> dict:
    """Enrich a mutation list with structural + literature annotations.

    Runs DSSP (secondary structure + RSA), InterPro/UniProt domain annotation,
    active-site distance, channel detection, and optionally PubMed / UniProt
    natural-variant lookup.

    For fusion proteins without UniProt ID, pass ``sequence`` and ``fasta_header``
    to enable sequence-based domain detection (linker scanning + InterProScan).

    Returns {'mutations': enriched_list, 'n_enriched': int, 'dssp_ok': bool,
             'domain_ok': bool, 'literature_ok': bool}
    """
    try:
        from zhuangzhou.mutation_enricher import enrich_mutations as _enrich
        import re as _re
        _MRE = _re.compile(r'^([A-Za-z*])(\d+)([A-Za-z*-])$')

        def _norm(m):
            if isinstance(m, dict):
                return m
            s = str(m).strip()
            hit = _MRE.match(s)
            if hit:
                return {'mutation': s, 'position': int(hit.group(2)),
                        'wt': hit.group(1).upper(), 'mut': hit.group(3).upper()}
            return {'mutation': s}

        mutations = [_norm(m) for m in mutations]
        active_set  = set(active_site)  if active_site  else None
        catalytic_set = set(catalytic) if catalytic else None
        cache_dir = (config or {}).get('cache_dir', None)
        enriched = _enrich(
            mutations     = mutations,
            pdb_path      = pdb_path,
            uniprot_id    = uniprot_id,
            active_site   = active_set,
            catalytic     = catalytic_set,
            protein_name  = protein_name,
            pfam_id       = pfam_id,
            run_literature= run_literature,
            cache_dir     = cache_dir,
            sequence      = sequence or kw.get('sequence', ''),
            fasta_header  = fasta_header or kw.get('fasta_header', ''),
        )
        has_ss   = bool(enriched and enriched[0].get('ss_type'))
        has_dom  = bool(enriched and enriched[0].get('domain') not in (None, 'Unknown'))
        has_lit  = bool(enriched and enriched[0].get('evidence_level') is not None)
        return {
            'mutations':     enriched,
            'n_enriched':    len(enriched),
            'dssp_ok':       has_ss,
            'domain_ok':     has_dom,
            'literature_ok': has_lit,
        }
    except Exception as exc:
        return {'error': str(exc), 'hint': 'Check pdb_path and dependencies',
                'mutations': mutations}


def register_skills(registry: SkillRegistry):
    registry.register(Skill(
        name='plip_analysis',
        description='Analyze protein-ligand interactions (H-bonds, salt bridges, hydrophobic, pi-stacking)',
        category='structure',
        fn=plip_analysis,
        inputs=[
            SkillParam('complex_pdb', 'path', 'Protein-ligand complex PDB'),
            SkillParam('output_dir', 'path', 'Output directory', required=False, is_output_path=True),
        ],
        outputs=[SkillParam('interactions', 'dict', 'Interaction summary')],
        source='scripts/plip_analysis.py',
        output_type='analysis',
    ))

    registry.register(Skill(
        name='cofactor_detection',
        description='Detect cofactors and metal ions in PDB',
        category='structure',
        fn=cofactor_detection,
        inputs=[SkillParam('pdb_path', 'path', 'Input PDB file')],
        outputs=[SkillParam('cofactors', 'dict', 'Cofactor list + coordination residues')],
        source='scripts/cofactor_detector.py',
    ))

    registry.register(Skill(
        name='mutation_classification',
        description='Classify mutations by structural role (active_site, buried_core, surface, etc.)',
        category='structure',
        fn=mutation_classification,
        inputs=[
            SkillParam('pdb_path', 'path', 'Input PDB'),
            SkillParam('mutations', 'list', 'Mutation strings'),
            SkillParam('binding_site', 'set', 'Active site residue numbers', required=False),
        ],
        outputs=[SkillParam('classifications', 'list', 'Classified mutations')],
        source='scripts/mutation_classifier.py',
    ))

    registry.register(Skill(
        name='pocket_detection',
        description='Detect binding pockets with fpocket (WSL)',
        category='structure',
        fn=pocket_detection,
        requires_wsl=True,
        inputs=[SkillParam('pdb_path', 'path', 'Input PDB')],
        outputs=[SkillParam('pockets', 'dict', 'Detected pockets')],
        source='fpocket (WSL)',
    ))

    registry.register(Skill(
        name='pdb_to_fasta',
        description='Extract protein sequence from PDB file and save as FASTA. '
                    'Reads SEQRES records (preferred) or ATOM/CA coordinates. '
                    'Supports multi-chain PDBs. Required before ESM-2 mutation scan.',
        category='structure',
        fn=pdb_to_fasta,
        inputs=[
            SkillParam('pdb_path', 'path', 'Input PDB file'),
            SkillParam('output_path', 'path', 'Output .fasta path', required=False, is_output_path=True),
            SkillParam('chains', 'str', 'Chain IDs to extract, comma-separated (default: all)', required=False),
            SkillParam('source', 'str', "'seqres' (default) or 'atom'", required=False),
        ],
        outputs=[
            SkillParam('fasta_path', 'path', 'Output FASTA file'),
            SkillParam('sequences', 'dict', 'Chain → sequence mapping'),
            SkillParam('n_residues', 'int', 'Total residue count'),
        ],
        output_type='sequence',
    ))

    registry.register(Skill(
        name='enrich_mutations',
        description=(
            'Enrich ESM-2 mutation list with structural and literature annotations. '
            'Adds: secondary structure (DSSP α/β/loop), real RSA, structural zone '
            '(buried/active_site/channel/surface), distance to active site, '
            'Pfam domain label, UniProt functional site, and optional '
            'UniProt natural-variant + PubMed literature evidence. '
            'For fusion proteins without UniProt ID, pass sequence and fasta_header '
            'to enable sequence-based domain detection (linker scanning + InterProScan). '
            'Run after esm2_scan, before build_mutation_report.'
        ),
        category='structure',
        fn=enrich_mutations,
        inputs=[
            SkillParam('pdb_path',      'path',   'Input PDB structure file'),
            SkillParam('mutations',     'list',   'Mutation dicts from ESM-2 scan'),
            SkillParam('uniprot_id',    'str',    'UniProt accession (e.g. Q8ZKF6)',   required=False),
            SkillParam('active_site',   'list',   'Active site residue numbers',       required=False),
            SkillParam('catalytic',     'list',   'Catalytic residue numbers',         required=False),
            SkillParam('protein_name',  'str',    'Protein name for PubMed queries',   required=False),
            SkillParam('pfam_id',       'str',    'Pfam domain ID (e.g. PF01872)',     required=False),
            SkillParam('run_literature','bool',   'Run PubMed/UniProt variant lookup', required=False),
            SkillParam('sequence',      'str',    'Full protein sequence (for fusion protein domain detection when no UniProt ID)', required=False),
            SkillParam('fasta_header',  'str',    'FASTA header line (helps parse fusion protein component names)', required=False),
        ],
        outputs=[
            SkillParam('mutations',     'list',   'Enriched mutation dicts'),
            SkillParam('n_enriched',    'int',    'Number of enriched mutations'),
        ],
        output_type='analysis',
    ))
