# CYP P450 HEME Bonded Model — Solution Plan

## Problem Statement

10ns MD simulation of CYP P450cam (2CPP) with CHARMM27 showed:
- Fe-SG distance drifted to 4.1 A mean (should be 2.32 A)
- HEME center-of-mass displaced 6.7 A
- Root cause: NO covalent Fe-S bond in topology

## Root Cause Chain

```
1. specbond.dat didn't fire during pdb2gmx
   (despite Fe-SG = 2.198 A, within 2.5 A cutoff)

2. CYS357 kept HG1 proton, stayed as CYS (not CYS2)

3. No Fe-SG covalent bond created in topol.top
   - Fe bonds: only Fe-NA, Fe-NB, Fe-NC, Fe-ND (porphyrin)
   - Missing: Fe-SG (axial thiolate)

4. Distance restraints (fc=1000 kJ/mol/nm^2) are 200x weaker
   than proper bonded term (k=209200 kJ/mol/nm^2)

5. Fe-S distance drifts → HEME COM drifts 6.7 A in 10ns
```

### Additional Finding: CYS2 Atom Type Incompatibility

Even IF specbond.dat fired:
- CYS2 in CHARMM27 = disulfide cysteine (SG atom type = SM)
- Fe-S bond requires atom type SS (from thioether/heme context)
- SM-FE bond parameters DON'T EXIST in ffbonded.itp
- SS-FE bond parameters: r0=0.232nm, k=209200 kJ/mol/nm^2
- This is likely a bug in the GROMACS CHARMM27 port

## Solution: Post-pdb2gmx Topology Surgery

### Implemented in `gromacs_md/heme_topology.py`

The fix operates AFTER pdb2gmx, which is the most robust approach
(doesn't depend on specbond.dat behavior, works across GROMACS versions).

### Algorithm

```
fix_heme_coordination(work_dir, coordination):
  1. Parse protein.gro → find atom indices for Fe, SG, CB, HG1, NA-ND
  2. Parse topol.top → check if Fe-SG bond already exists

  3. IF Fe-SG bond MISSING:
     a. Remove HG1 from CYS (topology + GRO)
        - Redistribute charge: SG (-0.23 + 0.16 = -0.07)
        - Renumber all atom indices > HG1 by -1
        - Remove HG1-referencing bonds/angles/dihedrals/pairs

     b. Inject explicit Fe-SG bond (no atom type dependency):
        [ bonds ]
        SG_idx  FE_idx  1  0.2320  209200.0  ; SS-FE params

     c. Inject coordination angles:
        [ angles ]
        CB_idx  SG_idx  FE_idx  1  100.6  418.4   ; CS-SS-FE
        SG_idx  FE_idx  NA_idx  1   90.0  836.8   ; SS-FE-NPH
        SG_idx  FE_idx  NB_idx  1   90.0  836.8
        SG_idx  FE_idx  NC_idx  1   90.0  836.8
        SG_idx  FE_idx  ND_idx  1   90.0  836.8

  4. IF specbond fired (CYS2 in topology):
     Fix SG atom type: SM -> SS

  5. Validate coordination geometry
```

### Why This Approach

| Approach | Pros | Cons |
|----------|------|------|
| Fix specbond.dat | Uses GROMACS native mechanism | Still has CYS2 SM type bug; unreliable |
| Custom FF residue | Clean, reusable | Requires FF modification, not portable |
| MCPB.py (AMBER) | Gold standard | Wrong FF ecosystem (AMBER not CHARMM) |
| **Topology surgery** | **Portable, explicit, correct** | **Slightly complex code** |

Key advantages of topology surgery:
- Uses explicit bond parameters (type 1 with r0 and k) — no atom type lookup
- Works regardless of specbond.dat behavior
- No force field files need modification
- Parameters are identical to CHARMM27's SS-FE terms

### CHARMM27 Parameters Used

From `charmm27.ff/ffbonded.itp`:

```
; Bonds
SS  FE    1  0.2320  209200.0     ; Fe-S covalent bond

; Angles
CS  SS  FE    5  100.6   418.4   ; C-S-Fe (CB-SG-Fe)
SS  FE  NPH   5   90.0   836.8   ; S-Fe-N (SG-Fe-porphyrin_N)

; Dihedrals
X   FE  SS  X   9  0.00  0.0  4  ; Fe-S torsion (zero barrier)
```

### Charge Balance

```
CYS (protonated):  SG = -0.23, HG1 = +0.16, subtotal = -0.07
CYS2 (disulfide):  SG = -0.08              , subtotal = -0.08
Our fix:           SG = -0.07 (= -0.23 + 0.16), no HG1

Total charge change: 0.00 (HG1 removed, charge absorbed by SG)
```

## Integration

### pipeline.py Changes

```python
# After pdb2gmx (Stage 2), BEFORE solvation:
heme_fix = fix_heme_coordination(work_dir, prep['coordination'])
# → Removes HG1, adds Fe-SG bond + angles
# → Also keeps distance restraints as safety net (backup)
```

### Expected Results

With bonded model:
- Fe-SG distance: stable at 2.32 ± 0.1 A (vs 4.1 A drift before)
- HEME COM displacement: < 1.0 A (vs 6.7 A before)
- RMSD of heme pocket: < 2.0 A
- Coordination geometry: preserved throughout simulation

## Validation Criteria

1. **Fe-SG distance**: mean ≤ 2.5 A, max ≤ 3.0 A
2. **HEME COM drift**: ≤ 2.0 A relative to protein backbone
3. **Fe-N porphyrin distances**: all 4 stable at ~2.0 A
4. **Total energy**: no anomalous jumps
5. **RMSD**: protein backbone < 3.0 A, heme pocket < 2.0 A

## References

1. Shahrokh et al. JCTC 2012 — QM-derived CYP450 heme parameters
2. GROMACS specbond.dat documentation
3. CHARMM27 force field — Foloppe & MacKerell (2000)
4. Li & Merz (2020) — Metal ion parameters for biomolecular simulations
