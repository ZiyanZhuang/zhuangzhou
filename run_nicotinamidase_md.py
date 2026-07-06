"""Run the nicotinamidase AMBER MD example from environment-provided inputs.

Required environment variables:
  NICOTINAMIDASE_PROTEIN_PDB
  NICOTINAMIDASE_LIGAND_MOL2
  NICOTINAMIDASE_OUTPUT_DIR
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from amber_md.general_pipeline import run_protein_ligand_md


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def main():
    result = run_protein_ligand_md(
        protein_pdb=_required_env("NICOTINAMIDASE_PROTEIN_PDB"),
        ligand_path=_required_env("NICOTINAMIDASE_LIGAND_MOL2"),
        output_dir=_required_env("NICOTINAMIDASE_OUTPUT_DIR"),
        ligand_name=os.environ.get("NICOTINAMIDASE_LIGAND_NAME", "NIC"),
        charge=int(os.environ.get("NICOTINAMIDASE_CHARGE", "0")),
        spin=int(os.environ.get("NICOTINAMIDASE_SPIN", "1")),
        resp_method=os.environ.get("NICOTINAMIDASE_RESP_METHOD", "multiwfn"),
        force_field=os.environ.get("NICOTINAMIDASE_FORCE_FIELD", "ff14SB"),
        water_model=os.environ.get("NICOTINAMIDASE_WATER_MODEL", "TIP3P"),
        box_buffer=float(os.environ.get("NICOTINAMIDASE_BOX_BUFFER", "12.0")),
        prod_ns=float(os.environ.get("NICOTINAMIDASE_PROD_NS", "10.0")),
        do_docking=os.environ.get("NICOTINAMIDASE_DOCKING", "1") not in {"0", "false", "False"},
        docking_exhaustiveness=int(os.environ.get("NICOTINAMIDASE_DOCKING_EXHAUSTIVENESS", "32")),
        engine=os.environ.get("NICOTINAMIDASE_ENGINE", "pmemd.cuda"),
    )

    print("\n" + "=" * 70)
    if "error" in result:
        print(f"  ERROR: {result['error']}")
        if "traceback" in result:
            print(result["traceback"])
    else:
        print("  SUCCESS - all pre-production files generated")
        print(f"  Output: {result['output_dir']}")
        print(f"  Topology: {result['prmtop']}")
        print(f"  Coordinates: {result['inpcrd']}")
        print(f"  Run script: {result['run_script']}")
        for step in result.get("next_steps", []):
            print(f"   - {step}")
    print("=" * 70)

    return result


if __name__ == "__main__":
    main()
