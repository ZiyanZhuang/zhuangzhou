import os
import numpy as np
import logging
from ..utils.system import run_wsl, win_to_wsl, write_file

logger = logging.getLogger(__name__)

class HemePipeline:
    """Automates Heme parameterization and PDB preparation using Shahrokh protocol."""
    
    VALID_STATES = ("CPDI", "DIOXY", "IC6")
    
    def __init__(self, params_base=None):
        self.params_base = params_base or os.environ.get("AMBER_HEME_PARAMS", "")

    def run(self, pdb_path, output_dir, state=None):
        """Full preparation pipeline."""
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Processing PDB: {pdb_path} to {output_dir}")
        
        # 1. Parse and Identify (Simplified for logic demonstration)
        # In actual implementation, we call the parser and geometry logic
        if state is None:
            state = "IC6" # Default/Auto-detect placeholder
            
        if not self.params_base:
            return {
                "error": "heme parameters not configured",
                "hint": "Set AMBER_HEME_PARAMS or pass params_base explicitly.",
            }
        params_dir = os.path.join(self.params_base, state)
        logger.info(f"Using Heme State: {state}, Params: {params_dir}")
        
        # 2. Superposition (Kabsch) logic call
        # 3. Write Prepared PDB
        # 4. Generate tleap.in
        
        return {"state": state, "output_dir": output_dir}

    def generate_tleap(self, output_dir, pdb_name, state, params_dir, cys_resid, hem_resid):
        """Generates tleap input with Fe-S bonding."""
        wsl_params = win_to_wsl(params_dir)
        content = f"""
source leaprc.protein.ff19SB
source leaprc.gaff2
source leaprc.water.tip3p
loadamberparams {wsl_params}/{state}.frcmod
HEM = loadmol2 {wsl_params}/HEM.mol2
CYP = loadmol2 {wsl_params}/CYP.mol2
mol = loadpdb {pdb_name}
bond mol.{cys_resid}.SG mol.{hem_resid}.FE
solvateoct mol TIP3PBOX 12.0
addionsrand mol Na+ 0
addionsrand mol Cl- 0
saveamberparm mol complex.prmtop complex.inpcrd
quit
"""
        write_file(os.path.join(output_dir, "tleap.in"), content)
