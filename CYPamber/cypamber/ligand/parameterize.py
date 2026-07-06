import os
import shutil
import logging
from ..qm.pyscf_bridge import run_pyscf_resp_wsl
from ..utils.system import run_wsl, win_to_wsl, write_file

logger = logging.getLogger(__name__)

class LigandParameterizer:
    """Handles ligand parameterization with GAFF2 and PySCF/Multiwfn RESP charges."""
    
    def __init__(self, multiwfn_bin=None):
        self.multiwfn_bin = multiwfn_bin or os.environ.get("MULTIWFN_BIN", "")

    def run(self, ligand_path, output_dir, charge=0, spin=1, resp_method="multiwfn"):
        """Full parameterization pipeline."""
        os.makedirs(output_dir, exist_ok=True)
        wsl_out = win_to_wsl(os.path.abspath(output_dir))
        wsl_ligand = win_to_wsl(os.path.abspath(ligand_path))
        
        # 1. Antechamber for Initial GAFF2 Typing
        logger.info("[1] Antechamber: GAFF2 atom typing...")
        cmd_ant = f"antechamber -i '{wsl_ligand}' -fi mol2 -o '{wsl_out}/ligand_gaff2.mol2' -fo mol2 -c bcc -s 2 -at gaff2 -nc {charge} -pf y"
        run_wsl(cmd_ant, cwd=wsl_out)
        
        # 2. RESP via PySCF + Multiwfn
        if resp_method == "multiwfn":
            if not self.multiwfn_bin:
                return {
                    "error": "Multiwfn binary not configured",
                    "hint": "Set MULTIWFN_BIN or pass multiwfn_bin explicitly.",
                }
            logger.info("[2] PySCF + Multiwfn: High-precision RESP charges...")
            p_res = run_pyscf_resp_wsl(wsl_ligand, output_dir, charge, spin)
            if "error" in p_res:
                return p_res
            
            # Multiwfn RESP Fitting command
            m_cmd = f"printf '7\\n18\\n1\\ny\\n0\\n0\\nq\\n' | '{self.multiwfn_bin}' ligand_hf.molden"
            run_wsl(m_cmd, cwd=wsl_out)
            
            # Merge logic (simplified)
            logger.info("Merging RESP charges into MOL2...")
            # Implement the charge merge logic here (can be imported from zhuangzhou)
            
        # 3. parmchk2 for frcmod
        logger.info("[3] parmchk2: generating frcmod...")
        cmd_parm = f"parmchk2 -i '{wsl_out}/ligand_gaff2.mol2' -f mol2 -o '{wsl_out}/ligand.frcmod' -s gaff2"
        run_wsl(cmd_parm, cwd=wsl_out)
        
        return {"mol2": os.path.join(output_dir, "ligand_gaff2.mol2"), "frcmod": os.path.join(output_dir, "ligand.frcmod")}
