"""
SessionContext — persistent analysis session state for HuiZi.
Tracks what protein/ligand is currently being analyzed.
"""
import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)


@dataclass
class AnalysisTarget:
    """
    Represents the current scientific analysis subject.
    Accumulates over the course of a session.
    """
    # Protein identity
    protein_name: Optional[str] = None
    protein_gene: Optional[str] = None
    organism: Optional[str] = None
    uniprot_id: Optional[str] = None
    pdb_id: Optional[str] = None
    fasta_sequence: Optional[str] = None
    pdb_path: Optional[str] = None
    function_description: Optional[str] = None
    ec_number: Optional[str] = None
    active_site_residues: List[str] = field(default_factory=list)

    # Structure quality (for predicted structures)
    structure_source: Optional[str] = None   # 'experimental'|'afdb'|'esmatlas'|'user'
    plddt_mean: Optional[float] = None        # Mean pLDDT score (predicted structures only)
    structure_quality_tier: Optional[str] = None  # 'high'|'confident'|'low'|'very_low'

    # Ligand
    ligand_name: Optional[str] = None
    ligand_smiles: Optional[str] = None
    ligand_pubchem_cid: Optional[str] = None
    ligand_mw: Optional[float] = None

    # Analysis history
    analyses: List[str] = field(default_factory=list)
    result_paths: Dict[str, str] = field(default_factory=dict)
    key_findings: List[str] = field(default_factory=list)

    # Timestamps
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def has_any(self) -> bool:
        return any([
            self.protein_name, self.uniprot_id, self.pdb_path,
            self.ligand_name, self.analyses
        ])

    def touch(self):
        self.updated_at = time.time()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'AnalysisTarget':
        known = set(cls.__dataclass_fields__.keys())
        return cls(**{k: v for k, v in d.items() if k in known})

    def summary(self) -> str:
        parts = []
        if self.protein_name:
            parts.append(self.protein_name)
        if self.organism:
            parts.append(f"({self.organism})")
        if self.ligand_name:
            parts.append(f"+ {self.ligand_name}")
        if self.analyses:
            parts.append(f"[{', '.join(self.analyses)}]")
        return " ".join(parts) if parts else "(no target set)"


class SessionContext:
    """
    Persistent session context backed by a JSON file.
    Auto-saves on mutation, auto-loads on init.
    """

    _STATE_DIR = os.path.join(os.path.expanduser("~"), ".zhuangzhou", "huizi")

    def __init__(self, state_dir: str = None, session_id: str = "default"):
        self._state_dir = state_dir or self._STATE_DIR
        self._session_id = session_id
        self._state_file = os.path.join(
            self._state_dir, self._session_id, "session_context.json"
        )
        os.makedirs(os.path.dirname(self._state_file), exist_ok=True)
        self._target: AnalysisTarget = self._load()

    @property
    def target(self) -> AnalysisTarget:
        return self._target

    def save(self):
        self._target.touch()
        try:
            with open(self._state_file, 'w', encoding='utf-8') as f:
                json.dump(self._target.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning("Failed to save HuiZi session context: %s", e)

    def reset(self):
        self._target = AnalysisTarget()
        self.save()

    def _load(self) -> AnalysisTarget:
        if not os.path.exists(self._state_file):
            return AnalysisTarget()
        try:
            with open(self._state_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return AnalysisTarget.from_dict(data)
        except json.JSONDecodeError as e:
            logger.warning("Session JSON corrupted, starting fresh: %s", e)
            return AnalysisTarget()
        except OSError:
            pass
        return AnalysisTarget()
