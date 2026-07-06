"""
Configuration for Komagataella phaffii codon optimization.
"""

import os

# Pichia-CLM paths
PICHIA_CLM_REPO = os.environ.get("PICHIA_CLM_REPO", "")
PICHIA_CLM_CHECKPOINT_DIR = os.environ.get(
    "PICHIA_CLM_CHECKPOINT_DIR",
    os.path.join(PICHIA_CLM_REPO, "Model_PichiaCLM", "2Target_AllData") if PICHIA_CLM_REPO else "",
)
PICHIA_CLM_HYPER_CSV = os.path.join(
    PICHIA_CLM_REPO, "Model_PichiaCLM", "Training",
    "BO_forHyperParameter", "Arch1", "Round3.csv"
) if PICHIA_CLM_REPO else os.environ.get("PICHIA_CLM_HYPER_CSV", "")

# K. phaffii Kazusa taxonomy ID
KPHAFFII_TAXID = 4922

# Model constraints
MAX_PROTEIN_LENGTH = 999  # Pichia-CLM max (1000 tokens - 1 for start)

# DNAchisel defaults
DEFAULT_GC_MIN = 0.30
DEFAULT_GC_MAX = 0.60
AVOID_PATTERNS = [
    "GGATCC",  # BamHI
    "GAATTC",  # EcoRI
    "AAGCTT",  # HindIII
    "AATAAA",  # poly-A signal
    "ATTTA",   # AU-rich element (mRNA instability)
]

# K. phaffii preferred codons (highest frequency per AA, from Kazusa 4922)
KPHAFFII_PREFERRED = {
    'A': 'GCT', 'C': 'TGT', 'D': 'GAT', 'E': 'GAA',
    'F': 'TTT', 'G': 'GGT', 'H': 'CAT', 'I': 'ATT',
    'K': 'AAG', 'L': 'TTG', 'M': 'ATG', 'N': 'AAC',
    'P': 'CCA', 'Q': 'CAA', 'R': 'AGA', 'S': 'TCT',
    'T': 'ACT', 'V': 'GTT', 'W': 'TGG', 'Y': 'TAC',
    '*': 'TAA',
}
