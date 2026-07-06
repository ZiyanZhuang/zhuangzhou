"""
MetaKG Configuration — paths, organism registry, compartment codes
"""

import os

# ── Paths ─────────────────────────────────────────────────────────────────────
METAKG_ROOT = os.environ.get("METAKG_ROOT", "")

MODELS = {
    "ecoli": {
        "json":    os.path.join(METAKG_ROOT, "iML1515.json"),
        "nodes":   os.path.join(METAKG_ROOT, "nodes.parquet"),
        "edges":   os.path.join(METAKG_ROOT, "edges.parquet"),
        "label":   "E. coli K-12 MG1655",
        "bigg_id": "iML1515",
        "genes":   1515,
        "rxns":    3150,
    },
    "yeast": {
        "json":    os.path.join(METAKG_ROOT, "iMM904", "iMM904.json"),
        "nodes":   os.path.join(METAKG_ROOT, "iMM904", "output", "nodes.parquet"),
        "edges":   os.path.join(METAKG_ROOT, "iMM904", "output", "edges.parquet"),
        "label":   "S. cerevisiae S288C",
        "bigg_id": "iMM904",
        "genes":   904,
        "rxns":    1577,
    },
}

# ── Compartments ──────────────────────────────────────────────────────────────
COMPARTMENTS = {
    "c": "cytoplasm",
    "e": "extracellular",
    "p": "periplasm",
    "m": "mitochondria",
    "x": "peroxisome",
    "n": "nucleus",
    "r": "ER",
    "g": "Golgi",
    "v": "vacuole",
}

# ── Cofactors of interest for balance analysis ────────────────────────────────
COFACTOR_PAIRS = {
    "nad":   ("nad_c",   "nadh_c",   "NAD⁺/NADH"),
    "nadp":  ("nadp_c",  "nadph_c",  "NADP⁺/NADPH"),
    "atp":   ("atp_c",   "adp_c",    "ATP/ADP"),
    "fad":   ("fad_c",   "fadh2_c",  "FAD/FADH₂"),
    "coa":   ("coa_c",   "accoa_c",  "CoA/Acetyl-CoA"),
    "q8":    ("q8_c",    "q8h2_c",   "Ubiquinone Q8"),
    "o2":    ("o2_c",    "h2o_c",    "O₂/H₂O (electron acceptor)"),
}

# ── Known toxic metabolites (mM thresholds from literature) ──────────────────
TOXICITY_DB = {
    # E. coli
    "acald_c":  {"threshold_mM": 5,    "mechanism": "aldehyde toxicity, protein crosslinking"},
    "mda_c":    {"threshold_mM": 0.1,  "mechanism": "malondialdehyde, oxidative stress"},
    "etoh_c":   {"threshold_mM": 400,  "mechanism": "membrane disruption"},
    "lac__D_c": {"threshold_mM": 100,  "mechanism": "acidification"},
    "succ_c":   {"threshold_mM": 200,  "mechanism": "mild; PDC competitor"},
    "fald_c":   {"threshold_mM": 0.5,  "mechanism": "formaldehyde, severe protein damage"},
    "h2o2_c":   {"threshold_mM": 0.1,  "mechanism": "ROS, DNA/protein oxidation"},
    "4hbz_c":   {"threshold_mM": 2.0,  "mechanism": "aromatic aldehyde toxicity"},
    "phenol_c": {"threshold_mM": 3.0,  "mechanism": "membrane disruption"},
    # Yeast
    "hmf_c":    {"threshold_mM": 10,   "mechanism": "HMF inhibition of glycolysis"},
    "furfural_c": {"threshold_mM": 8,  "mechanism": "enzyme inhibition"},
}

# ── Key E. coli promoters (Anderson library + constitutive/inducible) ─────────
ECOLI_PROMOTERS = {
    # Anderson constitutive library (relative strength vs J23100 = 1.0)
    "J23100": {"strength": 1.000, "type": "constitutive", "sequence": "TTGACGGCTAGCTCAGTCCTAGGTACAGTGCTAGC"},
    "J23101": {"strength": 0.700, "type": "constitutive", "sequence": "TTTACAGCTAGCTCAGTCCTAGGTATTATGCTAGC"},
    "J23102": {"strength": 0.860, "type": "constitutive", "sequence": "TTGACAGCTAGCTCAGTCCTAGGTACTGTGCTAGC"},
    "J23103": {"strength": 0.010, "type": "constitutive", "sequence": "CTGATAGCTAGCTCAGTCCTAGGGATTATGCTAGC"},
    "J23104": {"strength": 0.720, "type": "constitutive", "sequence": "TTGACAGCTAGCTCAGTCCTAGGTATTGTGCTAGC"},
    "J23105": {"strength": 0.240, "type": "constitutive", "sequence": "TTTACGGCTAGCTCAGTCCTAGGTACTATGCTAGC"},
    "J23106": {"strength": 0.470, "type": "constitutive", "sequence": "TTTACGGCTAGCTCAGTCCTAGGTATAATGCTAGC"},
    "J23107": {"strength": 0.360, "type": "constitutive", "sequence": "TTTACGGCTAGCTCAGTCCTAGGTATTATGCTAGC"},
    "J23108": {"strength": 0.510, "type": "constitutive", "sequence": "CTGACAGCTAGCTCAGTCCTAGGTATAATGCTAGC"},
    "J23109": {"strength": 0.040, "type": "constitutive", "sequence": "TTTACAGCTAGCTCAGTCCTAGGGACTGTGCTAGC"},
    "J23110": {"strength": 0.330, "type": "constitutive", "sequence": "TTTACGGCTAGCTCAGTCCTAGGTACAATGCTAGC"},
    "J23111": {"strength": 0.580, "type": "constitutive", "sequence": "TTGACGGCTAGCTCAGTCCTAGGTATAGTGCTAGC"},
    "J23113": {"strength": 0.010, "type": "constitutive", "sequence": "CTGATGGCTAGCTCAGTCCTAGGGATTATGCTAGC"},
    "J23114": {"strength": 0.120, "type": "constitutive", "sequence": "TTTATGGCTAGCTCAGTCCTAGGTACAATGCTAGC"},
    "J23115": {"strength": 0.150, "type": "constitutive", "sequence": "TTTATAGCTAGCTCAGCCCTAGGTATTATGCTAGC"},
    "J23116": {"strength": 0.160, "type": "constitutive", "sequence": "TTGACAGCTAGCTCAGTCCTAGGGACTATGCTAGC"},
    "J23117": {"strength": 0.062, "type": "constitutive", "sequence": "TTGACAGCTAGCTCAGTCCTAGGGATTGTGCTAGC"},
    "J23118": {"strength": 0.560, "type": "constitutive", "sequence": "TTGACGGCTAGCTCAGTCCTAGGTATAATGCTAGC"},
    "J23119": {"strength": 0.908, "type": "constitutive", "sequence": "TTGACAGCTAGCTCAGTCCTAGGTATAATGCTAGC"},
    # Inducible
    "Ptrc":   {"strength": 0.95,  "type": "inducible",    "inducer": "IPTG",   "sequence": None},
    "Plac":   {"strength": 0.60,  "type": "inducible",    "inducer": "IPTG",   "sequence": None},
    "Ptac":   {"strength": 1.00,  "type": "inducible",    "inducer": "IPTG",   "sequence": None},
    "Para":   {"strength": 0.80,  "type": "inducible",    "inducer": "L-arabinose", "sequence": None},
    "Prha":   {"strength": 0.70,  "type": "inducible",    "inducer": "L-rhamnose",  "sequence": None},
    "PT7":    {"strength": 5.00,  "type": "inducible",    "inducer": "IPTG (T7 RNAP)", "sequence": None},
    "Ptet":   {"strength": 0.85,  "type": "inducible",    "inducer": "aTc",    "sequence": None},
}

# ── S. cerevisiae promoters ────────────────────────────────────────────────────
YEAST_PROMOTERS = {
    "PTEF1":  {"strength": 1.00,  "type": "constitutive"},
    "PGPD1":  {"strength": 0.90,  "type": "constitutive"},
    "PADH1":  {"strength": 0.70,  "type": "constitutive"},
    "PCYC1":  {"strength": 0.30,  "type": "constitutive"},
    "PPGK1":  {"strength": 0.85,  "type": "constitutive"},
    "PGAL1":  {"strength": 2.00,  "type": "inducible",  "inducer": "galactose"},
    "PGAL10": {"strength": 1.80,  "type": "inducible",  "inducer": "galactose"},
    "PCUP1":  {"strength": 0.60,  "type": "inducible",  "inducer": "Cu²⁺"},
    "PMET3":  {"strength": 0.50,  "type": "repressible","repressor": "methionine"},
    "PTET":   {"strength": 0.75,  "type": "inducible",  "inducer": "doxycycline"},
}

# ── Common E. coli integration sites (attTn7, landing pads) ──────────────────
ECOLI_INTEGRATION_SITES = {
    "glmS":  {"locus": "b3729", "description": "safe harbor, downstream of glmS"},
    "lacZ":  {"locus": "b0344", "description": "classic, disrupts lacZ"},
    "ybhC":  {"locus": "b0766", "description": "neutral locus"},
    "rhtC":  {"locus": "b0813", "description": "threonine efflux; knock-in + KO"},
    "attTn7":{"locus": "b3929", "description": "single-copy, site-specific (glmS downstream)"},
}

# ── Well-characterized E. coli CRISPR PAM preferences ────────────────────────
CRISPR_SYSTEMS = {
    "SpCas9":  {"pam": "NGG",  "spacer_len": 20, "cut_offset": -3},
    "SaCas9":  {"pam": "NNGRRT", "spacer_len": 21, "cut_offset": -3},
    "Cas12a":  {"pam": "TTTV",   "spacer_len": 23, "cut_offset": 18},
    "dCas9":   {"pam": "NGG",    "spacer_len": 20, "cut_offset": None, "mode": "CRISPRi"},
}
