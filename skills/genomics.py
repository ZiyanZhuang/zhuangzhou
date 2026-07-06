"""Population-genomics computation skills.

These skills integrate the local beginner-genome-computation workflow into
Zhuangzhou as command-plan generators. They do not execute external genomics
software by default.
"""

from __future__ import annotations

from typing import Dict, List, Optional


def popgen_workflow_plan(
    analyses: Optional[List[str]] = None,
    vcf: str = "input.vcf.gz",
    popmap: str = "pop_map.txt",
    workdir: str = "popgen_work",
    threads: int = 8,
    maf: float = 0.05,
    geno: float = 0.1,
    ld: str = "50 10 0.2",
    window: int = 100000,
    step: int = 50000,
    pca_components: int = 10,
    k_min: int = 2,
    k_max: int = 10,
    admixture_seeds: Optional[List[int]] = None,
    treemix_root: str = "ROOT_POP",
    treemix_max_m: int = 5,
    treemix_block_size: int = 500,
    alignment: str = "snp.phy",
    output_dir: Optional[str] = None,
) -> Dict:
    try:
        from zhuangzhou.genomics.popgen import build_popgen_workflow

        return build_popgen_workflow(
            analyses=analyses,
            vcf=vcf,
            popmap=popmap,
            workdir=workdir,
            threads=threads,
            maf=maf,
            geno=geno,
            ld=ld,
            window=window,
            step=step,
            pca_components=pca_components,
            k_min=k_min,
            k_max=k_max,
            admixture_seeds=admixture_seeds,
            treemix_root=treemix_root,
            treemix_max_m=treemix_max_m,
            treemix_block_size=treemix_block_size,
            alignment=alignment,
        )
    except Exception as exc:
        return {"error": str(exc), "hint": "Check analyses, VCF path string, popmap schema, and numeric parameters."}


def popgen_input_contract(vcf: str = "input.vcf.gz", popmap: str = "pop_map.txt") -> Dict:
    try:
        from zhuangzhou.genomics.popgen import capture_version_commands, input_contract_commands

        return {
            "purpose": "Validate VCF sample IDs against a two-column sample/population map before population-level analysis.",
            "inputs": {"vcf": vcf, "popmap": popmap},
            "commands": [*capture_version_commands(), *input_contract_commands(vcf, popmap)],
            "outputs": ["vcf.samples", "popmap.samples", "populations.txt", "*.samples"],
            "validation": ["diff -u vcf.samples.sorted popmap.samples.sorted", "wc -l vcf.samples popmap.samples populations.txt"],
        }
    except Exception as exc:
        return {"error": str(exc)}


def register(registry):
    from .registry import Skill, SkillParam

    S = SkillParam
    registry.register(Skill(
        name="popgen_workflow_plan",
        description=(
            "Generate an auditable Linux command plan for VCF/PLINK population-genomics analyses: "
            "pi, Tajima's D, heterozygosity, Fst, PCA, ADMIXTURE, TreeMix, IQ-TREE, and VCF2Dis."
        ),
        category="genomics",
        fn=popgen_workflow_plan,
        inputs=[
            S("analyses", "list", "Analyses to include: diversity,fst,pca,admixture,treemix,iqtree,vcf2dis", required=False),
            S("vcf", "string", "VCF/BCF path as used on the Linux analysis host", required=False, default="input.vcf.gz"),
            S("popmap", "string", "Two-column sample_id population_id map", required=False, default="pop_map.txt"),
            S("workdir", "string", "Writable work directory on analysis host", required=False, default="popgen_work"),
            S("threads", "int", "Thread count", required=False, default=8),
            S("maf", "float", "Minor allele frequency filter", required=False, default=0.05),
            S("geno", "float", "Missing genotype/site filter", required=False, default=0.1),
            S("ld", "string", "PLINK --indep-pairwise parameters", required=False, default="50 10 0.2"),
            S("window", "int", "Window size for diversity statistics", required=False, default=100000),
            S("step", "int", "Window step for diversity statistics", required=False, default=50000),
            S("pca_components", "int", "Number of PCA components", required=False, default=10),
            S("k_min", "int", "Minimum ADMIXTURE K", required=False, default=2),
            S("k_max", "int", "Maximum ADMIXTURE K", required=False, default=10),
            S("admixture_seeds", "list", "Replicate random seeds for ADMIXTURE runs", required=False),
            S("treemix_root", "string", "TreeMix root population", required=False, default="ROOT_POP"),
            S("treemix_max_m", "int", "Maximum TreeMix migration edges", required=False, default=5),
            S("treemix_block_size", "int", "TreeMix block size -k", required=False, default=500),
            S("alignment", "string", "SNP/alignment file for IQ-TREE", required=False, default="snp.phy"),
        ],
        outputs=[
            S("blocks", "list", "Workflow blocks with commands, outputs, validation, and notes"),
            S("method_note", "string", "Method text suitable for report draft"),
        ],
        requires_gpu=False,
        requires_wsl=False,
        requires_network=False,
        output_type="analysis",
        risk_level="low",
        estimated_time="~1s",
        workflow_stage="plan",
        when_to_use="Planning population-genomics analyses from VCF/PLINK data before running external tools.",
    ))

    registry.register(Skill(
        name="popgen_input_contract",
        description="Generate version-capture and VCF-vs-popmap sample validation commands for population genomics.",
        category="genomics",
        fn=popgen_input_contract,
        inputs=[
            S("vcf", "string", "VCF path as used on Linux analysis host", required=False, default="input.vcf.gz"),
            S("popmap", "string", "Two-column sample_id population_id map", required=False, default="pop_map.txt"),
        ],
        outputs=[S("commands", "list", "Validation commands")],
        requires_gpu=False,
        requires_wsl=False,
        requires_network=False,
        output_type="analysis",
        risk_level="low",
        estimated_time="~1s",
        workflow_stage="plan",
    ))

    return 2


register_skills = register
