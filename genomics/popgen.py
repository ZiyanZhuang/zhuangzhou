"""Command-plan generator for beginner/intermediate population genomics.

The module does not execute VCFtools/PLINK/TreeMix jobs. It produces explicit,
auditable Linux command blocks with validation steps so Zhuangzhou can integrate
genomics analyses into DBTL-style project planning without hiding file-format
boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


DEFAULT_MAF = 0.05
DEFAULT_GENO = 0.1
DEFAULT_LD = "50 10 0.2"
DEFAULT_WINDOW = 100_000
DEFAULT_STEP = 50_000


@dataclass
class WorkflowBlock:
    name: str
    purpose: str
    commands: list[str]
    outputs: list[str] = field(default_factory=list)
    validation: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "purpose": self.purpose,
            "commands": self.commands,
            "outputs": self.outputs,
            "validation": self.validation,
            "notes": self.notes,
        }


def _vcf_flag(vcf_var: str = "$VCF") -> str:
    return f'$(case "{vcf_var}" in *.gz) echo "--gzvcf {vcf_var}" ;; *) echo "--vcf {vcf_var}" ;; esac)'


def capture_version_commands() -> list[str]:
    return [
        "date",
        "vcftools --version 2>&1 | head",
        "plink --version 2>&1 | head",
        "admixture --version 2>&1 | head || true",
        "iqtree --version 2>&1 | head || true",
        "treemix --version 2>&1 | head || true",
        "Rscript --version 2>&1 | head || true",
    ]


def input_contract_commands(vcf: str = "$VCF", popmap: str = "$POPMAP") -> list[str]:
    return [
        f'sed -i "s/\\r$//" "{popmap}"',
        f'if echo "{vcf}" | grep -q "\\.gz$"; then zcat "{vcf}" | grep "^#CHROM" | cut -f10- | tr "\\t" "\\n" > vcf.samples; else grep "^#CHROM" "{vcf}" | cut -f10- | tr "\\t" "\\n" > vcf.samples; fi',
        f"awk '{{print $1}}' \"{popmap}\" > popmap.samples",
        "sort vcf.samples > vcf.samples.sorted",
        "sort popmap.samples > popmap.samples.sorted",
        "diff -u vcf.samples.sorted popmap.samples.sorted",
        f"cut -f2 \"{popmap}\" | sort -u > populations.txt",
        f'while read pop; do awk -v p="$pop" \'$2==p {{print $1}}\' "{popmap}" > "${{pop}}.samples"; done < populations.txt',
    ]


def _diversity_block(window: int, step: int) -> WorkflowBlock:
    return WorkflowBlock(
        name="diversity",
        purpose="Estimate within-population nucleotide diversity, Tajima's D, allele frequencies, and heterozygosity.",
        commands=[
            "> pi.summary.tsv",
            "> tajima.summary.tsv",
            'while read pop; do\n'
            f'  vcftools $VCFTOOLS_VCF --keep "${{pop}}.samples" --window-pi {window} --window-pi-step {step} --out "${{pop}}.pi{window}"\n'
            f'  vcftools $VCFTOOLS_VCF --keep "${{pop}}.samples" --TajimaD {window} --out "${{pop}}.tajima{window}"\n'
            '  vcftools $VCFTOOLS_VCF --keep "${pop}.samples" --freq --out "${pop}.freq"\n'
            f'  awk -v p="$pop" \'NR==1{{for(i=1;i<=NF;i++) if($i=="PI") col=i}} NR>1 && col {{sum+=$col; n++}} END{{if(n) printf "%s\\t%.8g\\t%d\\n", p, sum/n, n}}\' "${{pop}}.pi{window}.windowed.pi" >> pi.summary.tsv\n'
            f'  awk -v p="$pop" \'NR>1 && $4 !~ /nan/i {{sum+=$4; n++}} END{{if(n) printf "%s\\t%.8g\\t%d\\n", p, sum/n, n}}\' "${{pop}}.tajima{window}.Tajima.D" >> tajima.summary.tsv\n'
            "done < populations.txt",
            "vcftools $VCFTOOLS_VCF --het --out all_samples",
            "awk 'NR>1 {ho=1-$2/$4; print $1, ho}' all_samples.het > individual.ho.tsv",
            "> population.ho.tsv",
            'while read pop; do awk -v p="$pop" \'NR==FNR{ho[$1]=$2; next} ($1 in ho){sum+=ho[$1]; n++} END{if(n) printf "%s\\t%.8g\\t%d\\n", p, sum/n, n}\' individual.ho.tsv "${pop}.samples" >> population.ho.tsv; done < populations.txt',
        ],
        outputs=[
            "pi.summary.tsv",
            "tajima.summary.tsv",
            "individual.ho.tsv",
            "population.ho.tsv",
            "*.freq.frq",
        ],
        validation=[
            "test -s pi.summary.tsv && head pi.summary.tsv",
            "test -s tajima.summary.tsv && head tajima.summary.tsv",
            "test -s all_samples.het && head all_samples.het",
        ],
        notes=[
            "Average the documented PI column from .windowed.pi; do not assume a fixed numeric column.",
            "Filter nan Tajima's D windows before summaries.",
            "Observed heterozygosity is computed as 1 - O(HOM)/N_SITES for VCFtools .het schema.",
        ],
    )


def _fst_block() -> WorkflowBlock:
    return WorkflowBlock(
        name="fst_pairwise",
        purpose="Compute pairwise Weir-Cockerham Fst between populations.",
        commands=[
            "mkdir -p fst_pairwise",
            'while read p1; do\n'
            '  while read p2; do\n'
            '    [ "$p1" = "$p2" ] && continue\n'
            '    out="fst_pairwise/${p1}_vs_${p2}"\n'
            '    vcftools $VCFTOOLS_VCF --weir-fst-pop "${p1}.samples" --weir-fst-pop "${p2}.samples" --out "$out"\n'
            "  done < populations.txt\n"
            "done < populations.txt",
        ],
        outputs=["fst_pairwise/*.weir.fst", "fst_pairwise/*.log"],
        validation=[
            "find fst_pairwise -name '*.weir.fst' -size +0 -print | head",
            "tail -n 5 fst_pairwise/*.log 2>/dev/null | head -40",
        ],
        notes=["State whether reporting mean Fst, weighted Fst, per-site Fst, or windowed summaries."],
    )


def _plink_pca_block(maf: float, geno: float, ld: str, pcs: int) -> WorkflowBlock:
    return WorkflowBlock(
        name="plink_pca",
        purpose="Create QC-filtered, LD-pruned PLINK files and compute PCA.",
        commands=[
            f"plink --vcf \"$VCF\" --maf {maf} --geno {geno} --make-bed --allow-extra-chr --set-missing-var-ids @:# --out qc",
            f"plink --bfile qc --indep-pairwise {ld} --out qc.ld",
            "plink --bfile qc --extract qc.ld.prune.in --make-bed --out qc.pruned",
            f"plink --bfile qc.pruned --pca {pcs} --out pca",
        ],
        outputs=["qc.bed/qc.bim/qc.fam", "qc.ld.prune.in", "qc.pruned.*", "pca.eigenvec", "pca.eigenval"],
        validation=[
            "test -s qc.pruned.bed && test -s pca.eigenvec",
            "wc -l qc.ld.prune.in pca.eigenvec",
            "head pca.eigenvec",
        ],
        notes=["Use LD-pruned SNPs for PCA to reduce LD-correlated signal inflation."],
    )


def _admixture_block(k_min: int, k_max: int, seeds: Iterable[int]) -> WorkflowBlock:
    seed_list = " ".join(str(seed) for seed in seeds)
    return WorkflowBlock(
        name="admixture",
        purpose="Scan ancestry components on LD-pruned PLINK BED input with cross-validation.",
        commands=[
            "mkdir -p admixture",
            f"for K in $(seq {k_min} {k_max}); do\n"
            f"  for seed in {seed_list}; do\n"
            '    admixture --cv -s "$seed" qc.pruned.bed "$K" > "admixture/K${K}.seed${seed}.log" 2>&1\n'
            '    mv "qc.pruned.${K}.Q" "admixture/K${K}.seed${seed}.Q"\n'
            '    mv "qc.pruned.${K}.P" "admixture/K${K}.seed${seed}.P"\n'
            "  done\n"
            "done",
            'grep -h "CV error" admixture/*.log > admixture/cv_errors.txt',
        ],
        outputs=["admixture/*.Q", "admixture/*.P", "admixture/*.log", "admixture/cv_errors.txt"],
        validation=["test -s admixture/cv_errors.txt && cat admixture/cv_errors.txt", "ls admixture/*.Q | head"],
        notes=["Lowest CV K is not automatically biologically true; compare with PCA, sampling design, and replicate stability."],
    )


def _treemix_block(root_pop: str, max_m: int, block_size: int) -> WorkflowBlock:
    return WorkflowBlock(
        name="treemix",
        purpose="Prepare population allele-frequency data and scan TreeMix migration edges.",
        commands=[
            "mkdir -p treemix_run",
            "( cd treemix_run\n"
            "  plink --vcf \"$VCF\" --make-bed --allow-extra-chr --set-missing-var-ids @:# --out temp\n"
            "  awk '{print $1, $2}' \"$POPMAP\" > pop_map.txt\n"
            "  awk 'NR==FNR{pop[$1]=$2; next} {print $1,$2,$3,$4,$5,pop[$2]}' pop_map.txt temp.fam > temp.new.fam\n"
            "  mv temp.new.fam temp.fam\n"
            "  plink --bfile temp --freq --family --out treemix_input\n"
            "  gzip -f treemix_input.frq.strat\n"
            "  # Convert treemix_input.frq.strat.gz to treemix.frq.gz with a vetted converter.\n"
            "  zcat treemix.frq.gz | awk 'NR==1{h=NF; print \"header\", h; next} NF!=h{print \"bad field count\", NR, NF; bad=1; exit} END{if(!bad) print \"field counts OK\"}'\n"
            f"  for m in $(seq 0 {max_m}); do treemix -i treemix.frq.gz -o \"m${{m}}\" -m \"$m\" -root \"{root_pop}\" -k {block_size} -global -se > \"m${{m}}.log\" 2>&1; done\n"
            "  echo -e \"m\\tlnL\" > likelihood.tsv\n"
            f"  for m in $(seq 0 {max_m}); do grep \"ln(likelihood)\" \"m${{m}}.log\" | tail -1 | awk -v m=\"$m\" '{{print m \"\\t\" $NF}}' >> likelihood.tsv; done\n"
            ")",
        ],
        outputs=["treemix_run/treemix.frq.gz", "treemix_run/m*.log", "treemix_run/likelihood.tsv"],
        validation=["test -s treemix_run/likelihood.tsv && cat treemix_run/likelihood.tsv", "tail -n 20 treemix_run/m0.log"],
        notes=["Validate TreeMix field counts before running; rows must match the population header after optional SNP-ID removal."],
    )


def _iqtree_block(alignment: str, prefix: str) -> WorkflowBlock:
    return WorkflowBlock(
        name="iqtree",
        purpose="Build a maximum-likelihood tree from SNP/alignment input using documented IQ-TREE syntax.",
        commands=[
            "iqtree --version",
            "iqtree -h | head -80",
            f"iqtree -s \"{alignment}\" -m MFP -B 1000 -alrt 1000 -T AUTO -pre \"{prefix}\"",
            f"# IQ-TREE 1 may use -bb 1000 instead of -B 1000; follow installed `iqtree -h`.",
        ],
        outputs=[f"{prefix}.treefile", f"{prefix}.iqtree", f"{prefix}.log"],
        validation=[f"test -s {prefix}.treefile && head {prefix}.treefile", f"tail -n 30 {prefix}.log"],
        notes=["Do not use unsupported flags such as `iqtree -nj`; follow the installed manual/help output."],
    )


def _vcf2dis_block() -> WorkflowBlock:
    return WorkflowBlock(
        name="vcf2dis",
        purpose="Compute VCF-based p-distance matrix.",
        commands=[
            'VCF2Dis_multi -InPut "$VCF" -OutPut p_distance.mat',
            "test -s p_distance.mat",
            "head -3 p_distance.mat",
        ],
        outputs=["p_distance.mat"],
        validation=["test -s p_distance.mat && head -3 p_distance.mat"],
        notes=["If VCF2Dis reports zero samples, first check VCF path/readability before changing biological filters."],
    )


def build_popgen_workflow(
    analyses: list[str] | None = None,
    vcf: str = "input.vcf.gz",
    popmap: str = "pop_map.txt",
    workdir: str = "popgen_work",
    threads: int = 8,
    maf: float = DEFAULT_MAF,
    geno: float = DEFAULT_GENO,
    ld: str = DEFAULT_LD,
    window: int = DEFAULT_WINDOW,
    step: int = DEFAULT_STEP,
    pca_components: int = 10,
    k_min: int = 2,
    k_max: int = 10,
    admixture_seeds: list[int] | None = None,
    treemix_root: str = "ROOT_POP",
    treemix_max_m: int = 5,
    treemix_block_size: int = 500,
    alignment: str = "snp.phy",
) -> dict:
    selected = analyses or ["diversity", "fst", "pca"]
    selected = [item.lower() for item in selected]
    admixture_seeds = admixture_seeds or [1, 2, 3, 4, 5]

    setup = WorkflowBlock(
        name="setup_and_input_contract",
        purpose="Define variables, capture software versions, and validate VCF/sample-map consistency.",
        commands=[
            'INPUT_ROOT="$PWD"',
            f'WORKDIR="{workdir}"',
            f'VCF="{vcf}"',
            f'POPMAP="{popmap}"',
            f"THREADS={threads}",
            'mkdir -p "$WORKDIR"',
            'cd "$WORKDIR"',
            'case "$VCF" in /*) ;; *) VCF="$INPUT_ROOT/$VCF" ;; esac',
            'case "$POPMAP" in /*) ;; *) POPMAP="$INPUT_ROOT/$POPMAP" ;; esac',
            'VCFTOOLS_VCF=$(case "$VCF" in *.gz) echo "--gzvcf $VCF" ;; *) echo "--vcf $VCF" ;; esac)',
            *capture_version_commands(),
            *input_contract_commands("$VCF", "$POPMAP"),
        ],
        outputs=["vcf.samples", "popmap.samples", "populations.txt", "*.samples"],
        validation=["diff -u vcf.samples.sorted popmap.samples.sorted", "wc -l vcf.samples popmap.samples populations.txt"],
        notes=["POPMAP must be two columns: sample_id population_id. Most shell recipes assume no header."],
    )

    blocks = [setup]
    if "diversity" in selected or "pi" in selected or "tajima" in selected or "heterozygosity" in selected:
        blocks.append(_diversity_block(window, step))
    if "fst" in selected:
        blocks.append(_fst_block())
    if "pca" in selected or "plink" in selected:
        blocks.append(_plink_pca_block(maf, geno, ld, pca_components))
    if "admixture" in selected or "structure" in selected:
        if not any(block.name == "plink_pca" for block in blocks):
            blocks.append(_plink_pca_block(maf, geno, ld, pca_components))
        blocks.append(_admixture_block(k_min, k_max, admixture_seeds))
    if "treemix" in selected or "gene_flow" in selected:
        blocks.append(_treemix_block(treemix_root, treemix_max_m, treemix_block_size))
    if "iqtree" in selected or "tree" in selected or "ml_tree" in selected:
        blocks.append(_iqtree_block(alignment, "ml"))
    if "vcf2dis" in selected or "p_distance" in selected or "distance" in selected:
        blocks.append(_vcf2dis_block())

    return {
        "purpose": "Population-genomics workflow plan from VCF/PLINK-style inputs.",
        "inputs": {
            "vcf": vcf,
            "popmap": popmap,
            "workdir": workdir,
            "analyses": selected,
        },
        "parameters": {
            "maf": maf,
            "geno": geno,
            "ld_pruning": ld,
            "window": window,
            "step": step,
            "pca_components": pca_components,
            "admixture_k": [k_min, k_max],
            "admixture_seeds": admixture_seeds,
            "treemix_root": treemix_root,
            "treemix_max_m": treemix_max_m,
            "treemix_block_size": treemix_block_size,
        },
        "blocks": [block.as_dict() for block in blocks],
        "method_note": method_note(selected, maf, geno, ld, window, step),
    }


def method_note(analyses: list[str], maf: float, geno: float, ld: str, window: int, step: int) -> str:
    names = ", ".join(analyses)
    return (
        "Population-genomic analyses were planned from VCF genotype data using "
        f"explicit sample-map validation and QC filters (MAF >= {maf}, missingness <= {geno}). "
        f"LD-sensitive analyses used PLINK pruning parameters `{ld}`. "
        f"Windowed diversity statistics used {window} bp windows with {step} bp step where applicable. "
        f"Planned endpoints: {names}. Tool versions and all command parameters should be recorded at runtime."
    )
