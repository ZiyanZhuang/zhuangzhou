from zhuangzhou.genomics import build_popgen_workflow, input_contract_commands, method_note
from zhuangzhou.skills.registry import SkillRegistry


def test_popgen_workflow_includes_requested_blocks():
    plan = build_popgen_workflow(
        analyses=["diversity", "fst", "pca", "admixture", "treemix", "iqtree", "vcf2dis"],
        vcf="cohort.vcf.gz",
        popmap="pop_map.txt",
        workdir="popgen_work",
        threads=4,
    )

    names = [block["name"] for block in plan["blocks"]]
    assert names == [
        "setup_and_input_contract",
        "diversity",
        "fst_pairwise",
        "plink_pca",
        "admixture",
        "treemix",
        "iqtree",
        "vcf2dis",
    ]
    assert plan["inputs"]["vcf"] == "cohort.vcf.gz"
    assert "MAF >= 0.05" in plan["method_note"]
    assert "LD-sensitive analyses used PLINK pruning parameters" in plan["method_note"]


def test_popgen_plan_preserves_input_paths_after_workdir_change():
    plan = build_popgen_workflow(vcf="cohort.vcf.gz", popmap="pop_map.txt", workdir="popgen_work")
    setup_commands = "\n".join(plan["blocks"][0]["commands"])

    assert 'INPUT_ROOT="$PWD"' in setup_commands
    assert 'cd "$WORKDIR"' in setup_commands
    assert 'VCF="$INPUT_ROOT/$VCF"' in setup_commands
    assert 'POPMAP="$INPUT_ROOT/$POPMAP"' in setup_commands


def test_treemix_block_runs_in_subshell_without_polluting_later_blocks():
    plan = build_popgen_workflow(analyses=["treemix", "iqtree", "vcf2dis"])
    blocks = {block["name"]: block for block in plan["blocks"]}
    treemix_commands = "\n".join(blocks["treemix"]["commands"])

    assert "( cd treemix_run" in treemix_commands
    assert treemix_commands.rstrip().endswith(")")
    assert "treemix_run/likelihood.tsv" in blocks["treemix"]["validation"][0]
    assert blocks["iqtree"]["commands"][2].startswith('iqtree -s "snp.phy"')
    assert blocks["vcf2dis"]["commands"][0] == 'VCF2Dis_multi -InPut "$VCF" -OutPut p_distance.mat'


def test_popgen_input_contract_documents_sample_validation():
    commands = "\n".join(input_contract_commands("cohort.vcf.gz", "pop_map.txt"))

    assert "diff -u vcf.samples.sorted popmap.samples.sorted" in commands
    assert "populations.txt" in commands
    assert "cohort.vcf.gz" in commands
    assert "pop_map.txt" in commands


def test_method_note_records_qc_and_window_parameters():
    note = method_note(["diversity", "pca"], maf=0.01, geno=0.2, ld="100 20 0.1", window=50000, step=10000)

    assert "MAF >= 0.01" in note
    assert "missingness <= 0.2" in note
    assert "`100 20 0.1`" in note
    assert "50000 bp windows with 10000 bp step" in note


def test_genomics_skills_are_registered():
    registry = SkillRegistry()

    workflow = registry.get("popgen_workflow_plan")
    contract = registry.get("popgen_input_contract")

    assert workflow is not None
    assert contract is not None
    assert workflow.category == "genomics"
    assert contract.category == "genomics"
    result = workflow.execute(analyses=["admixture"], vcf="cohort.vcf.gz", popmap="pop_map.txt", admixture_seeds=[11, 13])
    assert result["blocks"][0]["name"] == "setup_and_input_contract"
    assert result["parameters"]["admixture_seeds"] == [11, 13]
    assert any(block["name"] == "admixture" for block in result["blocks"])
