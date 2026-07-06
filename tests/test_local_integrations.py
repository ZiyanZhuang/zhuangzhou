from zhuangzhou.synbio.assembor import (
    query_yeast_parts,
    select_yeast_expression_parts,
    validate_yeast_parts,
)
from zhuangzhou.CYPamber.cypforge_bridge import cypforge_available


def test_assembor_yeast_parts_validate_and_select():
    report = validate_yeast_parts()
    assert report["valid"] is True
    assert any(line.startswith("rows=") for line in report["info"])

    promoters = query_yeast_parts(part_type="promoter", strength_bin="strong")
    assert promoters
    assert all(part["type"] == "promoter" for part in promoters)

    design = select_yeast_expression_parts("strong", "standard")
    assert design["promoters"]
    assert design["terminators"]


def test_cypforge_bridge_reports_workflow_modules():
    status = cypforge_available()
    assert status["available"] is True
    assert status["module_count"] >= 1
    assert status["modules"][0]["skill_id"].startswith("cypforge.")
