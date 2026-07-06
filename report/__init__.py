"""
Report Generation Module
=========================
Generates publication-quality Word (.docx) reports from computational results.

Usage:
    from zhuangzhou.report import build_mutation_report, build_docking_report

    # From aggregated pipeline results
    docx_path = build_mutation_report(results_dict, output_path='report.docx')

    # Or via skill system
    from zhuangzhou.skills import get_registry
    reg = get_registry()
    result = reg.get('generate_mutation_report').execute(results=..., output_path=...)
"""

from .docx_builder import DocxBuilder
from .templates import (
    build_mutation_report,
    build_docking_report,
    build_md_report,
    build_qm_report,
    build_general_report,
    build_enriched_docking_report,
    build_synbio_report,
)

__all__ = [
    'DocxBuilder',
    'build_mutation_report',
    'build_docking_report',
    'build_md_report',
    'build_qm_report',
    'build_general_report',
    'build_enriched_docking_report',
    'build_synbio_report',
]
