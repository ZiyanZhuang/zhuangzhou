"""
Report Configuration
====================
Style constants for scientific Word document generation.
Follows Nature/JACS formatting conventions.
"""

from docx.shared import Pt, Cm, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

# ── Page Layout ──
PAGE_WIDTH = Cm(21.0)     # A4
PAGE_HEIGHT = Cm(29.7)
MARGIN_TOP = Cm(2.54)
MARGIN_BOTTOM = Cm(2.54)
MARGIN_LEFT = Cm(3.17)
MARGIN_RIGHT = Cm(3.17)

# ── Fonts ──
FONT_TITLE = 'Arial'
FONT_BODY = 'Times New Roman'
FONT_CODE = 'Consolas'
FONT_CN = 'SimSun'        # 中文宋体

# ── Font Sizes ──
SIZE_TITLE = Pt(16)
SIZE_HEADING1 = Pt(14)
SIZE_HEADING2 = Pt(12)
SIZE_HEADING3 = Pt(11)
SIZE_BODY = Pt(10.5)
SIZE_CAPTION = Pt(9)
SIZE_TABLE = Pt(9)
SIZE_FOOTER = Pt(8)

# ── Colors ──
COLOR_PRIMARY = RGBColor(0x1A, 0x1A, 0x2E)    # Dark navy
COLOR_ACCENT = RGBColor(0x2E, 0x86, 0xC1)     # Scientific blue
COLOR_HEADER_BG = RGBColor(0x2E, 0x86, 0xC1)  # Table header background
COLOR_HEADER_FG = RGBColor(0xFF, 0xFF, 0xFF)   # Table header text
COLOR_ALT_ROW = RGBColor(0xF2, 0xF7, 0xFC)    # Alternating row
COLOR_CAPTION = RGBColor(0x55, 0x55, 0x55)     # Figure/table caption
COLOR_RED = RGBColor(0xC0, 0x39, 0x2B)         # Bad / destabilizing
COLOR_GREEN = RGBColor(0x27, 0xAE, 0x60)       # Good / stabilizing

# ── Spacing ──
PARA_SPACING_BEFORE = Pt(3)
PARA_SPACING_AFTER = Pt(6)
LINE_SPACING = 1.15

# ── Figure ──
FIGURE_WIDTH = Cm(14.0)           # Default figure width
FIGURE_WIDTH_HALF = Cm(7.0)      # Half-width (side-by-side)
FIGURE_WIDTH_FULL = Cm(16.0)     # Full-width

# ── Report Sections (ordered) ──
MUTATION_REPORT_SECTIONS = [
    'title_page',
    'abstract',
    'introduction',
    'methods',
    'results_structure',
    'results_mutations',
    'results_docking',
    'results_interactions',
    'results_dynamics',
    'discussion',
    'conclusion',
    'references',
    'appendix',
]
