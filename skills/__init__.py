"""
Zhuangzhou Skills System
========================
Every computational tool is a "skill" — a self-contained unit with:
  - name, description, category
  - input/output schema
  - execute() method
  - availability check

Skills are auto-discovered and registered. Workflows compose skills.
"""

from .registry import SkillRegistry, Skill, get_registry
from . import unified  # Consolidated skill wrappers

__all__ = ['SkillRegistry', 'Skill', 'get_registry', 'unified']
