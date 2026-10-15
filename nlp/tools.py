"""
Skill-to-Tool Converter -- Bridge between SkillRegistry and LLM tool_use.

Converts Zhuangzhou's Skill definitions into LLM-consumable tool schemas
(Anthropic format). OpenAI format conversion happens in the backend layer.
"""

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

# Map SkillParam.type to JSON Schema type
_TYPE_MAP = {
    'path': 'string',
    'string': 'string',
    'int': 'integer',
    'float': 'number',
    'bool': 'boolean',
    'list': 'array',
    'set': 'array',
    'dict': 'object',
    'json': ['string', 'object', 'array'],
}


def skill_to_tool(skill) -> dict:
    """Convert a single Skill to Anthropic tool schema."""
    properties = {}
    required = []

    for p in skill.inputs:
        json_type = _TYPE_MAP.get(p.type, 'string')
        prop = {'type': json_type}
        if json_type == 'array' or isinstance(json_type, list) and 'array' in json_type:
            prop['items'] = {}  # Lists may hold structured records or numeric coordinates.
        if p.description:
            prop['description'] = p.description
        if p.default is not None:
            prop['default'] = p.default
        if p.choices:
            prop['enum'] = p.choices
        properties[p.name] = prop

        if p.required:
            required.append(p.name)

    schema = {
        'type': 'object',
        'properties': properties,
    }
    if required:
        schema['required'] = required

    # Add hints about requirements and risk
    desc = skill.description
    tags = []
    if skill.requires_gpu:
        tags.append('GPU')
    if skill.requires_wsl:
        tags.append('WSL')
    if getattr(skill, 'remote_capable', False):
        tags.append('REMOTE')
    if skill.risk_level == 'high':
        tags.append('HIGH-RISK')
    if skill.estimated_time:
        tags.append(skill.estimated_time)
    if tags:
        desc += f" [{', '.join(tags)}]"

    return {
        'name': skill.name,
        'description': desc,
        'input_schema': schema,
    }


def registry_to_tools(registry) -> List[dict]:
    """Convert all skills in a SkillRegistry to tool schemas."""
    tools = []
    for skill in registry.list_all():
        # Skip skills explicitly marked as unavailable
        if skill.available is False:
            continue
        try:
            tools.append(skill_to_tool(skill))
        except Exception as e:
            logger.warning("Failed to convert skill %s: %s",
                           skill.name, e)
    return tools


def format_skill_catalog(registry) -> str:
    """Format skill catalog as a readable string for system prompt."""
    lines = []
    for cat in sorted(registry.categories()):
        skills = registry.list_by_category(cat)
        lines.append(f"\n## {cat.upper()} ({len(skills)} skills)")
        for s in skills:
            tags = []
            if s.requires_gpu:
                tags.append('GPU')
            if s.requires_wsl:
                tags.append('WSL')
            if getattr(s, 'remote_capable', False):
                tags.append('REMOTE')
            if s.risk_level == 'high':
                tags.append('HIGH-RISK')
            if s.estimated_time:
                tags.append(s.estimated_time)
            tag_str = f" [{','.join(tags)}]" if tags else ""
            params = ', '.join(
                f"{p.name}:{p.type}" + ("?" if not p.required else "")
                for p in s.inputs
            )
            lines.append(f"  - {s.name}({params}){tag_str}")
            lines.append(f"    {s.description}")

    return '\n'.join(lines)
