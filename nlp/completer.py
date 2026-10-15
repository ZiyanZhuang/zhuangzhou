"""
Zhuangzhou Slash Command Completer
==================================
Provides real-time autocompletion for:
  - Slash commands (/help, /skills, /model, /plan, ...)
  - Skill names after slash (/esm2_ → /esm2_mutation_scan)
  - Model names after /model
"""

from prompt_toolkit.completion import Completer, Completion
from zhuangzhou.nlp.slash import manager as slash_manager
from zhuangzhou.nlp.backends import list_available_models

# Lazy-loaded skill name cache
_skill_cache = None
_skill_cache_loaded = False


def _get_skill_names():
    """Lazily load and cache skill names + descriptions for completion."""
    global _skill_cache, _skill_cache_loaded
    if _skill_cache_loaded:
        return _skill_cache
    _skill_cache_loaded = True
    try:
        from zhuangzhou.skills.registry import get_registry
        registry = get_registry()
        _skill_cache = [
            (s.name, f"[{s.category}] {s.description[:60]}")
            for s in registry.list_all()
        ]
    except Exception:
        _skill_cache = []
    return _skill_cache


class ZhuangzhouCompleter(Completer):
    """Custom completer for Zhuangzhou slash commands and skill names."""

    def __init__(self, commands=None):
        self.commands = dict(commands or {})

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor

        # 1. Triggered by '/' at the start
        if text.startswith('/'):
            parts = text.split()

            # Case A: Just '/cmd' (completing the trigger itself)
            if len(parts) <= 1 and not text.endswith(' '):
                trigger_fragment = text.lower()

                # A1. Built-in slash commands first
                commands = {**self.commands, **{trigger: cmd.description
                            for trigger, cmd in slash_manager.commands.items()}}
                for trigger, description in commands.items():
                    if trigger.startswith(trigger_fragment):
                        yield Completion(
                            trigger,
                            start_position=-len(text),
                            display_meta=description
                        )

                # A2. Skill names (without /) — match by first letters after /
                skill_fragment = trigger_fragment[1:]  # strip leading /
                if skill_fragment:
                    for name, desc in _get_skill_names():
                        if name.startswith(skill_fragment):
                            yield Completion(
                                '/' + name,
                                start_position=-len(text),
                                display_meta=desc
                            )

            # Case B: Specific command arguments (e.g., /model )
            elif len(parts) >= 1:
                cmd_trigger = parts[0].lower()

                if cmd_trigger == '/model':
                    # Only complete models if the user has typed space after /model
                    if text.endswith(' ') or len(parts) > 1:
                        models = list_available_models()
                        # Get the current fragment after '/model '
                        fragment = parts[1] if len(parts) > 1 else ""

                        for m in models:
                            m_name = m['name']
                            if m_name.startswith(fragment):
                                yield Completion(
                                    m_name,
                                    start_position=-len(fragment),
                                    display_meta=f"{m['type'].upper()} ({m['size']})"
                                )

        # We don't complete normal text to avoid being intrusive
        return
