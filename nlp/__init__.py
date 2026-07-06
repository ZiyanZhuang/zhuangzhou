"""
Zhuangzhou NLP Layer -- Natural Language Control System
========================================================

The brain of Zhuangzhou. Turns natural language into computational actions.

Architecture:
    User (natural language)
         |
    ZhuangzhouAgent
         |-- LLM Backend (Claude / DeepSeek / Ollama / ...)
         |-- Skill Registry (56 computational skills)
         |-- Memory System (5-layer accumulating knowledge)
         |
    [1] User speaks: "帮我对2CPP做10ns金属蛋白MD"
    [2] Agent builds context (memory + skill catalog)
    [3] LLM reasons: which skills? what params?
    [4] Agent executes: run_metalloprotein_md(pdb='2CPP', time=10ns)
    [5] Agent narrates results back in natural language
    [6] Memory archives the interaction

Key insight: LLM is a pluggable reasoning engine, not the boss.
Zhuangzhou IS the system. LLM IS a replaceable part.
"""

from zhuangzhou.nlp.agent import ZhuangzhouAgent, chat, create_agent

__all__ = ['ZhuangzhouAgent', 'chat', 'create_agent']
