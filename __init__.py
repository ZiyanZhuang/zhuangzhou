"""
Zhuangzhou (搴勫懆) 鈥?Rational Enzyme Engineering Framework
==========================================================

"搴勫懆姊﹁澏" 鈥?The butterfly dream.

Zhuangzhou is the unified framework for computational enzyme engineering.
It integrates:
  - Skills:    All computational tools (PLM, docking, MD, analysis) as composable skills
  - Workflows: JSON-driven pipeline orchestration
  - Memory:    Persistent journal, context, and learned knowledge
  - Brain:     BeTaThink KG/RAG/Agent as the knowledge backend

Architecture (first principles):
  LLM = brain capacity (upper/lower bound)
  Agent = environment that feeds information to the brain
  Skills = tools the agent can use
  Workflows = predefined sequences of skills
  Memory = episodic (journal) + working (context) + semantic (knowledge)
  Brain = BeTaThink KG + RAG + multi-agent reasoning

Core modules:
  1. skills/     - Skill registry (PLM, structure, docking, simulation, chemistry, viz, brain, unimol)
  2. workflow/   - JSON workflow engine with templates
  3. memory/     - Journal, SessionContext, KnowledgeStore
  4. brain/      - BeTaThink API + GraphRAG parquet + offline fallback
  5. envs/       - Environment isolation (conda graphrag, dplm2, WSL, native)
  6. conversation.py - NLP conversation engine (LLM agent runtime-inspired agentic loop)
  7. agent.py    - Unified agent interface (LLM agent runtime tools + API + chat)
  8. commands/   - CLI command modules
  9. core.py     - 5-phase pipeline engine
  10. chemical_intuition.py - First-principles mutation analysis
  11. app.py     - Interactive REPL (Gemini CLI style)
"""

__version__ = "0.8.6"

