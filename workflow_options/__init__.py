"""
Workflow Options — User-Selectable Multi-Step Experimental Workflows
====================================================================
Zhuangzhou v0.9.2

Curated, literature-backed experimental workflows that users can browse,
select, and execute through the NLP agent. Each workflow is a structured
multi-step pipeline mapping to existing Zhuangzhou skills.

Available Workflows:
  1. pareto_active_learning — Robustness-Enhanced Enzyme Engineering
     (Pareto fitness+diversity library design + active learning DBTL)

Usage:
  from zhuangzhou.workflow_options import get_workflow_registry
  registry = get_workflow_registry()
  wf = registry.get('pareto_active_learning')
  wf.describe()           # Human-readable summary
  wf.execute(agent, ...)  # Run via agent
"""

from .registry import WorkflowRegistry, get_workflow_registry, list_workflows
from .harness import WorkflowHarness, HarnessState

__all__ = ['WorkflowRegistry', 'get_workflow_registry', 'list_workflows',
           'WorkflowHarness', 'HarnessState']
