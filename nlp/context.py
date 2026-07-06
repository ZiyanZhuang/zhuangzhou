"""
Zhuangzhou Context Engineering Engine
=====================================
Centralized manager for "Pre-baked" knowledge and workspace awareness.
Provides the 'Context Library' used for manual tool-calling models.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class ContextManager:
    """Singleton manager for workspace snapshot, project constitution, and skill manuals."""
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ContextManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, config: dict = None):
        if hasattr(self, '_initialized'): return
        self.config = config or {}
        self._cache = {}
        self._initialized = True
        # Cold start scan
        self.refresh()

    def refresh(self):
        """Perform a full scan of the workspace environment."""
        self._cache['files'] = self._scan_workspace()
        self._cache['constitution'] = self._read_constitution()

    def _scan_workspace(self, limit: int = 50) -> str:
        """Scan CWD for relevant scientific files."""
        cwd = os.getcwd()
        try:
            files = os.listdir(cwd)
            # Prioritize scientific extensions
            sci_exts = {'.pdb', '.sdf', '.mol2', '.itp', '.top', '.mdp', '.xtc', '.trr', '.log', '.out'}
            
            sci_files = [f for f in files if any(f.lower().endswith(ext) for ext in sci_exts)]
            other_files = [f for f in files if f not in sci_files and os.path.isfile(f)]
            
            combined = (sci_files + other_files)[:limit]
            res = ", ".join(combined)
            if len(files) > limit:
                res += " ...(truncated)"
            return res
        except Exception as e:
            return f"(Error scanning directory: {e})"

    def _read_constitution(self) -> str:
        """Read ZHUANGZHOU.md if it exists."""
        md_path = os.path.join(os.getcwd(), 'ZHUANGZHOU.md')
        if os.path.exists(md_path):
            try:
                with open(md_path, 'r', encoding='utf-8', errors='replace') as f:
                    return f.read().strip()
            except Exception as e:
                import logging
                logging.getLogger(__name__).debug("Failed to read ZHUANGZHOU.md: %s", e)
        return ""

    def get_skill_manual(self, registry, loaded_categories: set) -> str:
        """Generate a compact text manual for all registered skills.
        Used for models that don't support native tool calling.
        """
        manual_lines = []
        for skill in registry.list_all():
            if skill.category not in loaded_categories:
                continue
            
            # Ultra-compact format: name(param:type, ...) - description
            params = []
            for p in skill.inputs:
                req = "*" if p.required else ""
                params.append(f"{p.name}{req}:{p.type}")
            
            line = f"- {skill.name}({', '.join(params)}): {skill.description}"
            manual_lines.append(line)
            
        return "\n".join(manual_lines)

    def get_snapshot(self) -> Dict[str, str]:
        return {
            'cwd': os.getcwd(),
            'files': self._cache.get('files', ''),
            'constitution': self._cache.get('constitution', '')
        }

FEW_SHOT_REACT = """
## 示例 1: 结构获取与分析
User: "下载蛋白 1AKI 并分析其配体。"
Thought: 我需要先下载 1AKI 的 PDB 文件，然后使用 PLIP 分析其中的相互作用。
Action: {"name": "get_pdb_entry", "input": {"pdb_id": "1AKI"}}
Observation: {"status": "success", "pdb_file": "1AKI.pdb", "header": "LYSOZYME"}
Thought: 结构已下载。现在我将调用 plip_analyze 来检测其中的非标准残基和相互作用。
Action: {"name": "plip_analyze", "input": {"pdb_file": "1AKI.pdb"}}
Observation: {"interactions": [...], "ligands": ["NAG"]}
Thought: 分析完成。1AKI 是溶菌酶，包含配体 NAG。

## 示例 2: 错误恢复
User: "对 V21L 突变体运行 MD。"
Thought: 我需要先加载 pdb 文件。当前目录下有 protein.pdb。
Action: {"name": "mutate_minimize", "input": {"pdb_file": "protein.pdb", "mutation": "V21L"}}
Observation: {"error": "File 'protein.pdb' not found."}
Thought: 报错显示找不到 protein.pdb。我检查文件列表，发现实际文件名是 '6m0j.pdb'。我将修正文件名重试。
Action: {"name": "mutate_minimize", "input": {"pdb_file": "6m0j.pdb", "mutation": "V21L"}}
"""

# Global context manager
context_manager = ContextManager()
