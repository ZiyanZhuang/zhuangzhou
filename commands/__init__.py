"""Zhuangzhou CLI command modules."""

from .run import RunCommand
from .analyze import AnalyzeCommand
from .tools import ToolsCommand
from .scan import ScanCommand
from .dock import DockCommand
from .search import SearchCommand
from .config import ConfigCommand
from .status import StatusCommand

COMMANDS = {
    'run': RunCommand,
    'analyze': AnalyzeCommand,
    'tools': ToolsCommand,
    'scan': ScanCommand,
    'dock': DockCommand,
    'search': SearchCommand,
    'config': ConfigCommand,
    'status': StatusCommand,
}

__all__ = ['COMMANDS']
