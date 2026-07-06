"""Base command class for Zhuangzhou CLI."""

from rich.console import Console

console = Console()


class BaseCommand:
    """Base class for all CLI commands."""

    name: str = ""
    description: str = ""
    usage: str = ""
    aliases: list = []

    def __init__(self, app):
        self.app = app
        self.console = app.console
        self.config = app.config

    def execute(self, args: list):
        raise NotImplementedError

    def help(self):
        self.console.print(f"[bold bright_cyan]{self.name}[/] - {self.description}")
        if self.usage:
            self.console.print(f"  [dim]Usage: {self.usage}[/]")
        if self.aliases:
            self.console.print(f"  [dim]Aliases: {', '.join(self.aliases)}[/]")
