# src/cli/utils/banner.py

"""Beautiful ASCII banner and branding for the CLI."""

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich import box

console = Console()


BANNER_ASCII = """
╔═══════════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                               ║
║   ██████╗  ██████╗  ██╗ ███████╗ ██╗  ██╗ ████████╗ ██╗ ██╗  ██╗ ███████╗ ██╗  ██╗  █████╗    ║
║   ██╔══██╗ ██╔══██╗ ██║ ██╔════╝ ██║  ██║ ╚══██╔══╝ ██║ ██║ ██╔╝ ██╔════╝ ██║  ██║ ██╔══██╗   ║
║   ██║  ██║ ██████╔╝ ██║ ███████╗ ███████║    ██║    ██║ █████╔╝  ███████╗ ███████║ ███████║   ║
║   ██║  ██║ ██╔══██╗ ██║ ╚════██║ ██╔══██║    ██║    ██║ ██╔═██╗  ╚════██║ ██╔══██║ ██╔══██║   ║
║   ██████╔╝ ██║  ██║ ██║ ███████║ ██║  ██║    ██║    ██║ ██║  ██╗ ███████║ ██║  ██║ ██║  ██║   ║
║   ╚═════╝  ╚═╝  ╚═╝ ╚═╝ ╚══════╝ ╚═╝  ╚═╝    ╚═╝    ╚═╝ ╚═╝  ╚═╝ ╚══════╝ ╚═╝  ╚═╝ ╚═╝  ╚═╝   ║
║                                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════════════════════╝
"""


def print_banner():
    """Print the Drishtiksha CLI banner."""
    console.print(BANNER_ASCII, style="bold cyan", highlight=False)
    console.print(
        f"Version 3.0.0  •  Powered by AI\n",
        justify="leftcenter"
    )


def print_welcome():
    """Print welcome message for interactive mode."""
    print_banner()
    
    welcome_text = Text()
    welcome_text.append("Welcome to ", style="white")
    welcome_text.append("Drishtiksha", style="bold cyan")
    welcome_text.append(" - Your AI-powered deepfake detection assistant!\n\n", style="white")
    welcome_text.append("I'll guide you through analyzing your media files step-by-step.\n", style="dim")
    welcome_text.append("You can exit anytime by pressing ", style="dim")
    welcome_text.append("Ctrl+C", style="bold yellow")
    welcome_text.append(".", style="dim")
    
    console.print(Panel(
        welcome_text,
        title="[bold cyan]🤖 Interactive Mode[/bold cyan]",
        border_style="cyan",
        box=box.ROUNDED
    ))
    console.print()


def print_section_header(title: str, emoji: str = "📋"):
    """Print a styled section header."""
    console.print(f"\n[bold cyan]{emoji} {title}[/bold cyan]")
    console.print("[dim]" + "─" * 60 + "[/dim]")
