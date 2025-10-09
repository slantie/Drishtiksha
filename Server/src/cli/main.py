#!/usr/bin/env python3
# src/cli/main.py

"""
Drishtiksha CLI - Main entry point

A professional, interactive command-line interface for deepfake detection.
Built with Click, Rich, and best practices from industry-leading CLIs.
"""

import sys
import click
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich import box

from src.cli.commands import analyze, batch, interactive, stats, models, history
from src.cli.utils.banner import print_banner
from src.cli.utils.validators import validate_environment

console = Console()


class DrishtikshaCLI(click.Group):
    """Custom Click group with enhanced help formatting."""
    
    def format_help(self, ctx, formatter):
        """Custom help formatter with rich formatting."""
        print_banner()
        console.print()
        super().format_help(ctx, formatter)


@click.group(cls=DrishtikshaCLI, invoke_without_command=True)
@click.version_option(version="3.0.0", prog_name="Drishtiksha CLI")
@click.option('--debug', is_flag=True, help='Enable debug mode with verbose logging.')
@click.pass_context
def cli(ctx, debug):
    """
    [bold cyan]Drishtiksha[/bold cyan] - Professional Deepfake Detection CLI
    
    Analyze videos, images, and audio files for deepfake content using
    state-of-the-art machine learning models.
    
    [bold]Quick Start:[/bold]
    
      [cyan]# Analyze a single file[/cyan]
      $ drishtiksha analyze video.mp4
      
      [cyan]# Interactive mode (recommended for beginners)[/cyan]
      $ drishtiksha interactive
      
      [cyan]# Batch analyze multiple files[/cyan]
      $ drishtiksha batch ./media_folder/
    
    [bold]Examples:[/bold]
    
      [dim]# Analyze with specific model[/dim]
      $ drishtiksha analyze video.mp4 --model EFFICIENTNET-B7-V1
      
      [dim]# Generate visualization[/dim]
      $ drishtiksha analyze video.mp4 --visualize
      
      [dim]# Export results to JSON[/dim]
      $ drishtiksha analyze video.mp4 --output results.json
      
      [dim]# Check system status[/dim]
      $ drishtiksha stats
    
    Use [bold]drishtiksha COMMAND --help[/bold] for detailed command information.
    """
    ctx.ensure_object(dict)
    ctx.obj['DEBUG'] = debug
    
    # If no command specified, show help
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


# Register commands
cli.add_command(analyze.analyze)
cli.add_command(batch.batch)
cli.add_command(interactive.interactive)
cli.add_command(stats.stats)
cli.add_command(models.models)
cli.add_command(history.history)


def main():
    """Main entry point for the CLI."""
    try:
        # Validate environment before running (commented out for now to avoid issues)
        # validate_environment()
        cli(obj={})
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Operation cancelled by user.[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[bold red]❌ Fatal Error:[/bold red] {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
