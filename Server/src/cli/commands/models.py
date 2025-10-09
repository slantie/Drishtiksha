# src/cli/commands/models.py

"""Model management command."""

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from src.cli.config import get_cli_settings
from src.ml.registry import ModelManager
from src.cli.utils.banner import print_section_header

console = Console()


@click.command()
@click.option(
    '--verbose', '-v',
    is_flag=True,
    help='Show detailed model information.'
)
@click.option(
    '--type', '-t',
    type=click.Choice(['video', 'audio', 'image', 'all'], case_sensitive=False),
    default='all',
    help='Filter models by type.'
)
def models(verbose: bool, type: str):
    """
    List available models and their details.
    
    Shows all configured models, their types, descriptions, and availability.
    
    \b
    Examples:
      drishtiksha models
      drishtiksha models --verbose
      drishtiksha models --type video
      drishtiksha models --type audio
    """
    try:
        print_section_header("🤖 Available Models")
        
        # Get CLI settings (includes ALL models) and model manager
        cli_settings = get_cli_settings()
        model_manager = ModelManager(cli_settings)
        
        # Filter models by type
        models_list = []
        for model_name, config in model_manager.model_configs.items():
            if type == 'all':
                models_list.append((model_name, config))
            elif type == 'video' and config.isVideo:
                models_list.append((model_name, config))
            elif type == 'audio' and config.isAudio:
                models_list.append((model_name, config))
            elif type == 'image' and config.isImage:
                models_list.append((model_name, config))
        
        if not models_list:
            console.print(f"[yellow]No {type} models found.[/yellow]")
            return
        
        # Display models
        if verbose:
            _display_models_detailed(models_list)
        else:
            _display_models_simple(models_list)
        
        console.print(f"\n[dim]Total: {len(models_list)} model(s)[/dim]")
        
    except Exception as e:
        console.print(f"\n[bold red]❌ Error:[/bold red] {str(e)}")
        raise click.Abort()


def _display_models_simple(models_list):
    """Display models in simple table format."""
    table = Table(show_header=True, box=box.ROUNDED, border_style="cyan")
    table.add_column("Model Name", style="cyan", width=30)
    table.add_column("Type", justify="center", width=10)
    table.add_column("Description", style="dim", width=50)
    
    for model_name, config in models_list:
        # Determine type
        if config.isVideo:
            type_icon = "🎥"
        elif config.isAudio:
            type_icon = "🎵"
        elif config.isImage:
            type_icon = "🖼️"
        else:
            type_icon = "❓"
        
        # Truncate description if too long
        description = config.description
        if len(description) > 48:
            description = description[:45] + "..."
        
        table.add_row(
            model_name,
            type_icon,
            description
        )
    
    console.print(table)


def _display_models_detailed(models_list):
    """Display models with detailed information."""
    for idx, (model_name, config) in enumerate(models_list):
        if idx > 0:
            console.print()
        
        # Determine type
        if config.isVideo:
            type_str = "🎥 Video Model"
            color = "cyan"
        elif config.isAudio:
            type_str = "🎵 Audio Model"
            color = "magenta"
        elif config.isImage:
            type_str = "🖼️  Image Model"
            color = "yellow"
        else:
            type_str = "❓ Unknown Type"
            color = "white"
        
        # Create info table
        info_table = Table(show_header=False, box=box.SIMPLE)
        info_table.add_column(style="cyan", width=20)
        info_table.add_column(style="white")
        
        info_table.add_row("Name", f"[bold]{model_name}[/bold]")
        info_table.add_row("Type", type_str)
        info_table.add_row("Class", config.class_name)
        info_table.add_row("Description", config.description)
        info_table.add_row("Model Path", str(config.model_path))
        info_table.add_row("Device", config.device if hasattr(config, 'device') else "N/A")
        
        # Additional config info
        if hasattr(config, 'video_config') and config.video_config:
            info_table.add_row("", "")
            info_table.add_row("[bold]Video Config", "")
            for key, value in config.video_config.items():
                info_table.add_row(f"  {key}", str(value))
        
        console.print(Panel(
            info_table,
            border_style=color,
            box=box.ROUNDED
        ))
