# src/cli/commands/stats.py

"""System statistics and health check command."""

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
import psutil
import platform

from src.cli.config import get_cli_settings
from src.ml.registry import ModelManager
from src.cli.utils.banner import print_section_header

console = Console()


@click.command()
@click.option(
    '--detailed', '-d',
    is_flag=True,
    help='Show detailed system information.'
)
def stats(detailed: bool):
    """
    Display system and model statistics.
    
    Shows loaded models, GPU/CPU info, memory usage, and system health.
    
    \b
    Examples:
      drishtiksha stats
      drishtiksha stats --detailed
    """
    try:
        print_section_header("📊 System Statistics", "💻")
        
        # Get CLI settings (includes ALL models) and model manager
        cli_settings = get_cli_settings()
        model_manager = ModelManager(cli_settings)
        
        # Don't load models - just show stats about configuration
        console.print()
        
        # Display system info
        _display_system_info(cli_settings, detailed)
        
        console.print()
        
        # Display model info
        _display_model_info(model_manager)
        
        console.print()
        
        # Display resource usage
        _display_resource_usage()
        
    except Exception as e:
        console.print(f"\n[bold red]❌ Error:[/bold red] {str(e)}")
        raise click.Abort()


def _display_system_info(cli_settings, detailed: bool):
    """Display system information."""
    import torch
    
    table = Table(show_header=False, box=box.ROUNDED, border_style="cyan")
    table.add_column(style="cyan", width=25)
    table.add_column(style="white")
    
    table.add_row("Python Version", platform.python_version())
    table.add_row("Platform", platform.platform())
    table.add_row("Processor", platform.processor())
    
    # Device info
    device = cli_settings.device
    if device == "cuda" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        table.add_row("Device", f"CUDA ({gpu_name})")
        table.add_row("GPU Memory", f"{gpu_memory:.1f} GB")
    else:
        table.add_row("Device", "CPU")
    
    table.add_row("PyTorch Version", torch.__version__)
    table.add_row("CUDA Available", "Yes" if torch.cuda.is_available() else "No")
    
    if detailed:
        table.add_row("CPU Cores", str(psutil.cpu_count(logical=True)))
        table.add_row("RAM Total", f"{psutil.virtual_memory().total / (1024**3):.1f} GB")
    
    console.print(Panel(
        table,
        title="[bold cyan]🖥️  System Information[/bold cyan]",
        border_style="cyan"
    ))


def _display_model_info(model_manager: ModelManager):
    """Display configured models information (without loading them)."""
    table = Table(show_header=True, box=box.ROUNDED, border_style="cyan")
    table.add_column("Model", style="cyan", width=30)
    table.add_column("Type", justify="center", width=10)
    table.add_column("Device", justify="center", width=10)
    table.add_column("Status", justify="center", width=15)
    
    for model_name, config in model_manager.model_configs.items():
        # Determine type icon
        if config.isVideo:
            type_icon = "🎥 Video"
        elif config.isAudio:
            type_icon = "🎵 Audio"
        elif config.isImage:
            type_icon = "🖼️  Image"
        else:
            type_icon = "❓ Unknown"
        
        # Get model device
        device = config.device if hasattr(config, 'device') else "N/A"
        
        # Check if loaded in cache (don't trigger loading)
        if model_manager.is_model_loaded(model_name):
            status = "[green]✓ Loaded[/green]"
        else:
            status = "[dim]⚡ Ready (lazy)[/dim]"
        
        table.add_row(
            model_name,
            type_icon,
            device,
            status
        )
    
    console.print(Panel(
        table,
        title=f"[bold cyan]🤖 Models ({len(model_manager.model_configs)})[/bold cyan]",
        border_style="cyan"
    ))


def _display_resource_usage():
    """Display current resource usage."""
    import torch
    
    table = Table(show_header=False, box=box.ROUNDED, border_style="cyan")
    table.add_column(style="cyan", width=25)
    table.add_column(style="white", justify="right")
    table.add_column(style="white", width=30)
    
    # CPU usage
    cpu_percent = psutil.cpu_percent(interval=1)
    cpu_bar = _create_bar(cpu_percent)
    table.add_row("CPU Usage", f"{cpu_percent:.1f}%", cpu_bar)
    
    # RAM usage
    ram = psutil.virtual_memory()
    ram_percent = ram.percent
    ram_bar = _create_bar(ram_percent)
    table.add_row(
        "RAM Usage",
        f"{ram.used / (1024**3):.1f} / {ram.total / (1024**3):.1f} GB",
        ram_bar
    )
    
    # GPU usage (if available)
    if torch.cuda.is_available():
        gpu_memory_allocated = torch.cuda.memory_allocated(0) / (1024**3)
        gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        gpu_percent = (gpu_memory_allocated / gpu_memory_total) * 100
        gpu_bar = _create_bar(gpu_percent)
        table.add_row(
            "GPU Memory",
            f"{gpu_memory_allocated:.1f} / {gpu_memory_total:.1f} GB",
            gpu_bar
        )
    
    console.print(Panel(
        table,
        title="[bold cyan]📈 Resource Usage[/bold cyan]",
        border_style="cyan"
    ))


def _create_bar(percent: float, width: int = 20) -> str:
    """Create a simple progress bar."""
    filled = int((percent / 100) * width)
    bar = "█" * filled + "░" * (width - filled)
    
    # Color based on percentage
    if percent < 50:
        color = "green"
    elif percent < 80:
        color = "yellow"
    else:
        color = "red"
    
    return f"[{color}]{bar}[/{color}]"
