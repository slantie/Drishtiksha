# src/cli/commands/interactive.py

"""Interactive mode for guided analysis."""

import click
import questionary
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from src.cli.utils.banner import print_welcome, print_section_header
from src.cli.utils.validators import (
    get_media_type,
    find_media_files,
    ALL_SUPPORTED_EXTENSIONS
)
from src.cli.config import get_cli_settings
from src.ml.registry import ModelManager
from src.cli.commands.analyze import analyze as analyze_cmd

console = Console()


@click.command()
def interactive():
    """
    Interactive mode with guided prompts.
    
    Perfect for first-time users! This mode will guide you through
    the analysis process step-by-step with helpful prompts.
    Runs in a continuous loop until you choose to exit.
    """
    print_welcome()
    
    # Track session statistics
    session_stats = {
        'total_analyzed': 0,
        'real_count': 0,
        'fake_count': 0,
        'files': []
    }
    
    # Initialize model manager once with CLI settings (includes ALL models)
    cli_settings = get_cli_settings()
    model_manager = ModelManager(cli_settings)
    
    try:
        while True:
            try:
                console.print()
                console.print("─" * 70, style="dim")
                
                # Step 1: Select file
                console.print("\n[bold cyan]📁 Step 1: Select Media File[/bold cyan]")
                file_path = _select_file()
                if not file_path:
                    # Ask if user wants to exit or try again
                    if not questionary.confirm("❓ Try selecting another file?", default=True).ask():
                        break
                    continue
                
                media_type = get_media_type(file_path)
                file_size = file_path.stat().st_size / (1024 * 1024)  # MB
                console.print(f"\n  [green]✓[/green] File: [cyan]{file_path.name}[/cyan]")
                console.print(f"  [dim]Type: {media_type.upper()}  •  Size: {file_size:.2f} MB[/dim]\n")
                
                # Step 2: Select model
                console.print("[bold cyan]🤖 Step 2: Select Analysis Model[/bold cyan]")
                model = _select_model(media_type, model_manager)
                console.print(f"\n  [green]✓[/green] Model: [cyan]{model}[/cyan]\n")
                
                # Step 3: Options
                console.print("[bold cyan]⚙️  Step 3: Analysis Options[/bold cyan]")
                visualize = questionary.confirm(
                    "  🎨 Generate visualization?",
                    default=False
                ).ask()
                
                save_results = questionary.confirm(
                    "  💾 Save results to JSON?",
                    default=False
                ).ask()
                
                output_path = None
                if save_results:
                    output_path = questionary.text(
                        "  📄 Output filename:",
                        default=f"analysis-{file_path.stem}.json"
                    ).ask()
                
                console.print()
                
                # Step 4: Review and confirm
                console.print("[bold cyan]📋 Step 4: Review Configuration[/bold cyan]")
                review_table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
                review_table.add_column(style="dim", width=18)
                review_table.add_column(style="white")
                
                review_table.add_row("📁 File", file_path.name)
                review_table.add_row("🤖 Model", model)
                review_table.add_row("📊 Type", media_type.upper())
                review_table.add_row("🎨 Visualization", "[green]Yes[/green]" if visualize else "[dim]No[/dim]")
                review_table.add_row("💾 Save Results", "[green]Yes[/green]" if save_results else "[dim]No[/dim]")
                if output_path:
                    review_table.add_row("📄 Output", output_path)
                
                console.print(Panel(
                    review_table,
                    title="[bold cyan]Configuration Summary[/bold cyan]",
                    border_style="cyan",
                    box=box.ROUNDED
                ))
                console.print()
                
                # Confirm
                confirm = questionary.confirm(
                    "🚀 Start analysis?",
                    default=True
                ).ask()
                
                if not confirm:
                    console.print("  [yellow]⚠️  Analysis cancelled.[/yellow]")
                    if not questionary.confirm("❓ Analyze another file?", default=True).ask():
                        break
                    continue
                
                # Run analysis
                console.print("\n[bold green]🔬 Running analysis...[/bold green]\n")
                
                from click.testing import CliRunner
                runner = CliRunner()
                
                args = [str(file_path), '--model', model]
                if visualize:
                    args.append('--visualize')
                if output_path:
                    args.extend(['--output', output_path])
                
                result = runner.invoke(analyze_cmd, args, obj={'DEBUG': False}, catch_exceptions=False)
                
                # Parse result for session stats
                if result.exit_code == 0:
                    console.print("\n[bold green]✅ Analysis completed successfully![/bold green]")
                    
                    # Extract prediction from output (simple parsing)
                    result_text = result.output
                    is_real = "REAL" in result_text and "Prediction" in result_text
                    is_fake = "FAKE" in result_text and "Prediction" in result_text
                    
                    session_stats['total_analyzed'] += 1
                    if is_real:
                        session_stats['real_count'] += 1
                    elif is_fake:
                        session_stats['fake_count'] += 1
                    
                    session_stats['files'].append({
                        'name': file_path.name,
                        'model': model,
                        'prediction': 'REAL' if is_real else 'FAKE' if is_fake else 'UNKNOWN'
                    })
                else:
                    console.print(f"\n[bold red]❌ Analysis failed![/bold red]")
                
                # Show session summary
                _display_session_summary(session_stats)
                
                # Ask to continue
                console.print()
                if not questionary.confirm(
                    "🔄 Analyze another file?",
                    default=True
                ).ask():
                    break
                    
            except KeyboardInterrupt:
                console.print("\n[yellow]⚠️  Operation cancelled.[/yellow]")
                if not questionary.confirm("❓ Continue in interactive mode?", default=True).ask():
                    break
                continue
    
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Interactive mode terminated.[/yellow]")
    except Exception as e:
        console.print(f"\n[bold red]❌ Error:[/bold red] {str(e)}")
        raise
    finally:
        # Final session summary
        if session_stats['total_analyzed'] > 0:
            console.print("\n" + "═" * 70, style="cyan")
            console.print("[bold cyan]📊 Final Session Summary[/bold cyan]")
            console.print("═" * 70, style="cyan")
            _display_session_summary(session_stats)
            console.print("\n[bold green]Thank you for using Drishtiksha! 👋[/bold green]\n")


def _select_file() -> Path:
    """Interactive file selection."""
    # Ask for file path
    while True:
        file_path_str = questionary.text(
            "Enter media file path:",
            instruction="(Use absolute or relative path)"
        ).ask()
        
        if not file_path_str:
            return None
        
        file_path = Path(file_path_str).resolve()
        
        if not file_path.exists():
            console.print(f"[red]File not found: {file_path}[/red]")
            continue
        
        if not file_path.is_file():
            console.print(f"[red]Path is not a file: {file_path}[/red]")
            continue
        
        ext = file_path.suffix.lower()
        if ext not in ALL_SUPPORTED_EXTENSIONS:
            console.print(f"[red]Unsupported file type: {ext}[/red]")
            console.print(f"[dim]Supported: {', '.join(sorted(ALL_SUPPORTED_EXTENSIONS))}[/dim]")
            continue
        
        return file_path


def _select_model(media_type: str, model_manager: ModelManager) -> str:
    """Interactive model selection."""
    available_models = list(model_manager.model_configs.keys())
    
    # Filter models by media type
    suitable_models = []
    for model_name in available_models:
        config = model_manager.model_configs[model_name]
        
        if media_type == 'video' and config.isVideo:
            suitable_models.append(model_name)
        elif media_type == 'audio' and config.isAudio:
            suitable_models.append(model_name)
        elif media_type == 'image' and config.isImage:
            suitable_models.append(model_name)
    
    # If no suitable models, use all
    if not suitable_models:
        suitable_models = available_models
    
    # Create choices with descriptions
    choices = []
    for model_name in suitable_models:
        config = model_manager.model_configs[model_name]
        icon = "🎥" if config.isVideo else "🎵" if config.isAudio else "🖼️"
        choice = questionary.Choice(
            title=f"{icon} {model_name} - {config.description}",
            value=model_name
        )
        choices.append(choice)
    
    # Ask user to select
    selected = questionary.select(
        f"Select model for {media_type} analysis:",
        choices=choices
    ).ask()
    
    return selected


def _display_session_summary(session_stats: dict):
    """Display current session statistics."""
    if session_stats['total_analyzed'] == 0:
        return
    
    console.print()
    summary_table = Table(show_header=False, box=box.ROUNDED, border_style="cyan", padding=(0, 2))
    summary_table.add_column(style="cyan", width=25)
    summary_table.add_column(style="white", justify="right")
    
    summary_table.add_row("📊 Session Statistics", "")
    summary_table.add_row("─" * 25, "─" * 10)
    summary_table.add_row("Total Analyzed", f"[bold]{session_stats['total_analyzed']}[/bold]")
    summary_table.add_row("✅ Predicted REAL", f"[green]{session_stats['real_count']}[/green]")
    summary_table.add_row("⚠️  Predicted FAKE", f"[red]{session_stats['fake_count']}[/red]")
    
    console.print(Panel(
        summary_table,
        title="[bold cyan]Current Session[/bold cyan]",
        border_style="cyan"
    ))
    
    # Show analyzed files
    if session_stats['files']:
        console.print("\n[bold cyan]📝 Analyzed Files:[/bold cyan]")
        files_table = Table(box=box.SIMPLE, show_header=True, border_style="dim")
        files_table.add_column("File", style="cyan", no_wrap=True)
        files_table.add_column("Model", style="dim", no_wrap=True)
        files_table.add_column("Result", justify="center")
        
        for file_info in session_stats['files']:
            prediction = file_info['prediction']
            if prediction == 'REAL':
                result_str = "[green]✅ REAL[/green]"
            elif prediction == 'FAKE':
                result_str = "[red]⚠️  FAKE[/red]"
            else:
                result_str = "[dim]❓ UNKNOWN[/dim]"
            
            files_table.add_row(
                file_info['name'][:40],  # Truncate long names
                file_info['model'][:25],
                result_str
            )
        
        console.print(files_table)


