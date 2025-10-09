# src/cli/commands/analyze.py

"""Single file analysis command."""

import contextlib
import io
import json
import sys
import time
from pathlib import Path
from typing import Optional

import click
import questionary
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich import box

from src.cli.utils.validators import validate_file_path, get_media_type
from src.cli.utils.banner import print_section_header
from src.cli.config import get_cli_settings
from src.ml.registry import ModelManager

console = Console()


@contextlib.contextmanager
def suppress_tqdm():
    """Context manager to suppress TQDM output."""
    # Save original stderr
    old_stderr = sys.stderr
    
    try:
        # Redirect stderr to suppress TQDM
        sys.stderr = io.StringIO()
        yield
    finally:
        # Restore stderr
        sys.stderr = old_stderr


@click.command()
@click.argument('file_path', type=click.Path(exists=True), metavar='<file>')
@click.option(
    '--model', '-m',
    help='Model name to use for analysis (e.g., EFFICIENTNET-B7-V1).',
    metavar='<model>'
)
@click.option(
    '--custom', '-c',
    is_flag=True,
    help='Interactively select multiple models to use for analysis.'
)
@click.option(
    '--visualize', '-v',
    is_flag=True,
    help='Generate visualization video/image (saves to current directory).'
)
@click.option(
    '--output', '-o',
    type=click.Path(),
    help='Save results to JSON file.',
    metavar='<path>'
)
@click.option(
    '--verbose',
    is_flag=True,
    help='Show detailed analysis information.'
)
@click.pass_context
def analyze(ctx, file_path: str, model: Optional[str], custom: bool, visualize: bool, output: Optional[str], verbose: bool):
    """
    Analyze a single media file for deepfake content.
    
    \b
    By default, uses ALL models for the detected media type (ensemble).
    Use --model to specify a single model, or --custom for interactive selection.
    
    \b
    Examples:
      drishtiksha analyze video.mp4                              # All video models
      drishtiksha analyze video.mp4 --model EFFICIENTNET-B7-V1   # Single model
      drishtiksha analyze video.mp4 --custom                     # Interactive selection
      drishtiksha analyze video.mp4 --visualize --output results.json
      drishtiksha analyze audio.mp3 --model STFT-SPECTROGRAM-CNN-V1
    """
    try:
        # Validate file
        file_path = validate_file_path(file_path)
        media_type = get_media_type(file_path)
        
        # Show file info
        print_section_header("📄 File Information")
        console.print(f"[cyan]Path:[/cyan] {file_path}")
        console.print(f"[cyan]Type:[/cyan] {media_type.upper()}")
        console.print(f"[cyan]Size:[/cyan] {_format_size(file_path.stat().st_size)}")
        
        # Load CLI settings (includes ALL models) and model manager
        cli_settings = get_cli_settings()
        model_manager = ModelManager(cli_settings)
        
        # Validate flags
        if model and custom:
            console.print("[bold red]❌ Error:[/bold red] Cannot use both --model and --custom flags together.")
            raise click.Abort()
        
        # Determine which models to use
        if custom:
            # Interactive custom model selection
            models_to_use = _select_custom_models(media_type, model_manager)
            if not models_to_use:
                console.print("[yellow]⚠️  No models selected. Aborting.[/yellow]")
                raise click.Abort()
            console.print(f"[dim]Using {len(models_to_use)} selected model(s): {', '.join(models_to_use)}[/dim]\n")
        elif model is None:
            # Use ALL models for this media type (ensemble approach)
            models_to_use = _get_models_for_type(media_type, model_manager)
            console.print(f"[dim]Using all {len(models_to_use)} {media_type} models: {', '.join(models_to_use)}[/dim]\n")
        else:
            # Validate model exists
            if model not in model_manager.model_configs:
                console.print(f"[bold red]❌ Error:[/bold red] Model '{model}' not found.")
                _list_available_models(model_manager, media_type)
                raise click.Abort()
            models_to_use = [model]
            console.print(f"[dim]Using specified model: {model}[/dim]\n")
        
        # Run analysis with all selected models
        print_section_header("� Analyzing File", "🔬")
        
        all_results = []
        failed_models = []
        total_time = 0
        
        # Use Rich progress bar for clean, single-line progress display
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            overall_task = progress.add_task(
                "[cyan]Analyzing with multiple models...", 
                total=len(models_to_use)
            )
            
            for idx, model_name in enumerate(models_to_use, 1):
                # Update progress bar
                progress.update(
                    overall_task,
                    description=f"[cyan]Model {idx}/{len(models_to_use)}: {model_name}..."
                )
                
                try:
                    # Load model lazily
                    model_instance = model_manager.get_model(model_name)
                    
                    # Perform analysis with TQDM suppression for clean output
                    start_time = time.time()
                    with suppress_tqdm():
                        result = model_instance.analyze(
                            media_path=str(file_path),
                            generate_visualizations=visualize
                        )
                    elapsed = time.time() - start_time
                    total_time += elapsed
                    
                    all_results.append({
                        'model': model_name,
                        'result': result,
                        'time': elapsed
                    })
                    
                except Exception as e:
                    # Log failure but continue with other models
                    failed_models.append({
                        'model': model_name,
                        'error': str(e)
                    })
                
                progress.advance(overall_task)
        
        # Show completion message
        if all_results:
            console.print(f"\n[bold green]✓ {len(all_results)} model(s) completed in {total_time:.2f}s total[/bold green]")
        
        # Show failed models warning
        if failed_models:
            console.print(f"[bold yellow]⚠️  {len(failed_models)} model(s) failed to load or analyze[/bold yellow]")
            for failure in failed_models:
                error_msg = failure['error'][:100] + '...' if len(failure['error']) > 100 else failure['error']
                console.print(f"  [yellow]✗[/yellow] [dim]{failure['model']}: {error_msg}[/dim]")
        
        console.print()
        
        # Check if we have any successful results
        if not all_results:
            console.print("[bold red]❌ All selected models failed. No results to display.[/bold red]")
            if failed_models:
                console.print("\n[yellow]Suggestions:[/yellow]")
                console.print("  • Check if model weights are properly downloaded")
                console.print("  • Verify model configurations in config.yaml")
                console.print("  • Try selecting different models with --custom")
                console.print("  • Run 'drishtiksha models' to see available models")
            raise click.Abort()
        
        # Display results from all models
        _display_multi_model_results(all_results, file_path, verbose)
        
        # Show visualizations if generated
        if visualize:
            console.print()
            vis_count = 0
            for item in all_results:
                if hasattr(item['result'], 'visualization_path') and item['result'].visualization_path:
                    console.print(f"[green]✓[/green] [{item['model']}] Visualization: [cyan]{item['result'].visualization_path}[/cyan]")
                    vis_count += 1
            if vis_count > 0:
                console.print(f"\n[green]Generated {vis_count} visualization(s)[/green]")
        
        # Save to JSON if requested (save all results)
        if output:
            _save_all_results(all_results, output)
            console.print(f"[green]✓[/green] Results saved: [cyan]{output}[/cyan]")
        
    except click.Abort:
        raise
    except Exception as e:
        console.print(f"\n[bold red]❌ Analysis failed:[/bold red] {str(e)}")
        if ctx.obj.get('DEBUG'):
            console.print_exception()
        raise click.Abort()


def _get_models_for_type(media_type: str, model_manager: ModelManager) -> list:
    """Get all models that support the given media type."""
    suitable_models = []
    
    for model_name, config in model_manager.model_configs.items():
        if media_type == 'video' and config.isVideo:
            suitable_models.append(model_name)
        elif media_type == 'audio' and config.isAudio:
            suitable_models.append(model_name)
        elif media_type == 'image' and config.isImage:
            suitable_models.append(model_name)
    
    if not suitable_models:
        raise click.ClickException(f"No models available for {media_type} analysis.")
    
    return suitable_models


def _select_custom_models(media_type: str, model_manager: ModelManager) -> list:
    """Interactively select models using questionary checkbox."""
    available_models = _get_models_for_type(media_type, model_manager)
    
    if not available_models:
        console.print(f"[bold red]❌ No models available for {media_type} analysis.[/bold red]")
        return []
    
    # Create choices with model info
    choices = []
    for model_name in available_models:
        config = model_manager.model_configs[model_name]
        
        # Build description
        desc_parts = []
        if config.isVideo:
            desc_parts.append("video")
        if config.isAudio:
            desc_parts.append("audio")
        if config.isImage:
            desc_parts.append("image")
        
        description = f"{model_name} ({', '.join(desc_parts)})"
        choices.append(questionary.Choice(title=description, value=model_name))
    
    # Show interactive checkbox prompt
    console.print(f"\n[bold cyan]📋 Select Models for {media_type.upper()} Analysis[/bold cyan]")
    console.print("[dim]Use arrow keys to navigate, Space to select/deselect, Enter to confirm[/dim]\n")
    
    selected = questionary.checkbox(
        "Select one or more models:",
        choices=choices,
        style=questionary.Style([
            ('checkbox', 'fg:cyan'),
            ('selected', 'fg:green bold'),
            ('pointer', 'fg:cyan bold'),
            ('highlighted', 'fg:cyan bold'),
        ])
    ).ask()
    
    if selected is None:  # User cancelled (Ctrl+C)
        return []
    
    return selected


def _auto_select_model(media_type: str, model_manager: ModelManager) -> str:
    """Auto-select the best model for the media type (kept for backward compatibility)."""
    models = _get_models_for_type(media_type, model_manager)
    
    # Priority list for each media type
    VIDEO_MODELS = ['EFFICIENTNET-B7-V1', 'SIGLIP-LSTM-V4', 'COLOR-CUES-LSTM-V1', 'EYEBLINK-CNN-LSTM-V1']
    AUDIO_MODELS = ['STFT-SPECTROGRAM-CNN-V1', 'MEL-SPECTROGRAM-CNN-V1', 'SCATTERING-WAVE-V1']
    IMAGE_MODELS = ['MFF-MOE-V1', 'DISTIL-DIRE-V1', 'EFFICIENTNET-B7-V1']
    
    priority_list = {
        'video': VIDEO_MODELS,
        'audio': AUDIO_MODELS,
        'image': IMAGE_MODELS
    }.get(media_type, VIDEO_MODELS)
    
    # Find first available model from priority list
    for model in priority_list:
        if model in models:
            return model
    
    # Fallback to first available model
    return models[0]


def _list_available_models(model_manager: ModelManager, media_type: Optional[str] = None):
    """List available models."""
    console.print("\n[bold]Available models:[/bold]")
    for model_name, model_config in model_manager.model_configs.items():
        icon = "🎥" if model_config.isVideo else "🎵" if model_config.isAudio else "🖼️"
        console.print(f"  {icon} [cyan]{model_name}[/cyan] - {model_config.description}")


def _display_multi_model_results(all_results: list, file_path: Path, verbose: bool):
    """Display results from multiple models."""
    print_section_header("📊 Analysis Results Summary")
    
    # Create results table
    table = Table(show_header=True, box=box.ROUNDED, border_style="cyan")
    table.add_column("Model", style="cyan", width=30)
    table.add_column("Prediction", justify="center", width=12)
    table.add_column("Confidence", justify="right", width=12)
    table.add_column("Time", justify="right", width=10)
    
    real_count = 0
    fake_count = 0
    total_confidence = 0
    
    for item in all_results:
        result = item['result']
        prediction = result.prediction
        confidence = result.confidence * 100
        elapsed = item['time']
        
        if prediction == "REAL":
            pred_color = "green"
            emoji = "✅"
            real_count += 1
        else:
            pred_color = "red"
            emoji = "⚠️"
            fake_count += 1
        
        total_confidence += confidence
        
        table.add_row(
            item['model'],
            f"[{pred_color}]{emoji} {prediction}[/{pred_color}]",
            f"{confidence:.2f}%",
            f"{elapsed:.2f}s"
        )
    
    console.print(table)
    
    # Consensus/Ensemble result
    console.print()
    avg_confidence = total_confidence / len(all_results)
    
    if real_count > fake_count:
        final_prediction = "REAL"
        final_color = "green"
        final_emoji = "✅"
    elif fake_count > real_count:
        final_prediction = "FAKE"
        final_color = "red"
        final_emoji = "⚠️"
    else:
        final_prediction = "UNCERTAIN"
        final_color = "yellow"
        final_emoji = "❓"
    
    consensus_table = Table(show_header=False, box=box.ROUNDED, border_style=final_color)
    consensus_table.add_column(style="bold cyan", width=20)
    consensus_table.add_column(style="white")
    
    consensus_table.add_row("File", str(file_path.name))
    consensus_table.add_row("Models Used", str(len(all_results)))
    consensus_table.add_row("Votes REAL", f"[green]{real_count}[/green]")
    consensus_table.add_row("Votes FAKE", f"[red]{fake_count}[/red]")
    consensus_table.add_row("Ensemble Prediction", f"[bold {final_color}]{final_emoji} {final_prediction}[/bold {final_color}]")
    consensus_table.add_row("Average Confidence", f"{avg_confidence:.2f}%")
    
    console.print(Panel(
        consensus_table,
        title=f"[bold {final_color}]🎯 Ensemble Result[/bold {final_color}]",
        border_style=final_color,
        box=box.ROUNDED
    ))


def _display_results(result, file_path: Path, model: str, verbose: bool):
    """Display analysis results in a beautiful format (single model)."""
    # Main result panel
    prediction = result.prediction
    confidence = result.confidence * 100
    
    # Color based on prediction
    if prediction == "REAL":
        pred_color = "green"
        emoji = "✅"
    else:
        pred_color = "red"
        emoji = "⚠️"
    
    # Create result table
    table = Table(show_header=False, box=box.ROUNDED, border_style="cyan")
    table.add_column(style="bold cyan", width=20)
    table.add_column(style="white")
    
    table.add_row("File", str(file_path.name))
    table.add_row("Model", model)
    table.add_row("Prediction", f"[bold {pred_color}]{emoji} {prediction}[/bold {pred_color}]")
    table.add_row("Confidence", f"{confidence:.2f}%")
    table.add_row("Processing Time", f"{result.processing_time:.2f}s")
    
    console.print(Panel(
        table,
        title="[bold cyan]📊 Analysis Results[/bold cyan]",
        border_style="cyan",
        box=box.ROUNDED
    ))
    
    # Show detailed metrics if verbose
    if verbose and hasattr(result, 'metrics'):
        console.print()
        print_section_header("📈 Detailed Metrics")
        _display_metrics(result.metrics)


def _display_metrics(metrics):
    """Display detailed metrics."""
    if isinstance(metrics, dict):
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                console.print(f"  [cyan]{key}:[/cyan] {value}")
            elif isinstance(value, dict):
                console.print(f"  [cyan]{key}:[/cyan]")
                for sub_key, sub_value in value.items():
                    console.print(f"    [dim]{sub_key}:[/dim] {sub_value}")


def _save_results(result, output_path: str):
    """Save results to JSON file."""
    output_data = {
        "prediction": result.prediction,
        "confidence": result.confidence,
        "processing_time": result.processing_time,
        "media_type": result.media_type if hasattr(result, 'media_type') else "unknown",
        "note": result.note if hasattr(result, 'note') else None,
    }
    
    # Add metrics if available
    if hasattr(result, 'metrics'):
        if hasattr(result.metrics, 'model_dump'):
            output_data["metrics"] = result.metrics.model_dump()
        else:
            output_data["metrics"] = result.metrics
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)


def _save_all_results(all_results: list, output_path: str):
    """Save multi-model results to JSON file."""
    results_data = []
    
    for item in all_results:
        result = item['result']
        result_data = {
            "model": item['model'],
            "prediction": result.prediction,
            "confidence": result.confidence,
            "processing_time": item['time'],
            "media_type": result.media_type if hasattr(result, 'media_type') else "unknown",
            "note": result.note if hasattr(result, 'note') else None,
        }
        
        # Add metrics if available
        if hasattr(result, 'metrics'):
            if hasattr(result.metrics, 'model_dump'):
                result_data["metrics"] = result.metrics.model_dump()
            else:
                result_data["metrics"] = result.metrics
        
        # Add visualization path if available
        if hasattr(result, 'visualization_path') and result.visualization_path:
            result_data["visualization_path"] = result.visualization_path
        
        results_data.append(result_data)
    
    # Calculate ensemble prediction
    real_count = sum(1 for r in all_results if r['result'].prediction == "REAL")
    fake_count = sum(1 for r in all_results if r['result'].prediction == "FAKE")
    avg_confidence = sum(r['result'].confidence for r in all_results) / len(all_results)
    
    if real_count > fake_count:
        ensemble_prediction = "REAL"
    elif fake_count > real_count:
        ensemble_prediction = "FAKE"
    else:
        ensemble_prediction = "UNCERTAIN"
    
    output_data = {
        "individual_results": results_data,
        "ensemble": {
            "prediction": ensemble_prediction,
            "votes_real": real_count,
            "votes_fake": fake_count,
            "average_confidence": avg_confidence,
            "total_models": len(all_results)
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)


def _format_size(size_bytes: int) -> str:
    """Format file size."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"
