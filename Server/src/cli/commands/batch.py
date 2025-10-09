# src/cli/commands/batch.py

"""Batch analysis command for processing multiple files."""

import click
import json
from pathlib import Path
from typing import List
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.table import Table
from rich.panel import Panel
from rich import box

from src.cli.utils.validators import validate_directory_path, find_media_files, get_media_type
from src.cli.utils.banner import print_section_header
from src.cli.config import get_cli_settings
from src.ml.registry import ModelManager

console = Console()


@click.command()
@click.argument('directory', type=click.Path(exists=True, file_okay=False), metavar='<directory>')
@click.option(
    '--recursive', '-r',
    is_flag=True,
    help='Search for files recursively in subdirectories.'
)
@click.option(
    '--model', '-m',
    help='Model name to use for all files.',
    metavar='<model>'
)
@click.option(
    '--output', '-o',
    type=click.Path(),
    help='Save batch results to JSON file.',
    metavar='<path>'
)
@click.option(
    '--parallel', '-p',
    type=int,
    default=1,
    help='Number of parallel analysis jobs (default: 1).',
    metavar='<count>'
)
@click.pass_context
def batch(ctx, directory: str, recursive: bool, model: str, output: str, parallel: int):
    """
    Analyze multiple media files in a directory.
    
    \b
    Examples:
      drishtiksha batch ./videos/
      drishtiksha batch ./media/ --recursive
      drishtiksha batch ./media/ --model EFFICIENTNET-B7-V1 --output results.json
      drishtiksha batch ./media/ --parallel 2
    """
    try:
        # Validate directory
        dir_path = validate_directory_path(directory)
        
        # Find all media files
        console.print(f"\n[cyan]Scanning directory:[/cyan] {dir_path}")
        if recursive:
            console.print("[dim]Searching recursively...[/dim]")
        
        media_files = find_media_files(dir_path, recursive=recursive)
        
        if not media_files:
            console.print("[yellow]No supported media files found in directory.[/yellow]")
            return
        
        console.print(f"[green]Found {len(media_files)} media file(s)[/green]\n")
        
        # Group files by type
        video_files = [f for f in media_files if get_media_type(f) == 'video']
        audio_files = [f for f in media_files if get_media_type(f) == 'audio']
        image_files = [f for f in media_files if get_media_type(f) == 'image']
        
        # Show summary
        summary_table = Table(show_header=True, box=box.SIMPLE)
        summary_table.add_column("Type", style="cyan")
        summary_table.add_column("Count", style="white", justify="right")
        
        if video_files:
            summary_table.add_row("🎥 Videos", str(len(video_files)))
        if audio_files:
            summary_table.add_row("🎵 Audio", str(len(audio_files)))
        if image_files:
            summary_table.add_row("🖼️  Images", str(len(image_files)))
        
        console.print(summary_table)
        console.print()
        
        # Load CLI settings (includes ALL models) and model manager
        cli_settings = get_cli_settings()
        model_manager = ModelManager(cli_settings)
        
        # Don't preload models - they'll be loaded lazily when needed
        console.print("[dim]Using lazy loading - models will load on-demand[/dim]\n")
        
        # Process files
        print_section_header("🔍 Processing Files", "🔬")
        results = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task(f"Analyzing {len(media_files)} files...", total=len(media_files))
            
            for idx, file_path in enumerate(media_files, 1):
                try:
                    progress.update(task, description=f"[{idx}/{len(media_files)}] {file_path.name}")
                    
                    # Determine model to use
                    media_type = get_media_type(file_path)
                    analysis_model = model if model else _auto_select_model(media_type, model_manager)
                    
                    # Get model instance
                    model_instance = model_manager.get_model(analysis_model)
                    
                    # Analyze
                    result = model_instance.analyze(
                        media_path=str(file_path),
                        generate_visualizations=False  # Don't generate visualizations in batch mode
                    )
                    
                    # Store result
                    results.append({
                        'file': str(file_path),
                        'filename': file_path.name,
                        'media_type': media_type,
                        'model': analysis_model,
                        'prediction': result.prediction,
                        'confidence': result.confidence,
                        'processing_time': result.processing_time,
                        'status': 'success'
                    })
                    
                except Exception as e:
                    console.print(f"\n[red]Error analyzing {file_path.name}: {str(e)}[/red]")
                    results.append({
                        'file': str(file_path),
                        'filename': file_path.name,
                        'media_type': get_media_type(file_path),
                        'model': analysis_model if 'analysis_model' in locals() else 'unknown',
                        'prediction': 'ERROR',
                        'confidence': 0.0,
                        'processing_time': 0.0,
                        'status': 'failed',
                        'error': str(e)
                    })
                
                progress.advance(task)
        
        console.print(f"\n[green]✓[/green] Batch analysis completed\n")
        
        # Display results summary
        _display_batch_results(results)
        
        # Save results if requested
        if output:
            _save_batch_results(results, output)
            console.print(f"\n[green]✓[/green] Results saved: [cyan]{output}[/cyan]")
        
    except click.Abort:
        raise
    except Exception as e:
        console.print(f"\n[bold red]❌ Batch analysis failed:[/bold red] {str(e)}")
        if ctx.obj.get('DEBUG'):
            console.print_exception()
        raise click.Abort()


def _auto_select_model(media_type: str, model_manager: ModelManager) -> str:
    """Auto-select the best model for the media type."""
    available_models = list(model_manager.model_configs.keys())
    
    # Priority list for each media type
    VIDEO_MODELS = ['EFFICIENTNET-B7-V1', 'SIGLIP-LSTM-V4', 'COLOR-CUES-LSTM-V1']
    AUDIO_MODELS = ['STFT-SPECTROGRAM-CNN-V1', 'MEL-SPECTROGRAM-CNN-V1']
    IMAGE_MODELS = ['MFF-MOE-V1', 'DISTIL-DIRE-V1']
    
    priority_list = {
        'video': VIDEO_MODELS,
        'audio': AUDIO_MODELS,
        'image': IMAGE_MODELS
    }.get(media_type, VIDEO_MODELS)
    
    # Find first available model
    for model in priority_list:
        if model in available_models:
            return model
    
    return available_models[0] if available_models else None


def _display_batch_results(results: List[dict]):
    """Display batch results in a table."""
    # Count statistics
    total = len(results)
    success = sum(1 for r in results if r['status'] == 'success')
    failed = total - success
    
    real_count = sum(1 for r in results if r['prediction'] == 'REAL')
    fake_count = sum(1 for r in results if r['prediction'] == 'FAKE')
    
    # Statistics panel
    stats_table = Table(show_header=False, box=box.SIMPLE)
    stats_table.add_column(style="cyan", width=25)
    stats_table.add_column(style="white", justify="right")
    
    stats_table.add_row("Total Files", str(total))
    stats_table.add_row("Successfully Analyzed", f"[green]{success}[/green]")
    stats_table.add_row("Failed", f"[red]{failed}[/red]" if failed > 0 else "0")
    stats_table.add_row("", "")
    stats_table.add_row("Predicted REAL", f"[green]{real_count}[/green]")
    stats_table.add_row("Predicted FAKE", f"[red]{fake_count}[/red]")
    
    console.print(Panel(
        stats_table,
        title="[bold cyan]📊 Batch Analysis Summary[/bold cyan]",
        border_style="cyan",
        box=box.ROUNDED
    ))
    
    # Detailed results table
    console.print()
    results_table = Table(show_header=True, box=box.ROUNDED, border_style="cyan")
    results_table.add_column("File", style="white", width=30)
    results_table.add_column("Prediction", justify="center", width=12)
    results_table.add_column("Confidence", justify="right", width=12)
    results_table.add_column("Time (s)", justify="right", width=10)
    
    for result in results[:20]:  # Show first 20 results
        pred = result['prediction']
        conf = result['confidence'] * 100 if result['status'] == 'success' else 0
        time_str = f"{result['processing_time']:.2f}" if result['status'] == 'success' else "-"
        
        # Color based on prediction
        if pred == "REAL":
            pred_display = "[green]✓ REAL[/green]"
            conf_display = f"[green]{conf:.1f}%[/green]"
        elif pred == "FAKE":
            pred_display = "[red]⚠ FAKE[/red]"
            conf_display = f"[red]{conf:.1f}%[/red]"
        else:
            pred_display = "[dim]ERROR[/dim]"
            conf_display = "[dim]-[/dim]"
        
        results_table.add_row(
            result['filename'][:30],
            pred_display,
            conf_display,
            time_str
        )
    
    if len(results) > 20:
        results_table.add_row(
            f"[dim]... and {len(results) - 20} more[/dim]",
            "", "", ""
        )
    
    console.print(results_table)


def _save_batch_results(results: List[dict], output_path: str):
    """Save batch results to JSON file."""
    output_data = {
        "total_files": len(results),
        "successful": sum(1 for r in results if r['status'] == 'success'),
        "failed": sum(1 for r in results if r['status'] == 'failed'),
        "real_count": sum(1 for r in results if r['prediction'] == 'REAL'),
        "fake_count": sum(1 for r in results if r['prediction'] == 'FAKE'),
        "results": results
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
