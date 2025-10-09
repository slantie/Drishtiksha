# src/cli/utils/validators.py

"""Validation utilities for CLI inputs and environment."""

import os
import sys
from pathlib import Path
from typing import Optional, List
from rich.console import Console

console = Console()


# Supported file extensions
SUPPORTED_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v'}
SUPPORTED_AUDIO_EXTENSIONS = {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a', '.wma'}
SUPPORTED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff'}

ALL_SUPPORTED_EXTENSIONS = (
    SUPPORTED_VIDEO_EXTENSIONS | 
    SUPPORTED_AUDIO_EXTENSIONS | 
    SUPPORTED_IMAGE_EXTENSIONS
)


def validate_environment():
    """
    Validate that the environment is properly configured.
    Checks for required packages, models, and configuration.
    """
    # Check if running from correct directory
    server_root = Path(__file__).parent.parent.parent.parent
    if not (server_root / "src" / "ml").exists():
        console.print(
            "[bold red]❌ Error:[/bold red] CLI must be run from the Server directory.",
            style="red"
        )
        console.print("[dim]Current directory:[/dim]", Path.cwd())
        console.print("[dim]Expected directory:[/dim]", server_root)
        sys.exit(1)
    
    # Check for .env file
    env_file = server_root / ".env"
    if not env_file.exists():
        console.print(
            "[bold yellow]⚠️  Warning:[/bold yellow] .env file not found.",
            style="yellow"
        )
        console.print("[dim]Some features may not work correctly.[/dim]")


def validate_file_path(path: str) -> Path:
    """
    Validate that a file exists and is supported.
    
    Args:
        path: Path to the file
        
    Returns:
        Path object if valid
        
    Raises:
        click.BadParameter if invalid
    """
    import click
    
    file_path = Path(path).resolve()
    
    if not file_path.exists():
        raise click.BadParameter(f"File not found: {path}")
    
    if not file_path.is_file():
        raise click.BadParameter(f"Path is not a file: {path}")
    
    ext = file_path.suffix.lower()
    if ext not in ALL_SUPPORTED_EXTENSIONS:
        raise click.BadParameter(
            f"Unsupported file type: {ext}\n"
            f"Supported types: {', '.join(sorted(ALL_SUPPORTED_EXTENSIONS))}"
        )
    
    return file_path


def validate_directory_path(path: str) -> Path:
    """
    Validate that a directory exists.
    
    Args:
        path: Path to the directory
        
    Returns:
        Path object if valid
        
    Raises:
        click.BadParameter if invalid
    """
    import click
    
    dir_path = Path(path).resolve()
    
    if not dir_path.exists():
        raise click.BadParameter(f"Directory not found: {path}")
    
    if not dir_path.is_dir():
        raise click.BadParameter(f"Path is not a directory: {path}")
    
    return dir_path


def get_media_type(file_path: Path) -> str:
    """
    Determine the media type from file extension.
    
    Args:
        file_path: Path to the media file
        
    Returns:
        Media type: 'video', 'audio', or 'image'
    """
    ext = file_path.suffix.lower()
    
    if ext in SUPPORTED_VIDEO_EXTENSIONS:
        return 'video'
    elif ext in SUPPORTED_AUDIO_EXTENSIONS:
        return 'audio'
    elif ext in SUPPORTED_IMAGE_EXTENSIONS:
        return 'image'
    else:
        return 'unknown'


def find_media_files(directory: Path, recursive: bool = False) -> List[Path]:
    """
    Find all supported media files in a directory.
    
    Args:
        directory: Directory to search
        recursive: Whether to search recursively
        
    Returns:
        List of media file paths
    """
    media_files = []
    
    if recursive:
        pattern = "**/*"
    else:
        pattern = "*"
    
    for ext in ALL_SUPPORTED_EXTENSIONS:
        media_files.extend(directory.glob(f"{pattern}{ext}"))
        media_files.extend(directory.glob(f"{pattern}{ext.upper()}"))
    
    return sorted(set(media_files))


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def validate_model_name(model_name: str, available_models: List[str]) -> str:
    """
    Validate that a model name is available.
    
    Args:
        model_name: Name of the model
        available_models: List of available model names
        
    Returns:
        Model name if valid
        
    Raises:
        click.BadParameter if invalid
    """
    import click
    
    if model_name not in available_models:
        raise click.BadParameter(
            f"Model '{model_name}' not found.\n"
            f"Available models: {', '.join(available_models)}"
        )
    
    return model_name


def validate_output_path(path: str, force: bool = False) -> Path:
    """
    Validate output file path.
    
    Args:
        path: Output file path
        force: Whether to overwrite existing file
        
    Returns:
        Path object if valid
        
    Raises:
        click.BadParameter if invalid
    """
    import click
    
    output_path = Path(path).resolve()
    
    # Check if file already exists
    if output_path.exists() and not force:
        raise click.BadParameter(
            f"Output file already exists: {path}\n"
            f"Use --force to overwrite."
        )
    
    # Check if parent directory exists
    if not output_path.parent.exists():
        raise click.BadParameter(
            f"Output directory does not exist: {output_path.parent}"
        )
    
    return output_path
