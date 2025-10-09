#!/usr/bin/env python3
# Simple test

print("Testing basic imports...")

import sys
print(f"Python: {sys.version}")

print("Importing click...")
import click
print("✓ Click OK")

print("Importing rich...")
from rich.console import Console
console = Console()
console.print("[green]✓ Rich OK[/green]")

print("\nAll basic imports work!")
