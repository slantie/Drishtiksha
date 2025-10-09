#!/usr/bin/env python3
# Test script to debug CLI loading

print("1. Starting imports...")

try:
    import click
    print("✓ Click imported")
except Exception as e:
    print(f"✗ Click failed: {e}")

try:
    from rich.console import Console
    print("✓ Rich imported")
    console = Console()
except Exception as e:
    print(f"✗ Rich failed: {e}")

try:
    print("2. Importing src.config...")
    from src.config import settings
    print(f"✓ Settings imported: {settings.project_name}")
except Exception as e:
    print(f"✗ Settings failed: {e}")
    import traceback
    traceback.print_exc()

try:
    print("3. Importing ModelManager...")
    from src.ml.registry import ModelManager
    print("✓ ModelManager imported")
except Exception as e:
    print(f"✗ ModelManager failed: {e}")
    import traceback
    traceback.print_exc()

print("\n4. All imports successful! Testing CLI...")

try:
    from src.cli.main import main
    print("✓ CLI main imported")
except Exception as e:
    print(f"✗ CLI main failed: {e}")
    import traceback
    traceback.print_exc()
