#!/usr/bin/env python3
"""
Quick memory optimization and test script.
Run this before starting the server to optimize memory usage.
"""

import os
import gc
import sys

def optimize_memory():
    """Apply immediate memory optimizations."""
    print("🔧 Applying Memory Optimizations...")

    # Force garbage collection
    gc.collect()

    # Set memory-friendly environment variables
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TRANSFORMERS_CACHE"] = "./cache"  # Use local cache

    # Reduce Python memory usage
    sys.setrecursionlimit(1000)  # Reduce recursion limit

    print("✅ Memory optimizations applied")
    print("\n💡 Additional steps:")
    print("1. Close other applications (browser, IDE, etc.)")
    print("2. If the issue persists, restart your computer")
    print("3. Consider increasing virtual memory in Windows settings")


if __name__ == "__main__":
    optimize_memory()

    print("\n🚀 Now try starting the server:")
    print("uv run uvicorn src.app.main:app --host 0.0.0.0 --port 8000 --reload")
