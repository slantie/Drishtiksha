#!/usr/bin/env python3
"""
Memory diagnostic and optimization script for LSTM models.
This script helps identify and resolve memory issues when loading models.
"""

import os
import sys
import psutil
import gc
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def check_system_memory():
    """Check available system memory and virtual memory."""
    print("🔍 System Memory Analysis:")
    print("=" * 50)

    # Physical Memory
    memory = psutil.virtual_memory()
    print("Physical Memory:")
    print(f"  Total: {memory.total / (1024**3):.2f} GB")
    print(f"  Available: {memory.available / (1024**3):.2f} GB")
    print(f"  Used: {memory.used / (1024**3):.2f} GB ({memory.percent:.1f}%)")
    print(f"  Free: {memory.free / (1024**3):.2f} GB")

    # Virtual Memory (Swap)
    swap = psutil.swap_memory()
    print("\nVirtual Memory (Swap):")
    print(f"  Total: {swap.total / (1024**3):.2f} GB")
    print(f"  Used: {swap.used / (1024**3):.2f} GB ({swap.percent:.1f}%)")
    print(f"  Free: {swap.free / (1024**3):.2f} GB")

    # Recommendations
    print("\n💡 Recommendations:")
    if memory.available < 4 * (1024**3):  # Less than 4GB available
        print("  ⚠️  Low available memory detected")
        print("  - Close unnecessary applications")
        print("  - Consider increasing virtual memory")

    if swap.total < 8 * (1024**3):  # Less than 8GB swap
        print("  ⚠️  Small swap file detected")
        print("  - Consider increasing virtual memory size")
        print(
            "  - Windows: Control Panel > System > Advanced > Performance Settings > Virtual Memory"
        )

    return memory.available > 4 * (1024**3)  # Return True if we have enough memory


def optimize_environment():
    """Optimize Python environment for memory usage."""
    print("\n🔧 Optimizing Environment:")
    print("=" * 50)

    # Force garbage collection
    print("Running garbage collection...")
    collected = gc.collect()
    print(f"Collected {collected} objects")

    # Set environment variables for memory optimization
    os.environ["TORCH_CUDA_ARCH_LIST"] = ""  # Disable CUDA compilation
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Reduce tokenizer memory

    print("✅ Environment optimized")


def test_model_loading():
    """Test loading models with different memory configurations."""
    print("\n🧪 Testing Model Loading:")
    print("=" * 50)

    try:
        from src.config import settings
        from src.ml.registry import ModelManager

        # Test with v1 model first (simpler)
        print("Testing siglip-lstm-v1...")
        manager = ModelManager(settings.models)

        try:
            model = manager.get_model("siglip-lstm-v1")
            print("✅ siglip-lstm-v1 loaded successfully")

            # Test basic functionality
            info = model.get_info()
            print(f"Model info: {info}")

            return True

        except Exception as e:
            print(f"❌ siglip-lstm-v1 failed: {e}")
            return False

    except Exception as e:
        print(f"❌ Failed to import modules: {e}")
        return False


def suggest_solutions():
    """Provide specific solutions for memory issues."""
    print("\n💡 Memory Issue Solutions:")
    print("=" * 50)

    print("1. Immediate Solutions:")
    print("   - Close browser, IDE, and other memory-intensive apps")
    print("   - Restart your computer to free up memory")
    print("   - Stop the GPU training process temporarily")

    print("\n2. Windows Virtual Memory Settings:")
    print("   - Press Win+R, type 'sysdm.cpl', press Enter")
    print("   - Go to Advanced > Performance Settings > Advanced > Virtual Memory")
    print("   - Uncheck 'Automatically manage' and set custom size")
    print("   - Recommended: Initial = 4096 MB, Maximum = 8192 MB or higher")

    print("\n3. Model Configuration Changes:")
    print("   - Reduce num_frames from 120 to 60 or 30")
    print("   - Use smaller batch sizes")
    print("   - Enable low_cpu_mem_usage in model loading")

    print("\n4. Alternative Approaches:")
    print("   - Use the model on GPU when training is complete")
    print("   - Consider using a smaller/quantized model")
    print("   - Run inference in smaller batches")


def main():
    """Run comprehensive memory diagnosis."""
    print("🚀 LSTM Model Memory Diagnostic Tool")
    print("=" * 60)

    # Check system resources
    has_enough_memory = check_system_memory()

    # Optimize environment
    optimize_environment()

    # Test model loading
    if has_enough_memory:
        success = test_model_loading()
        if success:
            print("\n🎉 Model loading test passed!")
            print("You should be able to run the server now.")
        else:
            print("\n⚠️  Model loading failed despite sufficient memory.")
            suggest_solutions()
    else:
        print("\n⚠️  Insufficient memory detected.")
        suggest_solutions()

    print("\n" + "=" * 60)
    print("Diagnostic complete. Check the recommendations above.")


if __name__ == "__main__":
    main()
