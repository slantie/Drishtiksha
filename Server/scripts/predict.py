# scripts/predict.py

import argparse
import sys
import os
import time

# Add the project root to the Python path to allow for absolute imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import settings
from src.ml.registry import ModelManager

def main():
    parser = argparse.ArgumentParser(description="Run deepfake prediction on a local video file.")
    parser.add_argument("video_path", type=str, help="Path to the video file to analyze.")
    parser.add_argument(
        "--model_name", 
        type=str, 
        default=settings.default_model_name,
        help=f"Name of the model to use (default: {settings.default_model_name})."
    )
    
    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found at '{args.video_path}'")
        return

    print("Initializing Model Manager...")
    # The manager takes the 'models' section of the config
    manager = ModelManager(settings.models)
    
    try:
        print(f"Loading model: '{args.model_name}'...")
        start_load_time = time.time()
        model = manager.get_model(args.model_name)
        load_time = time.time() - start_load_time
        print(f"Model loaded in {load_time:.2f}s.")
        
        print(f"\nAnalyzing video: {args.video_path}")
        result = model.predict(args.video_path)
        
        print("\n--- Prediction Result ---")
        print(f"  Model Used:      {args.model_name}")
        print(f"  Prediction:      {result['prediction']}")
        print(f"  Confidence:      {result['confidence'] * 100:.2f}%")
        print(f"  Processing Time: {result['processing_time']:.2f}s")
        print("--------------------------")

    except (ValueError, RuntimeError) as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()