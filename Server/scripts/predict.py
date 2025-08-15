# scripts/predict.py

import argparse
import sys
import os
import time
import logging

# Add the project root to the Python path to allow for absolute imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import settings
from src.ml.registry import ModelManager

# Configure basic logging for the script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """A command-line tool to run deepfake prediction on a local video file."""
    parser = argparse.ArgumentParser(description="Run deepfake prediction on a local video file.")
    parser.add_argument("video_path", type=str, help="Path to the video file to analyze.")
    parser.add_argument(
        "--model", 
        type=str, 
        default=settings.default_model_name,
        help=f"Name of the model to use (default: {settings.default_model_name})."
    )
    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        logging.error(f"Video file not found at '{args.video_path}'")
        return

    try:
        logging.info("Initializing Model Manager...")
        # The refactored ModelManager now takes the full settings object
        manager = ModelManager(settings)
        
        logging.info(f"Loading model: '{args.model}'...")
        start_load_time = time.time()
        model = manager.get_model(args.model)
        load_time = time.time() - start_load_time
        logging.info(f"Model loaded in {load_time:.2f}s.")
        
        logging.info(f"Analyzing video: {os.path.basename(args.video_path)}")
        # Use the appropriate prediction method based on the model's capabilities
        if hasattr(model, "predict_detailed"):
            result = model.predict_detailed(args.video_path)
            prediction = result['prediction']
            confidence = result['confidence']
            processing_time = result['processing_time']
            avg_score = result.get('metrics', {}).get('final_average_score', 'N/A')
            logging.info(f"Detailed analysis complete. Average score: {avg_score:.4f}")
        else:
            result = model.predict(args.video_path)
            prediction = result['prediction']
            confidence = result['confidence']
            processing_time = result['processing_time']
        
        print("\n--- Prediction Result ---")
        print(f"  Model Used:      {args.model}")
        print(f"  Prediction:      {prediction}")
        print(f"  Confidence:      {confidence * 100:.2f}%")
        print(f"  Processing Time: {processing_time:.2f}s")
        print("--------------------------\n")

    except (ValueError, RuntimeError) as e:
        logging.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    main()