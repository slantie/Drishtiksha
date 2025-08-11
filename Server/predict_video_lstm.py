# /home/dell-pc-03/Deepfake/deepfake-detection/Raj/LSTM_video/predict_video_lstm.py

import torch
import cv2
import numpy as np
import os
from PIL import Image
from transformers import AutoProcessor
import torch.nn.functional as F

from src.utils import load_config
from src.model_lstm import create_lstm_model

def extract_frames(video_path, num_frames):
    """Helper function to extract frames, included here for self-sufficiency."""
    frames = []
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 1: return []
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        cap.release()
    except Exception as e:
        print(f"Error processing video {video_path}: {e}")
        return []
    return frames

def predict_video(video_path, model, processor, config, device):
    model.eval()
    print(f"Analyzing video: {video_path}")
    frames = extract_frames(video_path, config['num_frames_per_video'])
    
    if not frames:
        print("Could not extract frames. Exiting.")
        return

    inputs = processor(images=frames, return_tensors="pt")
    pixel_values = inputs['pixel_values'].to(device)

    with torch.no_grad():
        logits = model(pixel_values, num_frames_per_video=config['num_frames_per_video'])
        probabilities = F.softmax(logits, dim=1)
        # Find the max probability and its index across the classes (dimension 1)
        confidence, predicted_class_id = torch.max(probabilities, dim=1) # <-- The FIX is here (dim=1)
    
    label_map = {0: "Real", 1: "Fake (Synthesis)"}
    # .item() will now work because predicted_class_id is a single-element tensor
    predicted_label = label_map[predicted_class_id.item()]
    
    print("\n--- Prediction Result (Video) ---")
    print(f"Predicted Class: {predicted_label}")
    print(f"Confidence: {confidence.item() * 100:.2f}%")
    print("---------------------------------")

def main():
    # --- !!! EDIT THIS LINE WITH THE PATH TO YOUR VIDEO !!! ---
    video_to_predict_path = "D:\\Slantie@LDRP\\7th Semester\\NTRO\\Website\\Server\\id0_0001.mp4"
    # -----------------------------------------------------------

    if not os.path.exists(video_to_predict_path):
        print(f"Error: Video file not found at '{video_to_predict_path}'")
        return

    config = load_config('configs/config_lstm.yaml')
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading LSTM model from {config['best_model_save_path']}...")
    model = create_lstm_model(config)
    model.load_state_dict(torch.load(config['best_model_save_path'], map_location=device))
    model.to(device)
    
    processor = AutoProcessor.from_pretrained(config['base_model_path'])
    predict_video(video_to_predict_path, model, processor, config, device)

if __name__ == '__main__':
    main()