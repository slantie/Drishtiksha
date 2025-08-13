# src/ml/models/lstm_detector.py

import torch
from transformers import AutoProcessor
import time
from typing import Dict, Any

from src.ml.base import BaseModel
from src.ml.utils import extract_frames
# Import the model definition from the training package
from src.training.model_lstm import create_lstm_model

class LSTMDetector(BaseModel):
    """Concrete implementation of the SigLIP-LSTM detector for inference."""

    def load(self):
        """Loads the LSTM model, weights, and processor."""
        self.device = self.config.get("device", "cpu")
        model_def_config = self.config.get("model_definition", {})

        try:
            self.model = create_lstm_model(model_def_config)
            self.model.load_state_dict(torch.load(self.config["model_path"], map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            
            self.processor = AutoProcessor.from_pretrained(self.config["processor_path"])
            print(f"✅ LSTMDetector '{self.config['name']}' loaded successfully on device '{self.device}'.")

        except Exception as e:
            raise RuntimeError(f"Failed to load LSTMDetector model: {e}")

    def predict(self, video_path: str) -> Dict[str, Any]:
        """Analyzes a video, using sigmoid for binary prediction."""
        start_time = time.time()
        
        frames = extract_frames(video_path, self.config["num_frames"])
        if not frames:
            raise ValueError("Could not extract frames from the video.")

        inputs = self.processor(images=frames, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(self.device)

        with torch.no_grad():
            logits = self.model(pixel_values, num_frames_per_video=self.config["num_frames"])
            prob_fake = torch.sigmoid(logits.squeeze())
            predicted_class_id = (prob_fake > 0.5).long().item()
            confidence = prob_fake.item() if predicted_class_id == 1 else 1 - prob_fake.item()

        processing_time = time.time() - start_time
        label_map = {0: "REAL", 1: "FAKE"}
        
        return {
            "prediction": label_map.get(predicted_class_id, "UNKNOWN"),
            "confidence": confidence,
            "processing_time": processing_time,
        }