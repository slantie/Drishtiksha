# src/ml/models/lstm_detector_v2.py
import torch
import cv2
import numpy as np
# import os
import tempfile
import time
from PIL import Image
from transformers import AutoProcessor
# import torch.nn.functional as F
import matplotlib.pyplot as plt

from src.ml.base import BaseModel
from src.ml.utils import extract_frames
from src.training.model_lstm import create_lstm_model

class LSTMDetector(BaseModel):
    # ... (the __init__ and load methods remain exactly the same) ...
    def __init__(self, config):
        super().__init__(config)

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

    def predict(self, video_path: str):
        # ... (the original predict method for JSON results remains unchanged) ...
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

    # --- NEW METHOD FOR VISUALIZATION ---
    def predict_visualized(self, video_path: str) -> str:
        """
        Analyzes a video frame-by-frame and generates a new video file
        with a "suspicion" graph overlaid.

        Args:
            video_path: The path to the input video file.

        Returns:
            The path to the generated output video file.
        """
        self.model.eval()
        
        # --- Setup Video Capture ---
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video file for processing: {video_path}")
        
        # Get video properties for the output writer
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # --- Setup Video Writer ---
        # Create a temporary file to write the output video to
        output_temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        output_path = output_temp_file.name
        output_temp_file.close() # Close the file handle so VideoWriter can use the path

        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Define the codec
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        # --- Setup Matplotlib Plot ---
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(5, 2.5))
        line, = ax.plot([], [], marker='o', linestyle='-', color='#FF4136', markersize=2, linewidth=1)
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title("Suspicion Level", fontsize=10)
        ax.set_xlabel("Frame Number", fontsize=8)
        ax.set_ylabel("Fake Probability", fontsize=8)
        fig.tight_layout(pad=1.5)

        frame_scores = []
        
        print(f"Starting visualized analysis for video: {video_path}")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # --- Perform prediction on the single frame ---
            # Note: This uses the single-image forward pass of your model
            frame_tensor = self.processor(images=frame_pil, return_tensors="pt")['pixel_values'].to(self.device)
            with torch.no_grad():
                # Pass num_frames_per_video=1 to use the image classifier head
                logits_frame = self.model(frame_tensor, num_frames_per_video=1)
                prob_fake_frame = torch.sigmoid(logits_frame.squeeze()).item()
            
            frame_scores.append(prob_fake_frame)
            
            # --- Update and Render the Plot ---
            line.set_data(range(len(frame_scores)), frame_scores)
            if len(frame_scores) > ax.get_xlim()[1]:
                 ax.set_xlim(0, len(frame_scores) + 50) # Auto-expand x-axis
            
            fig.canvas.draw()
            buf = fig.canvas.buffer_rgba()
            plot_img_rgba = np.frombuffer(buf, dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (4,))
            plot_img_bgr = cv2.cvtColor(plot_img_rgba, cv2.COLOR_RGBA2BGR)
            
            # --- Combine Video Frame and Plot Image ---
            plot_h, plot_w, _ = plot_img_bgr.shape
            plot_aspect_ratio = plot_w / plot_h
            new_plot_h = int(frame_height * 0.25) # Plot takes 25% of the video height
            new_plot_w = int(new_plot_h * plot_aspect_ratio)
            
            resized_plot = cv2.resize(plot_img_bgr, (new_plot_w, new_plot_h))
            
            # Position the plot in the top-right corner
            y_offset = 20
            x_offset = frame_width - new_plot_w - 20
            
            # Create a semi-transparent black background for the plot
            overlay = frame.copy()
            cv2.rectangle(overlay, (x_offset, y_offset), (x_offset + new_plot_w, y_offset + new_plot_h), (0,0,0), -1)
            frame_with_bg = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
            
            # Place the plot on the frame
            frame_with_bg[y_offset:y_offset+new_plot_h, x_offset:x_offset+new_plot_w] = resized_plot

            # Write the modified frame to the output video
            out.write(frame_with_bg)
        
        # --- Cleanup ---
        print(f"Finished visualized analysis. Output saved to: {output_path}")
        cap.release()
        out.release()
        plt.close(fig) # Important to free memory in a server environment
        
        return output_path