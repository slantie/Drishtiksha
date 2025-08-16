import matplotlib
matplotlib.use("Agg")

import logging
import time
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import dlib
import numpy as np
import torch
from collections import deque

from src.config import ColorCuesConfig
from src.ml.base import BaseModel
from src.ml.architectures.color_cues_lstm import create_color_cues_model

logger = logging.getLogger(__name__)

class ColorCuesLSTMV1(BaseModel):
    config: ColorCuesConfig

    def __init__(self, config: ColorCuesConfig):
        super().__init__(config)
        self.dlib_detector: Optional[Any] = None
        self.dlib_predictor: Optional[Any] = None

    def load(self) -> None:
        start_time = time.time()
        # logger.info(f"Loading model '{self.config.class_name}' on device '{self.device}'.")
        try:
            if not os.path.exists(self.config.dlib_model_path):
                raise FileNotFoundError(f"Dlib model not found at: {self.config.dlib_model_path}")
            self.dlib_detector = dlib.get_frontal_face_detector()
            self.dlib_predictor = dlib.shape_predictor(self.config.dlib_model_path)
            model_params = self.config.model_dump()
            self.model = create_color_cues_model(model_params).to(self.device)
            checkpoint = torch.load(self.config.model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # logger.info("Checkpoint detected. Extracting 'model_state_dict'.")
                state_dict_to_load = checkpoint['model_state_dict']
            else:
                state_dict_to_load = checkpoint
            self.model.load_state_dict(state_dict_to_load)
            self.model.eval()
            load_time = time.time() - start_time
            logger.info(
                f"✅ Loaded Model: '{self.config.class_name}'\t | Device: '{self.device}'\t | Time: {load_time:.2f} seconds."
            )
        except Exception as e:
            logger.error(f"Failed to load model '{self.config.class_name}': {e}", exc_info=True)
            raise RuntimeError(f"Failed to load model '{self.config.class_name}'") from e

    def _extract_features_from_video(self, video_path: str) -> List[np.ndarray]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): raise IOError(f"Could not open video file: {video_path}")
        all_histograms = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames - 1, self.config.frames_per_video, dtype=int)
        try:
            for idx in frame_indices:
                ret, frame = cap.read()
                if not ret: continue
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.dlib_detector(gray_frame, 1)
                if faces:
                    shape = self.dlib_predictor(gray_frame, faces[0])
                    landmarks = np.array([(p.x, p.y) for p in shape.parts()])
                    margin = self.config.landmark_margin
                    min_coords, max_coords = np.min(landmarks, axis=0), np.max(landmarks, axis=0)
                    x1, y1 = max(0, min_coords[0] - margin), max(0, min_coords[1] - margin)
                    x2, y2 = min(frame.shape[1], max_coords[0] + margin), min(frame.shape[0], max_coords[1] + margin)
                    face_crop = frame[y1:y2, x1:x2]
                    if face_crop.size == 0: continue
                    img_float = face_crop.astype(np.float32) + 1e-6
                    rgb_sum = np.sum(img_float, axis=2)
                    rgb_sum[rgb_sum == 0] = 1
                    r = img_float[:, :, 2] / rgb_sum
                    g = img_float[:, :, 1] / rgb_sum
                    hist, _, _ = np.histogram2d(r.flatten(), g.flatten(), bins=self.config.histogram_bins, range=[[0, 1], [0, 1]])
                    cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
                    all_histograms.append(hist.astype(np.float32))
        finally:
            cap.release()
        return all_histograms

    def _analyze_sequences(self, video_path: str) -> Tuple[list[float], list[float], Optional[str]]:
        all_histograms = self._extract_features_from_video(video_path)
        if len(all_histograms) < self.config.sequence_length:
            logger.warning(f"Could not extract sufficient features from '{os.path.basename(video_path)}'. Required {self.config.sequence_length}, found {len(all_histograms)}. Returning a default low score.")
            note = "Could not find a clear face for a sufficient duration; result is a low-confidence fallback."
            return [0.1], [0.1], note
        sub_sequences = [all_histograms[i: i + self.config.sequence_length] for i in range(len(all_histograms) - self.config.sequence_length + 1)]
        if not sub_sequences:
            note = "Could not create analysis sequences from extracted features."
            return [0.1], [0.1], note
        sequences_tensor = torch.from_numpy(np.array(sub_sequences)).to(self.device)
        with torch.no_grad():
            logits = self.model(sequences_tensor).squeeze()
            predictions = torch.sigmoid(logits).cpu().numpy()
        per_sequence_scores = predictions.tolist() if isinstance(predictions, np.ndarray) else [float(predictions)]
        rolling_averages, rolling_window = [], deque(maxlen=self.config.rolling_window_size)
        for score in per_sequence_scores:
            rolling_window.append(score)
            rolling_averages.append(np.mean(list(rolling_window)))
        return per_sequence_scores, rolling_averages, None

    def predict(self, video_path: str) -> Dict[str, Any]:
        start_time = time.time()
        sequence_scores, _, note = self._analyze_sequences(video_path)
        avg_score = np.mean(sequence_scores)
        prediction = "FAKE" if avg_score > 0.5 else "REAL"
        confidence = avg_score if prediction == "FAKE" else 1 - avg_score
        return {"prediction": prediction, "confidence": confidence, "processing_time": time.time() - start_time, "note": note}

    def predict_detailed(self, video_path: str) -> Dict[str, Any]:
        start_time = time.time()
        sequence_scores, rolling_avg_scores, note = self._analyze_sequences(video_path)
        avg_score = np.mean(sequence_scores)
        prediction = "FAKE" if avg_score > 0.5 else "REAL"
        confidence = avg_score if prediction == "FAKE" else 1 - avg_score
        
        # Enhanced metrics to match other models' format
        return {
            "prediction": prediction, 
            "confidence": confidence, 
            "processing_time": time.time() - start_time, 
            "metrics": {
                "sequence_count": len(sequence_scores),
                "frame_count": len(sequence_scores),  # Alias for compatibility
                "per_sequence_scores": sequence_scores,
                "per_frame_scores": sequence_scores,  # Alias for compatibility
                "rolling_average_scores": rolling_avg_scores,
                "final_average_score": avg_score,
                "max_score": max(sequence_scores) if sequence_scores else 0.0,
                "min_score": min(sequence_scores) if sequence_scores else 0.0,
                "score_variance": np.var(sequence_scores),
                "suspicious_sequences_count": sum(1 for s in sequence_scores if s > 0.5),
                "suspicious_frames_count": sum(1 for s in sequence_scores if s > 0.5),  # Alias for compatibility
                "analysis_type": "sequence_based"
            }, 
            "note": note
        }

    def predict_frames(self, video_path: str) -> Dict[str, Any]:
        detailed_result = self.predict_detailed(video_path)
        metrics = detailed_result["metrics"]
        frame_predictions = [{"frame_index": i, "score": score, "prediction": "FAKE" if score > 0.5 else "REAL"} for i, score in enumerate(metrics["per_sequence_scores"])]
        return {"overall_prediction": detailed_result["prediction"], "overall_confidence": detailed_result["confidence"], "processing_time": detailed_result["processing_time"], "frame_predictions": frame_predictions, "temporal_analysis": {"rolling_averages": metrics["rolling_average_scores"], "consistency_score": 1.0 - metrics["score_variance"], "note": "Analysis is based on overlapping sequences of frames."}, "note": detailed_result.get("note")}

    def predict_visual(self, video_path: str) -> str:
        import tempfile, matplotlib.pyplot as plt
        detailed_result = self.predict_detailed(video_path)
        metrics = detailed_result["metrics"]
        sequence_scores = metrics["per_sequence_scores"]
        cap = cv2.VideoCapture(video_path)
        frame_width, frame_height, fps, total_frames = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FPS)) or 30, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        output_temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        
        # --- THE FINAL FIX: Break the single line into multiple assignments ---
        output_path = output_temp_file.name
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(6, 2.5))
        try:
            for i in range(total_frames):
                ret, frame = cap.read()
                if not ret: break
                current_sequence_idx = 0
                if len(sequence_scores) > 1:
                    current_sequence_idx = min(len(sequence_scores) - 1, int((i / total_frames) * len(sequence_scores)))
                ax.clear()
                ax.plot(range(len(sequence_scores)), sequence_scores, color="#4ECDC4", linewidth=2)
                ax.fill_between(range(len(sequence_scores)), sequence_scores, color="#4ECDC4", alpha=0.3)
                ax.plot(current_sequence_idx, sequence_scores[current_sequence_idx], 'o', color="yellow", markersize=8)
                ax.axhline(y=0.5, color="white", linestyle="--", alpha=0.7)
                ax.set_ylim(-0.05, 1.05)
                ax.set_xlim(0, len(sequence_scores) if len(sequence_scores) > 1 else 1)
                ax.set_title("Color Cues Sequence Analysis", fontsize=10)
                fig.tight_layout(pad=1.5)
                fig.canvas.draw()
                buf = fig.canvas.buffer_rgba()
                plot_img_rgba = np.frombuffer(buf, dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (4,))
                plot_img_bgr = cv2.cvtColor(plot_img_rgba, cv2.COLOR_RGBA2BGR)
                plot_h, plot_w, _ = plot_img_bgr.shape
                new_plot_h = int(frame_height * 0.3)
                new_plot_w = int(new_plot_h * (plot_w / plot_h))
                resized_plot = cv2.resize(plot_img_bgr, (new_plot_w, new_plot_h))
                y_offset, x_offset = 10, frame_width - new_plot_w - 10
                frame[y_offset:y_offset + new_plot_h, x_offset:x_offset + new_plot_w] = resized_plot
                out.write(frame)
        finally:
            cap.release()
            out.release()
            plt.close(fig)
        return output_path