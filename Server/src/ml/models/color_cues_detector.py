# src/ml/models/color_cues_detector.py

import os
import time
import torch
import torch.nn as nn
import numpy as np
import cv2
import dlib
from tqdm import tqdm
from typing import Dict, Any, Optional, List
import tempfile
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for server

from src.ml.base import BaseModel


class LSTMClassifier(nn.Module):
    """
    ColorCues LSTM Classifier for deepfake detection using color histogram features.
    """

    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.5):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, seq_len, _, _ = x.shape
        x_flattened = x.view(batch_size, seq_len, -1)
        h_lstm, _ = self.lstm(x_flattened)
        h_lstm_last = h_lstm[:, -1, :]
        out = self.dropout(h_lstm_last)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out


class ColorCuesDetector(BaseModel):
    """
    ColorCues LSTM-based deepfake detector using color histogram analysis.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.sequence_length = config.get("sequence_length", 32)
        self.frames_per_video = config.get("frames_per_video", 150)
        self.histogram_bins = config.get("histogram_bins", 32)
        self.landmark_margin = config.get("landmark_margin", 20)
        self.rolling_window_size = config.get("rolling_window_size", 10)

        # Model components
        self.model = None
        self.dlib_detector = None
        self.dlib_predictor = None

    def load(self):
        """Load the ColorCues model, dlib face detector, and predictor."""
        try:
            print("Loading ColorCues LSTM detector...")

            # Set device
            self.device = self.config.get("device", "cpu")
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

            print(f"Using device: {self.device}")

            # Load dlib models
            dlib_model_path = self.config.get("dlib_model_path")
            if not dlib_model_path or not os.path.exists(dlib_model_path):
                raise FileNotFoundError(f"Dlib model not found at: {dlib_model_path}")

            self.dlib_detector = dlib.get_frontal_face_detector()
            self.dlib_predictor = dlib.shape_predictor(dlib_model_path)
            print("✅ Dlib face detector and predictor loaded successfully")

            # Initialize and load the model
            input_size = self.histogram_bins * self.histogram_bins
            hidden_size = self.config.get("hidden_size", 64)
            dropout = self.config.get("dropout", 0.5)

            self.model = LSTMClassifier(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=1,
                dropout=dropout,
            ).to(self.device)

            # Load model weights
            model_path = self.config["model_path"]
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model weights not found at: {model_path}")

            checkpoint = torch.load(
                model_path, map_location=self.device, weights_only=True
            )

            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)

            self.model.eval()
            print("✅ ColorCues LSTM model loaded successfully")

        except Exception as e:
            print(f"❌ Failed to load ColorCues detector: {str(e)}")
            print("Troubleshooting tips:")
            print("   - Ensure dlib model file exists")
            print("   - Check model weights file path")
            print("   - Verify device compatibility")
            raise RuntimeError(f"Failed to load ColorCues detector: {e}")

    def extract_rg_histogram(
        self, image: np.ndarray, bins: int
    ) -> Optional[np.ndarray]:
        """Extract R-G histogram features from face crop."""
        if image is None or image.size == 0:
            return None

        img_float = image.astype(np.float32) + 1e-6
        rgb_sum = np.sum(img_float, axis=2)
        rgb_sum[rgb_sum == 0] = 1

        r = img_float[:, :, 2] / rgb_sum
        g = img_float[:, :, 1] / rgb_sum

        hist, _, _ = np.histogram2d(
            r.flatten(), g.flatten(), bins=bins, range=[[0, 1], [0, 1]]
        )

        hist = cv2.normalize(hist, None, 0, 255, cv2.NORM_MINMAX)
        return hist.astype(np.float32)

    def extract_features_from_video(
        self, video_path: str
    ) -> Optional[List[np.ndarray]]:
        """Extract color histogram features from video frames."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        all_histograms = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Sample frames uniformly
        frame_indices = np.linspace(
            0, total_frames - 1, self.frames_per_video, dtype=int
        )

        for idx in tqdm(frame_indices, desc="Extracting color features", leave=False):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.dlib_detector(gray_frame, 1)

            if len(faces) > 0:
                # Use the first detected face
                face = faces[0]
                shape = self.dlib_predictor(gray_frame, face)
                landmarks = np.array([(p.x, p.y) for p in shape.parts()])

                # Extract face region with margin
                x1 = max(0, np.min(landmarks[:, 0]) - self.landmark_margin)
                y1 = max(0, np.min(landmarks[:, 1]) - self.landmark_margin)
                x2 = min(frame.shape[1], np.max(landmarks[:, 0]) + self.landmark_margin)
                y2 = min(frame.shape[0], np.max(landmarks[:, 1]) + self.landmark_margin)

                face_crop = frame[y1:y2, x1:x2]
                hist = self.extract_rg_histogram(face_crop, self.histogram_bins)

                if hist is not None:
                    all_histograms.append(hist)

        cap.release()
        return all_histograms if len(all_histograms) >= self.sequence_length else None

    def predict(self, video_path: str, num_frames: int = None) -> Dict[str, Any]:
        """
        Analyze a single video for deepfake detection using color cues.

        Args:
            video_path: Path to the video file
            num_frames: Not used for ColorCues (uses frames_per_video config)

        Returns:
            Dict containing prediction results
        """
        start_time = time.time()

        try:
            print(
                f"Analyzing video with ColorCues detector: {os.path.basename(video_path)}"
            )

            # Extract features
            all_histograms = self.extract_features_from_video(video_path)

            if all_histograms is None:
                raise ValueError(
                    "Could not extract sufficient color features from video"
                )

            # Create sub-sequences for temporal analysis
            sub_sequences = [
                all_histograms[i : i + self.sequence_length]
                for i in range(len(all_histograms) - self.sequence_length + 1)
            ]

            if not sub_sequences:
                raise ValueError(
                    "Could not create temporal sequences from extracted features"
                )

            # Convert to tensor and run inference
            sequences_tensor = torch.from_numpy(
                np.array(sub_sequences, dtype=np.float32)
            ).to(self.device)

            with torch.no_grad():
                predictions = self.model(sequences_tensor).squeeze().cpu().numpy()

            # Calculate final results
            if isinstance(predictions, np.ndarray):
                avg_fake_score = float(np.mean(predictions))
                max_fake_score = float(np.max(predictions))
                min_fake_score = float(np.min(predictions))
                score_variance = float(np.var(predictions))
            else:
                # Single prediction
                avg_fake_score = float(predictions)
                max_fake_score = avg_fake_score
                min_fake_score = avg_fake_score
                score_variance = 0.0

            predicted_class_id = 1 if avg_fake_score > 0.5 else 0
            confidence = (
                avg_fake_score if predicted_class_id == 1 else 1 - avg_fake_score
            )

            processing_time = time.time() - start_time
            label_map = {0: "REAL", 1: "FAKE"}

            result = {
                "prediction": label_map[predicted_class_id],
                "confidence": confidence,
                "processing_time": processing_time,
                "fake_probability": avg_fake_score,
                "max_fake_score": max_fake_score,
                "min_fake_score": min_fake_score,
                "score_variance": score_variance,
                "num_sequences": len(sub_sequences),
                "num_features_extracted": len(all_histograms),
            }

            print(
                f"ColorCues analysis completed: {label_map[predicted_class_id]} "
                f"(confidence: {confidence:.3f}, time: {processing_time:.2f}s)"
            )

            return result

        except Exception as e:
            processing_time = time.time() - start_time
            print(f"ColorCues analysis failed: {str(e)}")
            raise ValueError(f"ColorCues analysis failed: {str(e)}")

    def predict_with_metrics(self, video_path: str) -> Dict[str, Any]:
        """
        Prediction with detailed metrics for compatibility with /analyze/detailed endpoint.
        """
        try:
            # Get temporal analysis
            analysis = self.predict_with_temporal_analysis(video_path)
            temporal_data = analysis["temporal_analysis"]

            # Convert to format expected by DetailedAnalysisResult
            result = {
                "prediction": analysis["prediction"],
                "confidence": analysis["confidence"],
                "processing_time": analysis["processing_time"],
                "metrics": {
                    "frame_count": temporal_data[
                        "sequence_count"
                    ],  # Use sequence count as frame equivalent
                    "per_frame_scores": temporal_data["per_sequence_scores"],
                    "rolling_average_scores": temporal_data["rolling_averages"],
                    "final_average_score": temporal_data["avg_score"],
                    "max_score": temporal_data["max_score"],
                    "min_score": temporal_data["min_score"],
                    "score_variance": temporal_data["score_variance"],
                    "suspicious_frames_count": temporal_data["suspicious_sequences"],
                    "suspicious_frames_percentage": temporal_data[
                        "suspicious_percentage"
                    ],
                },
            }

            return result

        except Exception as e:
            print(f"ColorCues predict_with_metrics failed: {str(e)}")
            raise ValueError(f"ColorCues detailed analysis failed: {str(e)}")

    def predict_with_temporal_analysis(self, video_path: str) -> Dict[str, Any]:
        """
        Enhanced prediction with detailed temporal analysis.
        """
        start_time = time.time()

        try:
            all_histograms = self.extract_features_from_video(video_path)

            if all_histograms is None:
                raise ValueError("Could not extract sufficient features")

            sub_sequences = [
                all_histograms[i : i + self.sequence_length]
                for i in range(len(all_histograms) - self.sequence_length + 1)
            ]

            sequences_tensor = torch.from_numpy(
                np.array(sub_sequences, dtype=np.float32)
            ).to(self.device)

            with torch.no_grad():
                predictions = self.model(sequences_tensor).squeeze().cpu().numpy()

            if isinstance(predictions, np.ndarray):
                per_sequence_scores = predictions.tolist()
            else:
                per_sequence_scores = [float(predictions)]

            # Calculate rolling averages
            rolling_averages = []
            for i in range(len(per_sequence_scores)):
                start_idx = max(0, i - self.rolling_window_size + 1)
                rolling_avg = np.mean(per_sequence_scores[start_idx : i + 1])
                rolling_averages.append(rolling_avg)

            avg_fake_score = np.mean(per_sequence_scores)
            predicted_class_id = 1 if avg_fake_score > 0.5 else 0
            confidence = (
                avg_fake_score if predicted_class_id == 1 else 1 - avg_fake_score
            )

            processing_time = time.time() - start_time
            label_map = {0: "REAL", 1: "FAKE"}

            return {
                "prediction": label_map[predicted_class_id],
                "confidence": confidence,
                "processing_time": processing_time,
                "temporal_analysis": {
                    "per_sequence_scores": per_sequence_scores,
                    "rolling_averages": rolling_averages,
                    "sequence_count": len(per_sequence_scores),
                    "avg_score": avg_fake_score,
                    "max_score": float(np.max(per_sequence_scores)),
                    "min_score": float(np.min(per_sequence_scores)),
                    "score_variance": float(np.var(per_sequence_scores)),
                    "suspicious_sequences": sum(
                        1 for score in per_sequence_scores if score > 0.5
                    ),
                    "suspicious_percentage": (
                        sum(1 for score in per_sequence_scores if score > 0.5)
                        / len(per_sequence_scores)
                    )
                    * 100,
                },
            }

        except Exception as e:
            processing_time = time.time() - start_time
            raise ValueError(f"Temporal analysis failed: {str(e)}")

    def predict_visualized(self, video_path: str) -> str:
        """
        Generate video visualization of temporal analysis results.
        """
        try:
            # Get temporal analysis
            analysis = self.predict_with_temporal_analysis(video_path)
            temporal_data = analysis["temporal_analysis"]

            # Setup video properties for visualization
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise IOError(f"Could not open video file for processing: {video_path}")

            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Create output video file
            output_temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            output_path = output_temp_file.name
            output_temp_file.close()

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

            # Setup matplotlib figure for graph overlay
            plt.style.use("dark_background")
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
            plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.15)

            sequences = range(len(temporal_data["per_sequence_scores"]))

            print(f"Starting ColorCues visualization for video: {video_path}")

            for frame_idx in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break

                # Calculate current sequence index
                sequence_idx = min(
                    frame_idx // self.frames_per_video,
                    len(temporal_data["per_sequence_scores"]) - 1,
                )
                current_score = (
                    temporal_data["per_sequence_scores"][sequence_idx]
                    if sequence_idx < len(temporal_data["per_sequence_scores"])
                    else 0.0
                )

                # Clear and redraw plots
                ax1.clear()
                ax2.clear()

                # Plot 1: Temporal progression with current position
                ax1.plot(
                    sequences,
                    temporal_data["per_sequence_scores"],
                    "o-",
                    color="#FF6B6B",
                    linewidth=2,
                    markersize=4,
                    alpha=0.8,
                )
                ax1.plot(
                    sequences,
                    temporal_data["rolling_averages"],
                    "-",
                    color="#4ECDC4",
                    linewidth=3,
                    label=f"Rolling Average ({self.rolling_window_size})",
                )

                # Highlight current position
                if sequence_idx < len(temporal_data["per_sequence_scores"]):
                    ax1.plot(
                        sequence_idx,
                        current_score,
                        "o",
                        color="yellow",
                        markersize=8,
                        markeredgecolor="white",
                        markeredgewidth=2,
                    )

                ax1.axhline(
                    y=0.5, color="white", linestyle="--", alpha=0.7, label="Threshold"
                )
                ax1.fill_between(
                    sequences,
                    temporal_data["per_sequence_scores"],
                    alpha=0.3,
                    color="#FF6B6B",
                )

                ax1.set_title(
                    "ColorCues Temporal Analysis - Sequence Scores", fontsize=12, pad=15
                )
                ax1.set_xlabel("Sequence Number")
                ax1.set_ylabel("Fake Probability")
                ax1.set_ylim(0, 1)
                ax1.legend(fontsize=8)
                ax1.grid(True, alpha=0.3)

                # Plot 2: Score distribution with current value highlighted
                ax2.hist(
                    temporal_data["per_sequence_scores"],
                    bins=15,
                    color="#45B7D1",
                    alpha=0.7,
                    edgecolor="white",
                )
                ax2.axvline(
                    x=0.5, color="red", linestyle="--", linewidth=2, label="Threshold"
                )
                ax2.axvline(
                    x=temporal_data["avg_score"],
                    color="yellow",
                    linestyle="-",
                    linewidth=2,
                    label=f"Average ({temporal_data['avg_score']:.3f})",
                )

                # Highlight current score
                ax2.axvline(
                    x=current_score,
                    color="orange",
                    linestyle="-",
                    linewidth=3,
                    alpha=0.8,
                    label=f"Current ({current_score:.3f})",
                )

                ax2.set_title("Score Distribution", fontsize=12, pad=15)
                ax2.set_xlabel("Fake Probability")
                ax2.set_ylabel("Frequency")
                ax2.legend(fontsize=8)
                ax2.grid(True, alpha=0.3)

                # Add frame info text
                frame_text = (
                    f"Frame: {frame_idx + 1}/{total_frames}\n"
                    f"Sequence: {sequence_idx + 1}\n"
                    f"Current Score: {current_score:.3f}\n"
                    f"Prediction: {analysis['prediction']}\n"
                    f"Confidence: {analysis['confidence']:.3f}"
                )

                fig.text(
                    0.02,
                    0.98,
                    frame_text,
                    transform=fig.transFigure,
                    fontsize=10,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="black", alpha=0.8),
                )

                # Convert matplotlib figure to image
                fig.canvas.draw()
                # Use tobytes() instead of tostring_rgb() for newer matplotlib versions
                try:
                    canvas_buffer = fig.canvas.buffer_rgba()
                    graph_img = np.frombuffer(canvas_buffer, dtype=np.uint8)
                    graph_img = graph_img.reshape(
                        fig.canvas.get_width_height()[::-1] + (4,)
                    )
                    graph_img = cv2.cvtColor(graph_img, cv2.COLOR_RGBA2BGR)
                except AttributeError:
                    # Fallback for older matplotlib versions
                    graph_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                    graph_img = graph_img.reshape(
                        fig.canvas.get_width_height()[::-1] + (3,)
                    )
                    graph_img = cv2.cvtColor(graph_img, cv2.COLOR_RGB2BGR)

                # Resize graph to fit video frame
                graph_height = frame_height // 3
                graph_width = frame_width
                graph_resized = cv2.resize(graph_img, (graph_width, graph_height))

                # Create combined frame
                combined_frame = np.zeros(
                    (frame_height, frame_width, 3), dtype=np.uint8
                )

                # Place original video at top (2/3 of height)
                video_height = frame_height - graph_height
                frame_resized = cv2.resize(frame, (frame_width, video_height))
                combined_frame[:video_height, :] = frame_resized

                # Place graph at bottom (1/3 of height)
                combined_frame[video_height:, :] = graph_resized

                # Write frame to output video
                out.write(combined_frame)

            # Clean up
            cap.release()
            out.release()
            plt.close(fig)

            print(f"ColorCues visualization saved to: {output_path}")
            return output_path

        except Exception as e:
            print(f"Visualization generation failed: {str(e)}")
            raise ValueError(f"Visualization generation failed: {str(e)}")

    def get_frame_analysis_summary(self, video_path: str) -> dict:
        """
        Quick analysis summary without creating visualization.
        Useful for API responses.

        Returns:
            dict: Summary statistics of the video analysis
        """
        try:
            # Analyze all frames with temporal data
            analysis_result = self.predict_with_temporal_analysis(video_path)

            temporal_data = analysis_result.get("temporal_analysis", {})
            if not temporal_data:
                return {"error": "Could not analyze video frames"}

            analysis = analysis_result  # Use the full result as it contains prediction/confidence
            frame_scores = temporal_data.get("per_sequence_scores", [])

            if not frame_scores:
                return {"error": "No frame scores available"}

            return {
                "prediction": analysis.get("prediction", "UNKNOWN"),
                "confidence": analysis.get("confidence", 0.0),
                "frame_count": len(frame_scores),
                "suspicious_frames": temporal_data.get("suspicious_sequences", 0),
                "average_suspicion": temporal_data.get("avg_score", 0.0),
                "max_suspicion": max(frame_scores) if frame_scores else 0.0,
                "min_suspicion": min(frame_scores) if frame_scores else 0.0,
                "suspicion_variance": float(np.var(frame_scores))
                if frame_scores
                else 0.0,
                "sequence_count": temporal_data.get("sequence_count", 0),
                "suspicious_percentage": temporal_data.get(
                    "suspicious_percentage", 0.0
                ),
            }

        except Exception as e:
            print(f"Error in get_frame_analysis_summary: {e}")
            return {"error": f"Analysis failed: {str(e)}"}

    def get_info(self) -> Dict[str, Any]:
        """Return model information."""
        return {
            "model_name": "ColorCues LSTM Detector",
            "model_type": "color_histogram_lstm",
            "version": "1.0",
            "description": "LSTM-based deepfake detection using R-G color histogram analysis",
            "input_features": "Color histograms from face regions",
            "sequence_length": self.sequence_length,
            "frames_per_video": self.frames_per_video,
            "histogram_bins": self.histogram_bins,
            "device": self.device,
        }
