# src/ml/models/lstm_detector_v3.py
import torch
import cv2
import numpy as np
import json
import tempfile
import time
from PIL import Image
from transformers import AutoProcessor
import matplotlib.pyplot as plt
from collections import deque
from tqdm import tqdm

from src.ml.base import BaseModel
from src.ml.utils import extract_frames
from src.training.model_lstm import create_lstm_model


class LSTMDetectorV3(BaseModel):
    """
    Enhanced LSTM Detector with improved visualization and metrics collection.
    Features:
    - Rolling average trend analysis
    - Enhanced visualization with fill areas and trend lines
    - Comprehensive metrics collection
    - Progress tracking
    - Per-frame analysis capabilities
    """

    def __init__(self, config):
        super().__init__(config)
        self.rolling_window_size = config.get("rolling_window_size", 10)

    def load(self):
        """Loads the LSTM model, weights, and processor with memory optimization."""
        self.device = self.config.get("device", "cpu")
        model_def_config = self.config.get("model_definition", {})

        print(f"Loading LSTMDetectorV3 on device: {self.device}")

        try:
            # Memory optimization: Clear cache before loading
            if hasattr(torch.cuda, "empty_cache"):
                torch.cuda.empty_cache()

            # Load model with low memory usage
            print("Creating LSTM model...")
            self.model = create_lstm_model(model_def_config)

            print("Loading model weights...")
            # Load with map_location to ensure CPU loading
            state_dict = torch.load(
                self.config["model_path"],
                map_location="cpu",  # Force CPU loading
                weights_only=True,  # Security: only load weights
            )
            self.model.load_state_dict(state_dict)

            # Move to target device after loading
            print(f"Moving model to {self.device}...")
            self.model.to(self.device)
            self.model.eval()

            # Load processor with memory optimization
            print("Loading processor...")
            self.processor = AutoProcessor.from_pretrained(
                self.config["processor_path"],
                torch_dtype=torch.float32 if self.device == "cpu" else torch.float16,
                low_cpu_mem_usage=True,  # Enable memory optimization
            )

            print(
                f"✅ LSTMDetectorV3 '{self.config['name']}' loaded successfully on device '{self.device}'."
            )

        except Exception as e:
            print(f"❌ Error details: {str(e)}")
            print(f"❌ Error type: {type(e).__name__}")

            # Provide helpful error messages
            if "paging file" in str(e).lower() or "memory" in str(e).lower():
                print("💡 Memory issue detected. Suggestions:")
                print("   - Close other applications to free RAM")
                print("   - Increase virtual memory (paging file) size")
                print("   - Use a smaller model if available")
                print("   - Consider using GPU if available")

            raise RuntimeError(f"Failed to load LSTMDetectorV3 model: {e}")

    def predict(self, video_path: str, num_frames: int = None):
        """
        Standard prediction method for backward compatibility.
        Uses the original logic with frame extraction.

        Args:
            video_path: Path to the video file
            num_frames: Number of frames to extract (optional, uses config default if not provided)
        """
        start_time = time.time()

        # Use provided num_frames or fall back to config default
        frames_to_extract = (
            num_frames if num_frames is not None else self.config["num_frames"]
        )

        print(f"Extracting {frames_to_extract} frames from video...")
        frames = extract_frames(video_path, frames_to_extract)
        if not frames:
            raise ValueError("Could not extract frames from the video.")

        print(f"Processing {len(frames)} frames...")
        inputs = self.processor(images=frames, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)

        with torch.no_grad():
            logits = self.model(pixel_values, num_frames_per_video=len(frames))
            prob_fake = torch.sigmoid(logits.squeeze())
            predicted_class_id = (prob_fake > 0.5).long().item()
            confidence = (
                prob_fake.item() if predicted_class_id == 1 else 1 - prob_fake.item()
            )

        processing_time = time.time() - start_time
        label_map = {0: "REAL", 1: "FAKE"}

        print(
            f"Analysis completed in {processing_time:.2f}s - Result: {label_map.get(predicted_class_id)}"
        )

        return {
            "prediction": label_map.get(predicted_class_id, "UNKNOWN"),
            "confidence": confidence,
            "processing_time": processing_time,
        }

    def predict_with_metrics(self, video_path: str):
        """
        Enhanced prediction method that returns detailed frame-by-frame analysis.

        Returns:
            dict: Contains prediction, confidence, processing_time, and detailed metrics
        """
        start_time = time.time()

        # Get frame-by-frame analysis
        frame_scores, rolling_avg_scores = self._analyze_video_frames(video_path)

        if not frame_scores:
            raise ValueError("Could not analyze frames from the video.")

        # Calculate final metrics
        final_avg_score = np.mean(frame_scores)
        predicted_class_id = 1 if final_avg_score > 0.5 else 0
        confidence = final_avg_score if predicted_class_id == 1 else 1 - final_avg_score

        processing_time = time.time() - start_time
        label_map = {0: "REAL", 1: "FAKE"}

        return {
            "prediction": label_map.get(predicted_class_id, "UNKNOWN"),
            "confidence": confidence,
            "processing_time": processing_time,
            "metrics": {
                "frame_count": len(frame_scores),
                "per_frame_scores": frame_scores,
                "rolling_average_scores": rolling_avg_scores,
                "final_average_score": final_avg_score,
                "max_score": max(frame_scores),
                "min_score": min(frame_scores),
                "score_variance": np.var(frame_scores),
                "suspicious_frames_count": sum(
                    1 for score in frame_scores if score > 0.5
                ),
                "suspicious_frames_percentage": (
                    sum(1 for score in frame_scores if score > 0.5) / len(frame_scores)
                )
                * 100,
            },
        }

    def _analyze_video_frames(self, video_path: str):
        """
        Analyzes video frame by frame and returns scores.

        Returns:
            tuple: (frame_scores, rolling_avg_scores)
        """
        self.model.eval()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video file for processing: {video_path}")

        frame_scores = []
        rolling_avg_scores = []
        rolling_window = deque(maxlen=self.rolling_window_size)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Analyzing {total_frames} frames...")

        for frame_count in tqdm(range(total_frames), desc="Processing frames"):
            ret, frame = cap.read()
            if not ret:
                break

            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Perform prediction on single frame
            frame_tensor = self.processor(images=frame_pil, return_tensors="pt")[
                "pixel_values"
            ].to(self.device)
            with torch.no_grad():
                # Use single frame prediction
                logits_frame = self.model(frame_tensor, num_frames_per_video=1)
                prob_fake_frame = torch.sigmoid(logits_frame.squeeze()).item()

            frame_scores.append(prob_fake_frame)
            rolling_window.append(prob_fake_frame)
            rolling_avg_scores.append(np.mean(rolling_window))

        cap.release()
        return frame_scores, rolling_avg_scores

    def predict_visualized(self, video_path: str) -> str:
        """
        Enhanced visualization method with improved graphics and analysis.

        Args:
            video_path: The path to the input video file.

        Returns:
            The path to the generated output video file.
        """
        self.model.eval()

        # Setup Video Capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video file for processing: {video_path}")

        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Setup Video Writer
        output_temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        output_path = output_temp_file.name
        output_temp_file.close()

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        # Setup Enhanced Matplotlib Plot
        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(6, 2.5))

        frame_scores = []
        rolling_avg_scores = []
        rolling_window = deque(maxlen=self.rolling_window_size)

        print(f"Starting enhanced visualized analysis for video: {video_path}")

        for frame_count in tqdm(range(total_frames), desc="Creating visualization"):
            ret, frame = cap.read()
            if not ret:
                break

            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Perform prediction on single frame
            frame_tensor = self.processor(images=frame_pil, return_tensors="pt")[
                "pixel_values"
            ].to(self.device)
            with torch.no_grad():
                logits_frame = self.model(frame_tensor, num_frames_per_video=1)
                prob_fake_frame = torch.sigmoid(logits_frame.squeeze()).item()

            frame_scores.append(prob_fake_frame)
            rolling_window.append(prob_fake_frame)
            rolling_avg_scores.append(np.mean(rolling_window))

            # Enhanced Plot Rendering
            ax.clear()

            # Fill area under the curve
            ax.fill_between(
                range(len(frame_scores)),
                frame_scores,
                color="#FF4136",
                alpha=0.4,
                label="Frame Scores",
            )

            # Plot frame scores
            ax.plot(
                range(len(frame_scores)),
                frame_scores,
                marker=".",
                linestyle="-",
                color="#FF851B",
                markersize=2,
                alpha=0.7,
                label="Individual Frames",
            )

            # Plot rolling average trend
            ax.plot(
                range(len(rolling_avg_scores)),
                rolling_avg_scores,
                linestyle="-",
                color="#7FDBFF",
                linewidth=2.5,
                label=f"Trend ({self.rolling_window_size}f avg)",
            )

            # Add threshold line
            ax.axhline(y=0.5, color="white", linestyle="--", linewidth=1, alpha=0.7)
            ax.text(
                ax.get_xlim()[1] * 0.01,
                0.52,
                "Threshold",
                fontsize=8,
                color="white",
                alpha=0.7,
            )

            # Styling
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlim(0, max(100, len(frame_scores) + 10))
            ax.set_title("Live Suspicion Analysis", fontsize=12)
            ax.set_xlabel("Frame", fontsize=9)
            ax.set_ylabel("Fake Probability", fontsize=9)
            ax.legend(fontsize=8, loc="upper left")
            ax.grid(True, linestyle=":", alpha=0.3)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            fig.tight_layout(pad=1.5)

            # Convert plot to image
            fig.canvas.draw()
            buf = fig.canvas.buffer_rgba()
            plot_img_rgba = np.frombuffer(buf, dtype=np.uint8).reshape(
                fig.canvas.get_width_height()[::-1] + (4,)
            )
            plot_img_bgr = cv2.cvtColor(plot_img_rgba, cv2.COLOR_RGBA2BGR)

            # Resize and position plot
            plot_h, plot_w, _ = plot_img_bgr.shape
            new_plot_h = int(frame_height * 0.3)  # 30% of frame height
            new_plot_w = int(new_plot_h * (plot_w / plot_h))
            resized_plot = cv2.resize(plot_img_bgr, (new_plot_w, new_plot_h))

            # Position plot in top-right corner
            y_offset, x_offset = 10, frame_width - new_plot_w - 10
            frame[
                y_offset : y_offset + new_plot_h, x_offset : x_offset + new_plot_w
            ] = resized_plot

            # Add enhanced score bar at bottom
            bar_height = 40
            bar_y = frame_height - bar_height
            bar_color_g = int(255 * (1 - prob_fake_frame))
            bar_color_r = int(255 * prob_fake_frame)
            cv2.rectangle(
                frame,
                (0, bar_y),
                (frame_width, frame_height),
                (0, bar_color_g, bar_color_r),
                -1,
            )

            # Add text overlay
            score_text = (
                f"Current: {prob_fake_frame:.3f} | Avg: {np.mean(rolling_window):.3f}"
            )
            cv2.putText(
                frame,
                score_text,
                (15, frame_height - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            # Add frame counter
            frame_text = f"Frame: {frame_count + 1}/{total_frames}"
            cv2.putText(
                frame,
                frame_text,
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            out.write(frame)

        # Cleanup
        cap.release()
        out.release()
        plt.close(fig)

        print(f"Enhanced visualized analysis completed. Output saved to: {output_path}")
        return output_path

    def save_analysis_metrics(self, video_path: str, output_dir: str = None) -> str:
        """
        Analyzes video and saves comprehensive metrics to JSON file.

        Args:
            video_path: Path to the video file
            output_dir: Directory to save metrics (optional, uses temp if not provided)

        Returns:
            Path to the saved JSON metrics file
        """
        if output_dir is None:
            output_dir = tempfile.gettempdir()

        # Get detailed analysis
        analysis_result = self.predict_with_metrics(video_path)

        # Prepare metrics for JSON export
        video_name = video_path.split("/")[-1].split("\\")[
            -1
        ]  # Cross-platform basename
        metrics = {
            "video_file": video_name,
            "analysis_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "final_verdict": analysis_result["prediction"],
            "final_confidence": f"{analysis_result['confidence'] * 100:.2f}%",
            "processing_time_seconds": analysis_result["processing_time"],
            "model_info": {
                "name": self.config.get("name", "LSTM_SIGLIP_V3"),
                "device": self.device,
                "rolling_window_size": self.rolling_window_size,
            },
            "detailed_metrics": analysis_result["metrics"],
        }

        # Save to JSON
        json_filename = f"METRICS_{video_name.split('.')[0]}.json"
        json_save_path = f"{output_dir}/{json_filename}"

        with open(json_save_path, "w") as f:
            json.dump(metrics, f, indent=4)

        print(f"Analysis metrics saved to: {json_save_path}")
        return json_save_path

    def get_frame_analysis_summary(self, video_path: str) -> dict:
        """
        Quick analysis summary without creating visualization.
        Useful for API responses.

        Returns:
            dict: Summary statistics of the video analysis
        """
        frame_scores, rolling_avg_scores = self._analyze_video_frames(video_path)

        if not frame_scores:
            return {"error": "Could not analyze video frames"}

        final_avg_score = np.mean(frame_scores)
        predicted_class_id = 1 if final_avg_score > 0.5 else 0
        label_map = {0: "REAL", 1: "FAKE"}

        return {
            "prediction": label_map[predicted_class_id],
            "confidence": final_avg_score
            if predicted_class_id == 1
            else 1 - final_avg_score,
            "frame_count": len(frame_scores),
            "suspicious_frames": sum(1 for score in frame_scores if score > 0.5),
            "average_suspicion": final_avg_score,
            "max_suspicion": max(frame_scores),
            "min_suspicion": min(frame_scores),
            "suspicion_variance": float(np.var(frame_scores)),
        }
