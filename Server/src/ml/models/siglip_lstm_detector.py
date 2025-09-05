# src/ml/models/siglip_lstm_detector.py

import os
import cv2
import time
import torch
import logging
import tempfile
import matplotlib
import numpy as np
from tqdm import tqdm
from collections import deque
import matplotlib.pyplot as plt
from transformers import AutoProcessor
from typing import Any, Dict, List, Tuple, Union

from src.ml.base import BaseModel, AnalysisResult
from src.app.schemas import VideoAnalysisResult, FramePrediction

from src.ml.event_publisher import event_publisher
from src.ml.utils import extract_frames, pad_frames
from src.ml.schemas import ProgressEvent, EventData
from src.ml.architectures.siglip_lstm import create_siglip_lstm_model
from src.config import SiglipLSTMv1Config, SiglipLSTMv3Config, SiglipLSTMv4Config

matplotlib.use("Agg")
logger = logging.getLogger(__name__)


class BaseSiglipLSTMDetector(BaseModel):
    """
    REFACTORED base class for all SigLIP+LSTM models.

    This class implements the unified `analyze` method, which orchestrates all
    stages of video processing: windowed analysis, final prediction, metric
    calculation, and visualization generation.
    """

    # Type hint config for better autocompletion in subclasses
    config: Union[SiglipLSTMv1Config, SiglipLSTMv3Config, SiglipLSTMv4Config]

    # --- Private Helper Methods ---

    def _get_final_prediction(self, media_path: str) -> Tuple[str, float, str]:
        """
        Performs a single, robust prediction on an evenly-spaced sequence of frames.
        This is used to get the final, authoritative prediction and confidence.
        """
        note = None
        
        frame_generator = extract_frames(media_path, self.config.num_frames)

        frames = pad_frames(frame_generator, self.config.num_frames)
        
        if not frames:
            logger.warning(f"Could not extract frames for final prediction from: {media_path}")
            return "REAL", 0.51, "Could not extract frames from video for final analysis."

        inputs = self.processor(images=frames, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(self.device)

        with torch.no_grad():
            logits = self.model(pixel_values, num_frames_per_video=self.config.num_frames)
            prob_fake = torch.sigmoid(logits.squeeze()).item()

        prediction = "FAKE" if prob_fake > 0.5 else "REAL"
        confidence = prob_fake if prediction == "FAKE" else 1 - prob_fake

        return prediction, confidence, note


    def _analyze_video_windows(self, media_path: str, **kwargs) -> Tuple[List[float], int]:
        """Analyzes the video in overlapping windows to get per-frame scores."""
        video_id = kwargs.get("video_id")
        user_id = kwargs.get("user_id")

        cap = cv2.VideoCapture(media_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video file: {media_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if total_frames == 0:
            return [], 0

        # Analyze 50 overlapping windows for a good temporal resolution
        num_windows = 50
        frame_scores = []
        seq_len = self.config.num_frames
        end_frame_indices = np.linspace(seq_len - 1, total_frames - 1, num_windows, dtype=int)

        for i, end_index in enumerate(tqdm(end_frame_indices, desc=f"Analyzing windows for {self.config.class_name}")):
            start_index = max(0, end_index - seq_len + 1)
            indices_to_extract = np.linspace(start_index, end_index, seq_len, dtype=int).tolist()
            frame_window = list(extract_frames(media_path, num_frames=seq_len, specific_indices=indices_to_extract))

            if not frame_window:
                continue

            inputs = self.processor(images=frame_window, return_tensors="pt")
            pixel_values = inputs['pixel_values'].to(self.device)

            with torch.no_grad():
                logits = self.model(pixel_values, num_frames_per_video=seq_len)
                prob_fake = torch.sigmoid(logits.squeeze()).item()

            frame_scores.append(prob_fake)

            if video_id and user_id:
                event_publisher.publish(ProgressEvent(
                    media_id=video_id,
                    user_id=user_id,
                    event="FRAME_ANALYSIS_PROGRESS",
                    message=f"Processed window {i + 1}/{num_windows}",
                    data=EventData(
                        model_name=self.config.class_name,
                        progress=i + 1,
                        total=num_windows
                    )
                ))

        return frame_scores, total_frames


    def _generate_visualization(
        self,
        media_path: str,
        frame_scores: List[float],
        total_frames: int,
        **kwargs
    ) -> str:
        """Generates a video with an overlay graph of the frame scores."""
        video_id = kwargs.get("video_id")
        user_id = kwargs.get("user_id")
        
        cap = cv2.VideoCapture(media_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

        output_temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        output_path = output_temp_file.name
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(6, 2.5))

        # Publish visualization start event
        if video_id and user_id:
            event_publisher.publish(ProgressEvent(
                media_id=video_id,
                user_id=user_id,
                event="FRAME_ANALYSIS_PROGRESS",
                message="Starting visualization generation",
                data=EventData(
                    model_name=self.config.class_name,
                    progress=0,
                    total=total_frames,
                    details={"phase": "visualization"}
                )
            ))

        try:
            for i in tqdm(range(total_frames), desc="Generating visualization"):
                ret, frame = cap.read()
                if not ret: break

                current_score_index = min(len(frame_scores) - 1, int((i / total_frames) * len(frame_scores)))
                score = frame_scores[current_score_index]

                ax.clear()
                ax.fill_between(range(len(frame_scores)), frame_scores, color="#FF4136", alpha=0.4)
                ax.plot(range(len(frame_scores)), frame_scores, color="#FF851B", linewidth=2)
                ax.plot(current_score_index, score, "o", color="yellow", markersize=8)
                ax.axhline(y=0.5, color="white", linestyle="--", alpha=0.7)
                ax.set_ylim(-0.05, 1.05)
                ax.set_xlim(0, len(frame_scores) - 1 if len(frame_scores) > 1 else 1)
                ax.set_title("Suspicion Analysis", fontsize=10)
                fig.tight_layout(pad=1.5)
                fig.canvas.draw()
                buf = fig.canvas.buffer_rgba()
                plot_img_rgba = np.frombuffer(buf, dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (4,))
                plot_img_bgr = cv2.cvtColor(plot_img_rgba, cv2.COLOR_RGBA2BGR)

                plot_h, plot_w, _ = plot_img_bgr.shape
                new_plot_h = int(frame_height * 0.3)
                new_plot_w = int(new_plot_h * (plot_w / plot_h))
                
                # FIX: Ensure the plot fits within frame boundaries
                if new_plot_w > frame_width - 20:  # Leave 10px margin on each side
                    new_plot_w = frame_width - 20
                    new_plot_h = int(new_plot_w * (plot_h / plot_w))
                
                # FIX: Ensure coordinates don't exceed frame boundaries
                x_offset = max(0, min(frame_width - new_plot_w - 10, frame_width - new_plot_w - 10))
                y_offset = 10
                
                # FIX: Double-check dimensions before assignment
                if (y_offset + new_plot_h <= frame_height and 
                    x_offset + new_plot_w <= frame_width and 
                    new_plot_h > 0 and new_plot_w > 0):
                    
                    resized_plot = cv2.resize(plot_img_bgr, (new_plot_w, new_plot_h))
                    frame[y_offset:y_offset + new_plot_h, x_offset:x_offset + new_plot_w] = resized_plot

                green, yellow, red = np.array([0, 255, 0]), np.array([0, 255, 255]), np.array([0, 0, 255])
                interp = score * 2 if score < 0.5 else (score - 0.5) * 2
                color_np = green * (1 - interp) + yellow * interp if score < 0.5 else yellow * (1 - interp) + red * interp
                color = tuple(map(int, color_np))

                bar_height = 40
                cv2.rectangle(frame, (0, frame_height - bar_height), (frame_width, frame_height), color, -1)
                cv2.putText(frame, f"Live Frame Suspicion: {score:.2f}", (15, frame_height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

                out.write(frame)
                
                # Publish visualization progress updates
                if (i + 1) % 100 == 0 and video_id and user_id:
                    event_publisher.publish(ProgressEvent(
                        media_id=video_id,
                        user_id=user_id,
                        event="FRAME_ANALYSIS_PROGRESS",
                        message=f"Generating visualization: {i + 1}/{total_frames} frames processed",
                        data=EventData(
                            model_name=self.config.class_name,
                            progress=i + 1,
                            total=total_frames,
                            details={"phase": "visualization"}
                        )
                    ))
        finally:
            cap.release()
            out.release()
            plt.close(fig)
            
        # Publish visualization completion event
        if video_id and user_id:
            event_publisher.publish(ProgressEvent(
                media_id=video_id,
                user_id=user_id,
                event="FRAME_ANALYSIS_PROGRESS",
                message="Visualization generation completed",
                data=EventData(
                    model_name=self.config.class_name,
                    progress=total_frames,
                    total=total_frames,
                    details={"phase": "visualization_complete"}
                )
            ))

        return output_path

    # --- Public API Method ---

    def analyze(self, media_path: str, **kwargs) -> AnalysisResult:
        """
        The single, unified entry point for running a comprehensive analysis.
        """
        start_time = time.time()
        video_id = kwargs.get("video_id")
        user_id = kwargs.get("user_id")

        # Publish analysis start event
        if video_id and user_id:
            event_publisher.publish(ProgressEvent(
                media_id=video_id,
                user_id=user_id,
                event="FRAME_ANALYSIS_PROGRESS",
                message=f"Starting analysis with {self.config.class_name}",
                data=EventData(
                    model_name=self.config.class_name,
                    progress=0,
                    total=None,
                    details={"phase": "initialization"}
                )
            ))

        try:
            # 1. Perform detailed window-by-window analysis to get temporal scores
            frame_scores, total_frames = self._analyze_video_windows(media_path, **kwargs)

            # 2. Get the final, authoritative prediction
            final_prediction, final_confidence, note = self._get_final_prediction(media_path)

            # 3. Handle fallback case if no frames were analyzed
            if not frame_scores:
                if video_id and user_id:
                    event_publisher.publish(ProgressEvent(
                        media_id=video_id,
                        user_id=user_id,
                        event="ANALYSIS_COMPLETE",
                        message=f"Analysis completed with fallback: {final_prediction} (confidence: {final_confidence:.3f})",
                        data=EventData(
                            model_name=self.config.class_name,
                            details={
                                "prediction": final_prediction,
                                "confidence": final_confidence,
                                "processing_time": time.time() - start_time,
                                "fallback": True
                            }
                        )
                    ))
                
                return VideoAnalysisResult(
                    prediction=final_prediction,
                    confidence=final_confidence,
                    processing_time=time.time() - start_time,
                    note=note or "Could not extract temporal features; result is based on a single analysis.",
                    frame_count=total_frames,
                    frames_analyzed=0,
                    frame_predictions=[],
                    metrics={},
                    visualization_path=None
                )

            # 4. Format frame-by-frame results into the standard schema
            frame_predictions = [
                FramePrediction(
                    index=i,
                    score=score,
                    prediction="FAKE" if score > 0.5 else "REAL"
                ) for i, score in enumerate(frame_scores)
            ]

            # 5. Calculate rolling averages and other metrics
            rolling_avg_scores = []
            if hasattr(self.config, 'rolling_window_size'):
                rolling_window = deque(maxlen=self.config.rolling_window_size)
                for score in frame_scores:
                    rolling_window.append(score)
                    rolling_avg_scores.append(np.mean(list(rolling_window)))

            metrics = {
                "rolling_average_scores": rolling_avg_scores,
                "final_average_score": np.mean(frame_scores),
                "max_score": max(frame_scores),
                "min_score": min(frame_scores),
                "score_variance": np.var(frame_scores),
                "suspicious_frames_count": sum(1 for s in frame_scores if s > 0.5),
            }

            # 6. Generate the visualization video
            visualization_path = self._generate_visualization(media_path, frame_scores, total_frames, **kwargs)

            # 7. Publish analysis completion
            if video_id and user_id:
                event_publisher.publish(ProgressEvent(
                    media_id=video_id,
                    user_id=user_id,
                    event="ANALYSIS_COMPLETE",
                    message=f"Analysis completed: {final_prediction} (confidence: {final_confidence:.3f})",
                    data=EventData(
                        model_name=self.config.class_name,
                        details={
                            "prediction": final_prediction,
                            "confidence": final_confidence,
                            "processing_time": time.time() - start_time,
                            "total_frames": total_frames,
                            "windows_analyzed": len(frame_scores)
                        }
                    )
                ))

            # 8. Assemble and return the final, comprehensive result object
            return VideoAnalysisResult(
                prediction=final_prediction,
                confidence=final_confidence,
                processing_time=time.time() - start_time,
                note=note,
                frame_count=total_frames,
                frames_analyzed=len(frame_scores),
                frame_predictions=frame_predictions,
                metrics=metrics,
                visualization_path=visualization_path
            )
            
        except Exception as e:
            # Publish analysis failure event
            if video_id and user_id:
                event_publisher.publish(ProgressEvent(
                    media_id=video_id,
                    user_id=user_id,
                    event="ANALYSIS_FAILED",
                    message=f"Analysis failed: {str(e)}",
                    data=EventData(
                        model_name=self.config.class_name,
                        details={"error": str(e), "processing_time": time.time() - start_time}
                    )
                ))
            raise


class SiglipLSTMV1(BaseSiglipLSTMDetector):
    config: SiglipLSTMv1Config

    def load(self) -> None:
        start_time = time.time()
        try:
            model_architecture_config = self.config.model_definition.model_dump()
            self.model = create_siglip_lstm_model(model_architecture_config)
            state_dict = torch.load(self.config.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            self.processor = AutoProcessor.from_pretrained(self.config.processor_path)
            logger.info(f"✅ Loaded Model: '{self.config.class_name}' | Device: '{self.device}' | Time: {time.time() - start_time:.2f}s.")
        except Exception as e:
            logger.error(f"Failed to load model '{self.config.class_name}': {e}", exc_info=True)
            raise RuntimeError(f"Failed to load model '{self.config.class_name}'") from e


class SiglipLSTMV3(BaseSiglipLSTMDetector):
    config: SiglipLSTMv3Config

    def load(self) -> None:
        start_time = time.time()
        try:
            model_architecture_config = self.config.model_definition.model_dump()
            self.model = create_siglip_lstm_model(model_architecture_config)
            state_dict = torch.load(self.config.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            self.processor = AutoProcessor.from_pretrained(self.config.processor_path)
            logger.info(f"✅ Loaded Model: '{self.config.class_name}' | Device: '{self.device}' | Time: {time.time() - start_time:.2f}s.")
        except Exception as e:
            logger.error(f"Failed to load model '{self.config.class_name}': {e}", exc_info=True)
            raise RuntimeError(f"Failed to load model '{self.config.class_name}'") from e


class SiglipLSTMV4(BaseSiglipLSTMDetector):
    config: SiglipLSTMv4Config

    def load(self) -> None:
        start_time = time.time()
        try:
            model_architecture_config = self.config.model_definition.model_dump()
            self.model = create_siglip_lstm_model(model_architecture_config)
            state_dict = torch.load(self.config.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            self.processor = AutoProcessor.from_pretrained(self.config.processor_path)
            logger.info(f"✅ Loaded Model: '{self.config.class_name}' | Device: '{self.device}' | Time: {time.time() - start_time:.2f}s.")
        except Exception as e:
            logger.error(f"Failed to load model '{self.config.class_name}': {e}", exc_info=True)
            raise RuntimeError(f"Failed to load model '{self.config.class_name}'") from e