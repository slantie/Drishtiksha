# Server/src/ml/detectors/lip_fd_detector.py

import torch
import os
import shutil
import tempfile
import logging
import time
import numpy as np
import librosa
import matplotlib.pyplot as plt
import cv2
from moviepy.editor import VideoFileClip
from torchvision import transforms
from typing import List, Tuple

from src.ml.base import BaseModel, AnalysisResult
from src.app.schemas import VideoAnalysisResult, FramePrediction
from src.config import LipFDv1Config
from src.ml.exceptions import MediaProcessingError, InferenceError

# Correctly import from the new dependency location
from src.ml.dependencies.lipfd_model import build_model

logger = logging.getLogger(__name__)

class LipFDetectorV1(BaseModel):
    """
    Handler for the LipFD (Lips Are Lying) audio-visual deepfake detector.
    """
    config: LipFDv1Config

    def load(self) -> None:
        """Loads the LipFD model and its dependencies."""
        start_time = time.time()
        try:
            self.model = build_model(self.config.model_definition.arch)
            state_dict = torch.load(self.config.model_path, map_location="cpu")
            self.model.load_state_dict(state_dict["model"])
            self.model.to(self.device)
            self.model.eval()

            load_time = time.time() - start_time
            logger.info(f"✅ Loaded Model: '{self.config.model_name}' | Device: '{self.device}' | Time: {load_time:.2f}s.")
        
        except Exception as e:
            logger.error(f"Failed to load model '{self.config.model_name}': {e}", exc_info=True)
            raise RuntimeError(f"Failed to load model '{self.config.model_name}'") from e

    def _get_crops_and_normalized_tensors(self, img_tensor: torch.Tensor) -> Tuple[torch.Tensor, List[List[torch.Tensor]]]:
        """Internal helper to create model-ready crops from a combined image tensor."""
        crops = [[], [], []]
        img_tensor_batch = img_tensor.unsqueeze(0)
        
        frames_batch = [img_tensor_batch[:, :, 500:, i:i + 500] for i in range(0, 2500, 500)]
        
        crops_1_0x_list = [transforms.Resize((224, 224), antialias=True)(f) for f in frames_batch]
        
        crop_idx_face = (28, 196)
        crops_0_65x_list = [transforms.Resize((224, 224), antialias=True)(c[:, :, crop_idx_face[0]:crop_idx_face[1], crop_idx_face[0]:crop_idx_face[1]]) for c in crops_1_0x_list]
        
        crop_idx_lip = (61, 163)
        crops_0_45x_list = [transforms.Resize((224, 224), antialias=True)(c[:, :, crop_idx_lip[0]:crop_idx_lip[1], crop_idx_lip[0]:crop_idx_lip[1]]) for c in crops_1_0x_list]
        
        crops[0] = [t.squeeze(0) for t in crops_1_0x_list]
        crops[1] = [t.squeeze(0) for t in crops_0_65x_list]
        crops[2] = [t.squeeze(0) for t in crops_0_45x_list]
        
        img_resized = transforms.Resize((1120, 1120), antialias=True)(img_tensor)
        
        normalizer = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        
        return normalizer(img_resized), [[normalizer(t) for t in sub] for sub in crops]

    def _process_video_windows(self, video_path: str, temp_dir: str) -> List[Tuple[torch.Tensor, List[List[torch.Tensor]]]]:
        """Processes a video into audio-visual windows for the model."""
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        temp_audio_path = os.path.join(temp_dir, f"{video_name}.wav")
        
        try:
            with VideoFileClip(video_path) as clip:
                if clip.audio is None:
                    raise MediaProcessingError("Video file has no audio track, which is required for LipFD analysis.")
                clip.audio.write_audiofile(temp_audio_path, codec='pcm_s16le', logger=None)
        except Exception as e:
            raise MediaProcessingError(f"Could not extract audio track. Error: {e}")

        try:
            data, sr = librosa.load(temp_audio_path, sr=16000)
            if len(data) == 0: raise ValueError("Audio track is empty.")
            mel = librosa.power_to_db(librosa.feature.melspectrogram(y=data, sr=sr), ref=np.min)
            temp_spec_path = os.path.join(temp_dir, "mel.png")
            plt.imsave(temp_spec_path, mel, cmap='viridis')
            mel_img = (plt.imread(temp_spec_path) * 255).astype(np.uint8)
        except Exception as e:
            raise MediaProcessingError(f"Failed to generate spectrogram from audio. Error: {e}")

        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= self.config.model_definition.window_len:
            cap.release()
            raise MediaProcessingError("Video is too short for analysis window.")

        indices = np.linspace(0, frame_count - self.config.model_definition.window_len - 1, self.config.model_definition.n_extract, dtype=int)
        mapping = mel_img.shape[1] / frame_count
        
        processed_windows = []
        for start_idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
            frames = [cap.read()[1] for _ in range(self.config.model_definition.window_len) if cap.isOpened()]
            if len(frames) != self.config.model_definition.window_len: continue

            frames_resized = [cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (500, 500)) for f in frames]
            
            begin, end = int(start_idx * mapping), int((start_idx + self.config.model_definition.window_len) * mapping)
            if begin >= end: continue
            
            sub_mel = cv2.resize(mel_img[:, begin:end], (500 * self.config.model_definition.window_len, 500))
            sub_mel_rgb = sub_mel[:,:,:3] if sub_mel.shape[2] == 4 else sub_mel

            frames_concat = np.concatenate(frames_resized, axis=1)
            combined_img = np.concatenate((sub_mel_rgb, frames_concat), axis=0)
            
            img_tensor = torch.from_numpy(combined_img).float().permute(2, 0, 1)
            processed_windows.append(self._get_crops_and_normalized_tensors(img_tensor))
        
        cap.release()
        return processed_windows

    def analyze(self, media_path: str, **kwargs) -> AnalysisResult:
        start_time = time.time()
        temp_dir = tempfile.mkdtemp(prefix="lipfd_")
        
        try:
            processed_windows = self._process_video_windows(media_path, temp_dir)
            if not processed_windows:
                raise InferenceError("Could not process video into valid audio-visual windows.")

            video_scores = []
            with torch.no_grad():
                for img_tensor, crops in processed_windows:
                    img_tensor = img_tensor.unsqueeze(0).to(self.device)
                    crops = [[t.unsqueeze(0).to(self.device) for t in sublist] for sublist in crops]

                    features = self.model.get_features(img_tensor).to(self.device)
                    score, _, _ = self.model(crops, features)
                    video_scores.append(score.sigmoid().item())

            if not video_scores:
                raise InferenceError("Model inference on processed windows yielded no scores.")
            
            final_score = np.mean(video_scores)
            prediction = "FAKE" if final_score >= 0.5 else "REAL"
            confidence = final_score if prediction == "FAKE" else 1 - final_score

            frame_predictions = [
                FramePrediction(index=i, score=score, prediction="FAKE" if score > 0.5 else "REAL")
                for i, score in enumerate(video_scores)
            ]

            return VideoAnalysisResult(
                prediction=prediction,
                confidence=confidence,
                processing_time=time.time() - start_time,
                frames_analyzed=len(processed_windows),
                frame_predictions=frame_predictions,
                metrics={"final_average_score": final_score, "window_scores": video_scores}
            )

        finally:
            shutil.rmtree(temp_dir)
