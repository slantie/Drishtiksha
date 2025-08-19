# src/ml/models/scattering_wave_detector.py

import os
import io
import time
import torch
import logging
import librosa
import numpy as np
from pydub import AudioSegment
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from src.ml.base import BaseModel
from src.config import ScatteringWaveV1Config
from src.ml.architectures.scattering_wave_classifier import create_scattering_wave_model
from src.ml.event_publisher import publish_progress

logger = logging.getLogger(__name__)

class ScatteringWaveV1(BaseModel):
    """
    Audio deepfake detector that analyzes the Mel Spectrogram of an audio clip
    using a Wavelet Scattering Transform model.
    """
    config: ScatteringWaveV1Config

    def __init__(self, config: ScatteringWaveV1Config):
        super().__init__(config)
        self.transform: transforms.Compose

    def load(self) -> None:
        start_time = time.time()
        try:
            model_params = self.config.model_dump()
            self.model = create_scattering_wave_model(model_params)
            
            state_dict = torch.load(self.config.model_path, map_location=torch.device('cpu'))
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()

            self.transform = transforms.Compose([
                transforms.Resize(self.config.image_size, antialias=True),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
            ])
            load_time = time.time() - start_time
            logger.info(f"✅ Loaded Model: '{self.config.class_name}'\t | Device: '{self.device}'\t | Time: {load_time:.2f}s.")
        except Exception as e:
            logger.error(f"Failed to load model '{self.config.class_name}': {e}", exc_info=True)
            raise RuntimeError(f"Failed to load model '{self.config.class_name}'") from e

    def _extract_and_process_audio(self, video_path: str, video_id: str = None, user_id: str = None) -> torch.Tensor:
        """
        Full audio processing pipeline:
        1. Extracts audio from video using pydub.
        2. Standardizes to 16kHz mono WAV in-memory.
        3. Loads, pads/trims, and pre-emphasizes with librosa.
        4. Generates a Mel Spectrogram image in-memory.
        5. Converts the image to a PyTorch tensor for inference.
        """
        # 1. Extract and standardize audio
        if video_id: publish_progress({"videoId": video_id, "userId": user_id, "event": "AUDIO_EXTRACTION_START"})
        try:
            audio_segment = AudioSegment.from_file(video_path)
            audio_segment = audio_segment.set_channels(1).set_frame_rate(self.config.sampling_rate)
            wav_buffer = io.BytesIO()
            audio_segment.export(wav_buffer, format="wav")
            wav_buffer.seek(0)
        except Exception as e:
            raise IOError(f"Pydub failed to extract or process audio from video: {e}")
        if video_id: publish_progress({"videoId": video_id, "userId": user_id, "event": "AUDIO_EXTRACTION_COMPLETE"})
        
        # 2. Load and preprocess waveform
        try:
            y, sr = librosa.load(wav_buffer, sr=self.config.sampling_rate)
            target_length = int(sr * self.config.duration_seconds)
            if len(y) < target_length:
                y = np.pad(y, (0, target_length - len(y)), 'constant')
            else:
                y = y[:target_length]
            y_preemphasized = librosa.effects.preemphasis(y)
        except Exception as e:
            raise ValueError(f"Librosa failed to process the audio waveform: {e}")

        # 3. Generate Mel Spectrogram image in memory
        if video_id: publish_progress({"videoId": video_id, "userId": user_id, "event": "SPECTROGRAM_GENERATION_START"})
        mel_spec = librosa.feature.melspectrogram(y=y_preemphasized, sr=sr, n_fft=2048, hop_length=512, n_mels=256)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        fig = plt.figure(figsize=(4, 4), dpi=100)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        librosa.display.specshow(mel_spec_db, sr=sr, ax=ax)
        
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        img_buffer.seek(0)
        if video_id: publish_progress({"videoId": video_id, "userId": user_id, "event": "SPECTROGRAM_GENERATION_COMPLETE"})
        
        # 4. Convert image to tensor
        image = Image.open(img_buffer)
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        img_buffer.close()
        
        return tensor

    def predict_detailed(self, video_path: str, **kwargs) -> Dict[str, Any]:
        start_time = time.time()
        video_id = kwargs.get("video_id")
        user_id = kwargs.get("user_id")

        try:
            tensor = self._extract_and_process_audio(video_path, video_id, user_id)
        except (IOError, ValueError) as e:
            return {
                "prediction": "REAL", "confidence": 0.9, "processing_time": time.time() - start_time,
                "metrics": {"average_real_score": 0.9, "average_fake_score": 0.1},
                "note": f"Audio analysis failed: {e}. Prediction is a fallback."
            }
        
        with torch.no_grad():
            output = self.model(tensor)
            prob_real = torch.sigmoid(output).item()
        
        prob_fake = 1.0 - prob_real
        prediction = "REAL" if prob_real >= 0.5 else "FAKE"
        confidence = prob_real if prediction == "REAL" else prob_fake

        return {
            "prediction": prediction,
            "confidence": float(confidence),
            "processing_time": time.time() - start_time,
            "metrics": {
                "average_real_score": float(prob_real),
                "average_fake_score": float(prob_fake),
            }
        }

    def predict(self, video_path: str, **kwargs) -> Dict[str, Any]:
        result = self.predict_detailed(video_path, **kwargs)
        # Simplify the result for the quick analysis endpoint
        return {
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "processing_time": result["processing_time"],
            "note": result.get("note"),
        }