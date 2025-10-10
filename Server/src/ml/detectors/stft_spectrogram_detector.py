# Server/src/ml/models/stft_spectrogram_detector.py

import os
import io
import time
import torch
import librosa
import logging
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
from PIL import Image
from torchvision import transforms
from typing import Tuple

from src.ml.base import BaseModel, AnalysisResult
from src.app.schemas import AudioAnalysisResult, AudioProperties, PitchAnalysis, EnergyAnalysis, SpectralAnalysis, AudioVisualization
from src.config import STFTSpectrogramCNNConfig
from src.ml.architectures.stft_spectrogram_cnn import create_stft_spectrogram_model
from src.ml.exceptions import MediaProcessingError, InferenceError

logger = logging.getLogger(__name__)

class STFTSpectrogramCNNV1(BaseModel):
    """
    Detector for audio deepfakes using stacked STFT Spectrograms and a CNN.
    """
    config: STFTSpectrogramCNNConfig

    def load(self) -> None:
        start_time = time.time()
        try:
            self.model = create_stft_spectrogram_model(self.config.model_dump())
            state_dict = torch.load(self.config.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()

            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((self.config.img_height, self.config.img_width), antialias=True),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            load_time = time.time() - start_time
            logger.info(f"Loaded Model: '{self.config.model_name}' | Device: '{self.device}' | Time: {load_time:.2f}s.")
        except Exception as e:
            logger.error(f"Failed to load model '{self.config.model_name}': {e}", exc_info=True)
            raise RuntimeError(f"Failed to load model '{self.config.model_name}'") from e

    def _get_audio_chunks(self, media_path: str) -> Tuple[np.ndarray, int]:
        """Standardizes and prepares the audio signal."""
        try:
            audio = AudioSegment.from_file(media_path)
            audio = audio.set_channels(1).set_frame_rate(self.config.sampling_rate)
            wav_buffer = io.BytesIO()
            audio.export(wav_buffer, format="wav")
            wav_buffer.seek(0)
            y, sr = librosa.load(wav_buffer, sr=self.config.sampling_rate)
            return y, sr
        except Exception as e:
            raise MediaProcessingError(f"Failed to load or process audio file: {e}")

    def _chunk_to_tensor(self, audio_chunk: np.ndarray, sr: int) -> torch.Tensor:
        """Replicates the STFT preprocessing pipeline for a single audio chunk."""
        stft_wide = librosa.stft(audio_chunk, n_fft=self.config.n_fft_wide, hop_length=self.config.hop_length, window="hann")
        spec_wide = librosa.amplitude_to_db(np.abs(stft_wide), ref=np.max)
        stft_narrow = librosa.stft(audio_chunk, n_fft=self.config.n_fft_narrow, hop_length=self.config.hop_length, window="hann")
        spec_narrow = librosa.amplitude_to_db(np.abs(stft_narrow), ref=np.max)

        fig, axs = plt.subplots(2, 1, figsize=(6, 6), gridspec_kw={'hspace': 0})
        librosa.display.specshow(spec_wide, sr=sr, hop_length=self.config.hop_length, ax=axs[0])
        librosa.display.specshow(spec_narrow, sr=sr, hop_length=self.config.hop_length, ax=axs[1])
        for ax in axs: ax.axis("off")
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches="tight", pad_inches=0, dpi=self.config.dpi)
        plt.close(fig); buf.seek(0)
        
        image = Image.open(buf).convert("RGB")
        tensor = self.transform(image)
        buf.close()
        return tensor

    def analyze(self, media_path: str, **kwargs) -> AnalysisResult:
        start_time = time.time()
        y, sr = self._get_audio_chunks(media_path)

        chunk_len = int(self.config.chunk_duration_s * sr)
        if len(y) < chunk_len:
             raise MediaProcessingError(f"Audio duration is less than the required chunk size of {self.config.chunk_duration_s}s.")

        if len(y) % chunk_len != 0:
            pad_len = chunk_len - (len(y) % chunk_len)
            y = np.pad(y, (0, pad_len), mode="constant")

        num_chunks = len(y) // chunk_len
        chunk_predictions = []

        try:
            with torch.no_grad():
                for i in range(num_chunks):
                    chunk = y[i * chunk_len : (i + 1) * chunk_len]
                    tensor = self._chunk_to_tensor(chunk, sr).unsqueeze(0).to(self.device)
                    output = self.model(tensor)
                    probabilities = torch.softmax(output, dim=1)[0]
                    chunk_predictions.append(probabilities.cpu().numpy())
        except Exception as e:
            raise InferenceError(f"Error during model inference for chunks: {e}")

        avg_probs = np.mean(chunk_predictions, axis=0)
        prob_real, prob_fake = avg_probs[0], avg_probs[1]
        
        prediction = "REAL" if prob_real >= prob_fake else "FAKE"
        confidence = prob_real if prediction == "REAL" else prob_fake

        # --- Generate final visualization and metrics from the full audio ---
        first_chunk = y[:chunk_len]
        stft_wide = librosa.stft(first_chunk, n_fft=self.config.n_fft_wide, hop_length=self.config.hop_length)
        spec_wide_db = librosa.amplitude_to_db(np.abs(stft_wide), ref=np.max)
        
        fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
        librosa.display.specshow(spec_wide_db, sr=sr, ax=ax)
        ax.set_axis_off()
        fig.tight_layout(pad=0)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png", prefix="spec_") as tmp:
            plt.savefig(tmp.name, format='png')
            spectrogram_path = tmp.name
        plt.close(fig)

        pitch_values, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        valid_pitch = pitch_values[~np.isnan(pitch_values)]
        
        return AudioAnalysisResult(
            prediction=prediction,
            confidence=float(confidence),
            processing_time=time.time() - start_time,
            properties=AudioProperties(duration_seconds=len(y)/sr, sample_rate=sr, channels=1),
            pitch=PitchAnalysis(
                mean_pitch_hz=float(np.mean(valid_pitch)) if len(valid_pitch) > 0 else None,
                pitch_stability_score=max(0.0, 1.0 - (np.std(valid_pitch) / 100.0)) if len(valid_pitch) > 1 else None
            ),
            energy=EnergyAnalysis(
                rms_energy=float(np.mean(librosa.feature.rms(y=y))),
                silence_ratio=1.0 - (len(librosa.effects.split(y, top_db=40)) / (len(y)/sr))
            ),
            spectral=SpectralAnalysis(
                spectral_centroid=float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))),
                spectral_contrast=float(np.mean(librosa.feature.spectral_contrast(y=y, sr=sr)))
            ),
            visualization=AudioVisualization(spectrogram_url=spectrogram_path, spectrogram_data=spec_wide_db.tolist())
        )
