# src/ml/detectors/mel_spectrogram_detector.py

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
from typing import Tuple, List, Union, Generator
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg") # Ensure non-interactive backend for server

from src.ml.base import BaseModel, AnalysisResult
from src.app.schemas import AudioAnalysisResult, AudioProperties, PitchAnalysis, EnergyAnalysis, SpectralAnalysis, AudioVisualization
# Import both config versions
from src.config import MelSpectrogramCNNConfig, MelSpectrogramCNNv3Config
from src.ml.architectures.mel_spectrogram_cnn import create_mel_spectrogram_model
from src.ml.exceptions import MediaProcessingError, InferenceError

logger = logging.getLogger(__name__)

# --- Base Class for Mel-Spectrogram Logic (for shared methods) ---

class BaseMelSpectrogramDetector(BaseModel):
    """Abstract base class for Mel-Spectrogram-based models."""
    config: Union[MelSpectrogramCNNConfig, MelSpectrogramCNNv3Config]

    def load(self) -> None:
        """Loads the model and defines the necessary image transforms."""
        start_time = time.time()
        try:
            # Architecture is shared: create_mel_spectrogram_model from mel_spectrogram_cnn.py
            self.model = create_mel_spectrogram_model(self.config.model_dump())
            
            # The .pth file should contain only the model's state dict
            state_dict = torch.load(self.config.model_path, map_location=self.device)
            # Handle possible dictionary wrapper from training (e.g., {'model_state_dict': ...})
            self.model.load_state_dict(state_dict.get('model_state_dict', state_dict)) 
            self.model.to(self.device)
            self.model.eval()

            self.transform = transforms.Compose([
                transforms.Resize((256, 256), antialias=True),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
            ])
            load_time = time.time() - start_time
            logger.info(f"Loaded Model: '{self.config.model_name}' | Device: '{self.device}' | Time: {load_time:.2f}s.")
        except Exception as e:
            logger.error(f"Failed to load model '{self.config.model_name}': {e}", exc_info=True)
            raise RuntimeError(f"Failed to load model '{self.config.model_name}'") from e

    def _get_audio_chunks(self, media_path: str) -> Tuple[np.ndarray, int]:
        """Standardizes audio to mono, target SR, and returns waveform."""
        try:
            audio = AudioSegment.from_file(media_path)
            audio = audio.set_channels(1).set_frame_rate(self.config.sampling_rate)
            wav_buffer = io.BytesIO()
            audio.export(wav_buffer, format="wav")
            wav_buffer.seek(0)
            y, sr = librosa.load(wav_buffer, sr=self.config.sampling_rate, mono=True)
            return y, sr
        except Exception as e:
            raise MediaProcessingError(f"Failed to load or process audio file: {e}")

    def _get_audio_window_generator(self, y: np.ndarray, sr: int) -> Generator[np.ndarray, None, None]:
        """Generates audio chunks with/without overlap based on config type."""
        chunk_len = int(self.config.chunk_duration_s * sr)
        
        if isinstance(self.config, MelSpectrogramCNNv3Config):
            chunk_overlap = int(self.config.chunk_overlap_s * sr)
            step_size = max(1, chunk_len - chunk_overlap)
        else:
            # Default to non-overlapping for non-V3 models (V1, V2)
            step_size = chunk_len

        start = 0
        while start + chunk_len <= len(y):
            yield y[start : start + chunk_len]
            start += step_size
            
    def _chunk_to_tensor_and_spec(self, audio_chunk: np.ndarray, sr: int) -> Tuple[torch.Tensor, np.ndarray]:
        """Converts an audio chunk to the model's required tensor and the mel spec data."""
        y_preemphasized = librosa.effects.preemphasis(audio_chunk)
        mel_spec = librosa.feature.melspectrogram(
            y=y_preemphasized, sr=sr, n_fft=self.config.n_fft,
            hop_length=self.config.hop_length, n_mels=self.config.n_mels
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Create plot in memory to generate image
        fig = plt.figure(figsize=(4, 4), dpi=self.config.dpi)
        ax = plt.Axes(fig, [0., 0., 1., 1.]); ax.set_axis_off(); fig.add_axes(ax)
        librosa.display.specshow(mel_spec_db, sr=sr, ax=ax)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig); buf.seek(0)
        
        image = Image.open(buf)
        tensor = self.transform(image)
        buf.close()
        return tensor, mel_spec_db

    def _generate_analysis_metrics(self, y: np.ndarray, sr: int, duration_s: float) -> Tuple[AudioProperties, PitchAnalysis, EnergyAnalysis, SpectralAnalysis]:
        """Generates all unified audio metrics."""
        
        properties = AudioProperties(duration_seconds=duration_s, sample_rate=sr, channels=1)
        
        # Pitch Analysis
        pitch_values, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        valid_pitch = pitch_values[~np.isnan(pitch_values)]
        mean_pitch = float(np.mean(valid_pitch)) if len(valid_pitch) > 0 else None
        pitch_std_dev = float(np.std(valid_pitch)) if len(valid_pitch) > 1 else 0.0
        pitch_stability = max(0.0, 1.0 - (pitch_std_dev / 100.0)) if mean_pitch else None
        pitch_analysis = PitchAnalysis(mean_pitch_hz=mean_pitch, pitch_stability_score=pitch_stability)

        # Energy Analysis
        rms_energy = float(np.mean(librosa.feature.rms(y=y)))
        silent_intervals = librosa.effects.split(y, top_db=40)
        total_audio_duration = sum((end - start) / sr for start, end in silent_intervals)
        silence_ratio = 1.0 - (total_audio_duration / duration_s) 
        energy_analysis = EnergyAnalysis(rms_energy=rms_energy, silence_ratio=silence_ratio)

        # Spectral Analysis
        spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        spectral_contrast = float(np.mean(librosa.feature.spectral_contrast(y=y, sr=sr)))
        spectral_analysis = SpectralAnalysis(spectral_centroid=spectral_centroid, spectral_contrast=spectral_contrast)
        
        return properties, pitch_analysis, energy_analysis, spectral_analysis


# --- Concrete Detector Implementations ---

class MelSpectrogramCNNV2(BaseMelSpectrogramDetector): # Handles V2 from config
    """Non-overlapping chunk analysis (Standard V1/V2 implementation)."""
    config: MelSpectrogramCNNConfig
    
    def analyze(self, media_path: str, **kwargs) -> AnalysisResult:
        start_time = time.time()
        y, sr = self._get_audio_chunks(media_path)
        duration_s = librosa.get_duration(y=y, sr=sr)
        
        chunk_predictions: List[float] = []
        last_mel_spec_db = None
        
        if duration_s < self.config.chunk_duration_s:
            raise MediaProcessingError(f"Audio duration ({duration_s:.2f}s) is less than the required chunk size of {self.config.chunk_duration_s}s.")

        try:
            window_generator = self._get_audio_window_generator(y, sr)
            
            with torch.no_grad():
                for chunk in tqdm(window_generator, desc=f"Analyzing {self.config.class_name} chunks"):
                    tensor, mel_spec_db = self._chunk_to_tensor_and_spec(chunk, sr)
                    last_mel_spec_db = mel_spec_db # Keep the last one for visualization
                    
                    output = self.model(tensor.unsqueeze(0).to(self.device))
                    prob_real = torch.sigmoid(output).item()
                    chunk_predictions.append(prob_real)
        except Exception as e:
            raise InferenceError(f"Error during model inference for chunks: {e}")
        
        if not chunk_predictions:
            raise InferenceError("No valid chunks could be processed for analysis.")

        # Final Aggregation (using probability of REAL)
        avg_prob_real = np.mean(chunk_predictions)
        prob_fake = 1.0 - avg_prob_real
        prediction = "REAL" if avg_prob_real >= 0.5 else "FAKE"
        confidence = avg_prob_real if prediction == "REAL" else prob_fake
        
        # Metrics & Visualization
        properties, pitch, energy, spectral = self._generate_analysis_metrics(y, sr, duration_s)
        
        # Save the final Mel spec as a file (standard visualization)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png", prefix="spec_") as tmp:
            fig = plt.figure(figsize=(4, 4), dpi=100)
            ax = plt.Axes(fig, [0., 0., 1., 1.]); ax.set_axis_off(); fig.add_axes(ax)
            if last_mel_spec_db is not None:
                librosa.display.specshow(last_mel_spec_db, sr=sr, ax=ax)
            plt.savefig(tmp.name, format='png', bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            spectrogram_path = tmp.name

        return AudioAnalysisResult(
            prediction=prediction,
            confidence=float(confidence),
            processing_time=time.time() - start_time,
            properties=properties,
            pitch=pitch, energy=energy, spectral=spectral,
            # This returns List[List[float]] - the 2D spectrogram matrix
            visualization=AudioVisualization(spectrogram_url=spectrogram_path, spectrogram_data=last_mel_spec_db.tolist() if last_mel_spec_db is not None else None)
        )


class MelSpectrogramCNNV3(BaseMelSpectrogramDetector): # Handles V3 from config
    """Overlapping chunk analysis (New implementation with temporal plot)."""
    config: MelSpectrogramCNNv3Config

    def _generate_temporal_visualization(self, chunk_predictions_fake_scores: List[float], sr: int, final_verdict: str) -> str:
        """Generates a temporal plot of the chunk predictions and saves it as a file."""
        chunk_step_s = self.config.chunk_duration_s - self.config.chunk_overlap_s
        time_points = np.arange(len(chunk_predictions_fake_scores)) * chunk_step_s
        
        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
        ax.plot(time_points, chunk_predictions_fake_scores, color='#FF851B', linewidth=2)
        ax.axhline(0.5, color='white', linestyle='--', alpha=0.7)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"Chunk Prediction Over Time ({self.config.class_name})", color='white')
        ax.set_xlabel("Time (seconds)", color='white')
        ax.set_ylabel("Fake Score (0.0=Real, 1.0=Fake)", color='white')
        ax.tick_params(colors='white')
        
        # Highlight final verdict
        ax.text(0.98, 0.02, final_verdict, transform=ax.transAxes, fontsize=12, verticalalignment='bottom', horizontalalignment='right', color='white',
                bbox={'facecolor': '#2C3E50', 'alpha': 0.8, 'pad': 5})
        
        fig.tight_layout(pad=0.5)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png", prefix="analysis_") as tmp:
            plt.savefig(tmp.name, format='png', facecolor=fig.get_facecolor())
            visualization_path = tmp.name
        plt.close(fig)
        return visualization_path
            
    def analyze(self, media_path: str, **kwargs) -> AnalysisResult:
        """The core logic for V3 using overlapping chunks and specialized visualization."""
        start_time = time.time()
        y, sr = self._get_audio_chunks(media_path)
        duration_s = librosa.get_duration(y=y, sr=sr)
        
        if duration_s < self.config.chunk_duration_s:
            raise MediaProcessingError(f"Audio duration ({duration_s:.2f}s) is less than the required chunk size of {self.config.chunk_duration_s}s.")

        chunk_predictions_fake_scores: List[float] = []

        try:
            # Use overlapping generator for V3
            window_generator = self._get_audio_window_generator(y, sr)
            
            with torch.no_grad():
                for chunk in tqdm(window_generator, desc=f"Analyzing {self.config.class_name} chunks"):
                    tensor, _ = self._chunk_to_tensor_and_spec(chunk, sr)
                    
                    output = self.model(tensor.unsqueeze(0).to(self.device))
                    prob_real = torch.sigmoid(output).item()
                    prob_fake = 1.0 - prob_real
                    chunk_predictions_fake_scores.append(prob_fake)

        except Exception as e:
            raise InferenceError(f"Error during model inference for chunks: {e}")

        if not chunk_predictions_fake_scores:
            raise InferenceError("No valid chunks could be processed for analysis.")

        # Final Aggregation
        avg_prob_fake = np.mean(chunk_predictions_fake_scores)
        prediction = "FAKE" if avg_prob_fake >= 0.5 else "REAL"
        confidence = avg_prob_fake if prediction == "FAKE" else 1 - avg_prob_fake
        final_verdict_str = f"{prediction.upper()} (Confidence: {confidence:.2%})"
        
        # Metrics & Visualization
        properties, pitch, energy, spectral = self._generate_analysis_metrics(y, sr, duration_s)
        
        # 1. Temporal Analysis Plot (the core visualization for this model)
        temporal_plot_path = self._generate_temporal_visualization(chunk_predictions_fake_scores, sr, final_verdict_str)
        
        # 2. Assemble and return the result
        return AudioAnalysisResult(
            prediction=prediction,
            confidence=float(confidence),
            processing_time=time.time() - start_time,
            properties=properties,
            pitch=pitch, energy=energy, spectral=spectral,
            visualization=AudioVisualization(
                # Use the temporal plot as the main visualization for this feature
                spectrogram_url=temporal_plot_path, 
                # This now passes validation as List[float] due to the schema fix
                spectrogram_data=chunk_predictions_fake_scores 
            ),
            metrics={
                "average_chunk_fake_score": avg_prob_fake,
                "chunk_predictions": chunk_predictions_fake_scores,
                "chunk_duration_s": self.config.chunk_duration_s,
                "chunk_overlap_s": self.config.chunk_overlap_s,
                "num_chunks": len(chunk_predictions_fake_scores),
            }
        )
