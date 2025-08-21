import os
import io
import time
import torch
import logging
import librosa
import numpy as np
import tempfile
from pydub import AudioSegment
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from fastapi import HTTPException, status
from typing import Dict, Any
from src.ml.base import BaseModel
from src.config import ScatteringWaveV1Config
from src.ml.architectures.scattering_wave_classifier import create_scattering_wave_model
from src.ml.event_publisher import publish_progress
from src.app.schemas import AudioAnalysisData, AudioProperties, PitchAnalysis, EnergyAnalysis, SpectralAnalysis, AudioVisualization

logger = logging.getLogger(__name__)

class ScatteringWaveV1(BaseModel):
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

    # --- REFACTORED: This function will now return all intermediate products for analysis ---
    def _extract_and_process_audio(self, video_path: str, video_id: str = None, user_id: str = None):
        if video_id: publish_progress({"videoId": video_id, "userId": user_id, "event": "AUDIO_EXTRACTION_START"})
        try:
            audio_segment = AudioSegment.from_file(video_path)
            num_channels = audio_segment.channels
            audio_segment = audio_segment.set_channels(1).set_frame_rate(self.config.sampling_rate)
            wav_buffer = io.BytesIO()
            audio_segment.export(wav_buffer, format="wav")
            wav_buffer.seek(0)
        except Exception as e:
            raise IOError(f"Pydub failed. The file may be corrupt or have no audio track. Error: {e}")
        if video_id: publish_progress({"videoId": video_id, "userId": user_id, "event": "AUDIO_EXTRACTION_COMPLETE"})
        
        try:
            y, sr = librosa.load(wav_buffer, sr=self.config.sampling_rate)
            target_length = int(sr * self.config.duration_seconds)
            if len(y) < target_length:
                y = np.pad(y, (0, target_length - len(y)), 'constant')
            else:
                y = y[:target_length]
            y_preemphasized = librosa.effects.preemphasis(y)
        except Exception as e:
            raise ValueError(f"Librosa failed to process the audio waveform. Error: {e}")

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
        
        image = Image.open(img_buffer)
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # --- CHANGE: Return the raw mel_spec_db numpy array as well ---
        return tensor, y, sr, num_channels, img_buffer, mel_spec_db

    # --- REFACTORED: This is now the main orchestration method ---
    def predict_detailed(self, video_path: str, **kwargs) -> Dict[str, Any]:
        start_time = time.time()
        video_id = kwargs.get("video_id")
        user_id = kwargs.get("user_id")

        try:
            tensor, y, sr, channels, spec_img_buffer, mel_spec_db = self._extract_and_process_audio(video_path, video_id, user_id)
        except (IOError, ValueError) as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Audio analysis failed. Could not process audio from the provided video. Reason: {e}"
            )
        
        # 1. Model Inference
        with torch.no_grad():
            output = self.model(tensor)
            prob_real = torch.sigmoid(output).item()
        
        prob_fake = 1.0 - prob_real
        prediction = "REAL" if prob_real >= 0.5 else "FAKE"
        confidence = prob_real if prediction == "REAL" else prob_fake

        pitch_values, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        valid_pitch_values = pitch_values[~np.isnan(pitch_values)]
        mean_pitch = float(np.mean(valid_pitch_values)) if len(valid_pitch_values) > 0 else None
        pitch_std_dev = float(np.std(valid_pitch_values)) if len(valid_pitch_values) > 1 else 0.0
        pitch_stability = max(0.0, 1.0 - (pitch_std_dev / 100.0)) if mean_pitch else None
        
        rms_energy = float(np.mean(librosa.feature.rms(y=y)))
        silent_intervals = librosa.effects.split(y, top_db=40)
        total_silent_duration = sum((end - start) / sr for start, end in silent_intervals)
        silence_ratio = 1.0 - (total_silent_duration / self.config.duration_seconds)

        spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        spectral_contrast = float(np.mean(librosa.feature.spectral_contrast(y=y, sr=sr)))

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png", prefix="spec_") as tmp:
            tmp.write(spec_img_buffer.getvalue())
            spec_filename = os.path.basename(tmp.name)
        spec_url = f"/analyze/visualization/{spec_filename}"
        spec_img_buffer.close()

        # --- CHANGE: Assemble the response object, now including the spectrogram data ---
        response_data = AudioAnalysisData(
            prediction=prediction,
            confidence=float(confidence),
            processing_time=time.time() - start_time,
            properties=AudioProperties(duration_seconds=self.config.duration_seconds, sample_rate=sr, channels=channels),
            pitch=PitchAnalysis(mean_pitch_hz=mean_pitch, pitch_stability_score=pitch_stability),
            energy=EnergyAnalysis(rms_energy=rms_energy, silence_ratio=silence_ratio),
            spectral=SpectralAnalysis(spectral_centroid=spectral_centroid, spectral_contrast=spectral_contrast),
            visualization=AudioVisualization(
                spectrogram_url=spec_url,
                spectrogram_data=mel_spec_db.tolist() 
            )
        )
        
        return response_data.model_dump()

    def predict(self, video_path: str, **kwargs) -> Dict[str, Any]:
        result = self.predict_detailed(video_path, **kwargs)
        return {
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "processing_time": result["processing_time"],
        }