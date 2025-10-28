# Server/src/ml/detectors/audio_mft_crnn_detector.py

import os
import io
import time
import torch
import librosa
import logging
import numpy as np
from pydub import AudioSegment
from typing import Dict, List, Tuple

from src.ml.base import BaseModel, AnalysisResult
from src.app.schemas import AudioAnalysisResult, AudioProperties, PitchAnalysis, EnergyAnalysis, SpectralAnalysis
from src.config import AudioMFTCRNNConfig # We will create this in the next step
from src.ml.architectures.audio_mft_crnn import create_audio_mft_crnn_model
from src.ml.exceptions import MediaProcessingError, InferenceError

logger = logging.getLogger(__name__)

class AudioMFTCRNNV1(BaseModel):
    """
    Detector for audio deepfakes using a Multi-Feature Temporal (MFT)
    Convolutional Recurrent Neural Network (CRNN).
    """
    config: AudioMFTCRNNConfig

    def load(self) -> None:
        start_time = time.time()
        try:
            # Pass the full config, as the model needs it for dynamic sizing
            self.model = create_audio_mft_crnn_model(self.config.model_dump())
            state_dict = torch.load(self.config.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()

            load_time = time.time() - start_time
            logger.info(f"✅ Loaded Model: '{self.config.model_name}' | Device: '{self.device}' | Time: {load_time:.2f}s.")
        except Exception as e:
            logger.error(f"Failed to load model '{self.config.model_name}': {e}", exc_info=True)
            raise RuntimeError(f"Failed to load model '{self.config.model_name}'") from e

    def _robust_load_audio(self, audio_path: str, sr: int) -> np.ndarray:
        """Loads audio robustly using pydub and librosa."""
        try:
            audio = AudioSegment.from_file(audio_path).set_channels(1).set_frame_rate(sr)
            wav_buffer = io.BytesIO()
            audio.export(wav_buffer, format="wav")
            wav_buffer.seek(0)
            y, _ = librosa.load(wav_buffer, sr=sr, mono=True)
            return y
        except Exception as e:
            raise MediaProcessingError(f"Failed to load or convert audio file: {e}")

    def _extract_features(self, y_chunk: np.ndarray) -> Dict[str, np.ndarray]:
        """Extracts all configured audio features from a single chunk."""
        # This is a direct port of your original 'extract_features' function
        sr = self.config.preprocessing_params.sampling_rate
        method_config = self.config.combined_vector_model
        feature_config = self.config.feature_extraction
        hop_length = method_config.shared_hop_length
        features = {}

        y_preemphasized = librosa.effects.preemphasis(y_chunk)
        mel_spectrogram = librosa.feature.melspectrogram(y=y_preemphasized, sr=sr, n_fft=method_config.mel_n_fft, hop_length=hop_length, n_mels=method_config.n_mels)
        features['mel_spectrogram'] = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        stft_wide = librosa.stft(y_chunk, n_fft=method_config.stft_n_fft_wide, hop_length=hop_length, window="hann")
        features['stft_wide'] = librosa.amplitude_to_db(np.abs(stft_wide), ref=np.max)
        
        stft_narrow = librosa.stft(y_chunk, n_fft=method_config.stft_n_fft_narrow, hop_length=hop_length, window="hann")
        features['stft_narrow'] = librosa.amplitude_to_db(np.abs(stft_narrow), ref=np.max)

        n_fft_for_optional = method_config.stft_n_fft_narrow
        if feature_config.mfcc.enabled:
            mfccs = librosa.feature.mfcc(y=y_chunk, sr=sr, n_mfcc=feature_config.mfcc.n_mfcc, n_fft=n_fft_for_optional, hop_length=hop_length)
            if feature_config.mfcc.include_deltas:
                mfccs = np.concatenate((mfccs, librosa.feature.delta(mfccs), librosa.feature.delta(mfccs, order=2)))
            features['mfcc'] = mfccs
        
        if feature_config.chroma_features.enabled:
            features['chroma'] = librosa.feature.chroma_stft(y=y_chunk, sr=sr, n_chroma=feature_config.chroma_features.n_chroma, n_fft=n_fft_for_optional, hop_length=hop_length)
        
        if feature_config.spectral_contrast.enabled:
            # Need to handle potential short audio segments
            S = np.abs(librosa.stft(y=y_chunk, n_fft=n_fft_for_optional, hop_length=hop_length))
            if S.shape[1] > 0:
                features['spectral_contrast'] = librosa.feature.spectral_contrast(S=S, sr=sr, n_bands=feature_config.spectral_contrast.n_bands)
            else: # Create a zero-filled array if STFT is empty
                features['spectral_contrast'] = np.zeros((feature_config.spectral_contrast.n_bands + 1, S.shape[1]))

        if feature_config.zero_crossing_rate.enabled:
            zcr = librosa.feature.zero_crossing_rate(y=y_chunk, hop_length=hop_length)
            features['zcr_summary'] = np.array([np.mean(zcr), np.std(zcr)])

        return features

    def _combine_features_to_tensor(self, features: Dict[str, np.ndarray]) -> torch.Tensor:
        """Combines extracted features into a single tensor for the model."""
        active_features = [features['mel_spectrogram'], features['stft_wide'], features['stft_narrow']]
        time_steps = active_features[0].shape[1]
        
        feature_config = self.config.feature_extraction
        if feature_config.mfcc.enabled: active_features.append(features['mfcc'])
        if feature_config.chroma_features.enabled: active_features.append(features['chroma'])
        if feature_config.spectral_contrast.enabled: active_features.append(features['spectral_contrast'])
        if feature_config.zero_crossing_rate.enabled:
            active_features.append(np.tile(features['zcr_summary'].reshape(-1, 1), (1, time_steps)))
        
        # Ensure all feature arrays have the same number of time steps by padding/truncating
        aligned_features = []
        for feat in active_features:
            if feat.shape[1] > time_steps:
                aligned_features.append(feat[:, :time_steps])
            elif feat.shape[1] < time_steps:
                padding = np.zeros((feat.shape[0], time_steps - feat.shape[1]))
                aligned_features.append(np.hstack((feat, padding)))
            else:
                aligned_features.append(feat)

        combined_features = np.vstack(aligned_features)
        return torch.from_numpy(combined_features).float().unsqueeze(0).unsqueeze(0)

    def analyze(self, media_path: str, **kwargs) -> AnalysisResult:
        start_time = time.time()
        prep_config = self.config.preprocessing_params
        
        y = self._robust_load_audio(media_path, prep_config.sampling_rate)
        duration_s = librosa.get_duration(y=y, sr=prep_config.sampling_rate)
        
        chunk_len = int(prep_config.chunk_duration_s * prep_config.sampling_rate)
        step_size = chunk_len - int(prep_config.chunk_overlap_s * prep_config.sampling_rate)

        if len(y) < chunk_len:
            raise MediaProcessingError(f"Audio duration ({duration_s:.2f}s) is less than the required chunk size of {prep_config.chunk_duration_s}s.")

        chunk_predictions = []
        try:
            with torch.no_grad():
                start, chunk_idx = 0, 0
                while start + chunk_len <= len(y):
                    audio_chunk = y[start : start + chunk_len]
                    features = self._extract_features(audio_chunk)
                    input_tensor = self._combine_features_to_tensor(features).to(self.device)
                    
                    output = self.model(input_tensor)
                    prob = torch.sigmoid(output).item()
                    chunk_predictions.append(prob)
                    
                    start += step_size
                    chunk_idx += 1
        except Exception as e:
            raise InferenceError(f"Error during model inference on audio chunks: {e}")

        if not chunk_predictions:
            raise InferenceError("No valid audio chunks could be processed for analysis.")

        # Final aggregation (probability of FAKE)
        avg_prob_fake = np.mean(chunk_predictions)
        prediction = "FAKE" if avg_prob_fake >= 0.5 else "REAL"
        confidence = avg_prob_fake if prediction == "FAKE" else 1 - avg_prob_fake
        
        # Generate final metrics for the entire audio clip
        properties = AudioProperties(duration_seconds=duration_s, sample_rate=prep_config.sampling_rate, channels=1)
        pitch_values, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        valid_pitch = pitch_values[~np.isnan(pitch_values)]
        mean_pitch = float(np.mean(valid_pitch)) if len(valid_pitch) > 0 else None
        
        return AudioAnalysisResult(
            prediction=prediction,
            confidence=float(confidence),
            processing_time=time.time() - start_time,
            properties=properties,
            pitch=PitchAnalysis(mean_pitch_hz=mean_pitch, pitch_stability_score=None),
            energy=EnergyAnalysis(rms_energy=float(np.mean(librosa.feature.rms(y=y))), silence_ratio=0.0),
            spectral=SpectralAnalysis(spectral_centroid=float(np.mean(librosa.feature.spectral_centroid(y=y, sr=prep_config.sampling_rate))), spectral_contrast=0.0)
        )