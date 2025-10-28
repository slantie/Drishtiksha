# Server/src/ml/architectures/audio_mft_crnn.py

import torch
import torch.nn as nn
from typing import Dict, Any

class AudioMFTCRNN(nn.Module):
    """
    A Convolutional Recurrent Neural Network (CRNN) for audio deepfake detection.
    It processes a stacked tensor of multiple audio features (Mel, STFT, MFCC, etc.).
    """
    def __init__(self, config: Dict[str, Any]):
        super(AudioMFTCRNN, self).__init__()
        self.config = config

        # --- 1. CNN Feature Extractor ---
        self.cnn_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=(2, 1))
        )

        # --- 2. Dynamic Calculation of LSTM input size ---
        lstm_input_size = self._get_lstm_input_size()

        # --- 3. LSTM for Temporal Modeling ---
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )

        # --- 4. Classifier ---
        self.classifier = nn.Sequential(
            nn.Linear(128 * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 1)
        )

    def _get_lstm_input_size(self) -> int:
        """
        Calculates the LSTM's input size dynamically based on the features
        enabled in the configuration.
        """
        method_config = self.config['combined_vector_model']
        feature_conf = self.config['feature_extraction']

        # Start with the mandatory features
        num_initial_features = method_config['n_mels']
        num_initial_features += (method_config['stft_n_fft_wide'] // 2) + 1
        num_initial_features += (method_config['stft_n_fft_narrow'] // 2) + 1
        
        # Add dimensions for optional features
        if feature_conf.get('mfcc', {}).get('enabled'):
            n_mfcc = feature_conf['mfcc']['n_mfcc']
            num_initial_features += n_mfcc * 3 if feature_conf['mfcc'].get('include_deltas') else n_mfcc
        if feature_conf.get('chroma_features', {}).get('enabled'):
            num_initial_features += feature_conf['chroma_features']['n_chroma']
        if feature_conf.get('spectral_contrast', {}).get('enabled'):
            num_initial_features += feature_conf['spectral_contrast']['n_bands'] + 1
        if feature_conf.get('zero_crossing_rate', {}).get('enabled'):
            num_initial_features += 2

        # Perform a dummy forward pass through the CNN to get the output shape
        dummy_input = torch.randn(1, 1, int(num_initial_features), 32) # Assuming 32 time steps
        cnn_output = self.cnn_extractor(dummy_input)
        
        batch_size, num_channels, height, width = cnn_output.shape
        reshaped_output = cnn_output.permute(0, 3, 1, 2).reshape(batch_size, width, -1)
        
        return reshaped_output.shape[2]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn_extractor(x)
        batch_size, num_channels, height, width = x.shape
        x = x.permute(0, 3, 1, 2).reshape(batch_size, width, -1)
        _, (h_n, _) = self.lstm(x)
        x = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        x = self.classifier(x)
        return x

def create_audio_mft_crnn_model(config: Dict[str, Any]) -> AudioMFTCRNN:
    """Factory function to create an instance of the AudioMFTCRNN model."""
    model = AudioMFTCRNN(config)
    return model