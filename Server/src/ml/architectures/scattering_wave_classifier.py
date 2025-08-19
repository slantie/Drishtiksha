# src/ml/architectures/scattering_wave_classifier.py

import torch
import torch.nn as nn
from kymatio.torch import Scattering2D
from typing import Tuple

class AudioClassifier(nn.Module):
    """
    Audio classifier using a 2D Wavelet Scattering Transform for feature extraction
    followed by a series of dense layers for classification.
    """
    def __init__(self, input_shape: Tuple[int, int] = (256, 256)):
        super(AudioClassifier, self).__init__()
        self.input_shape = input_shape
        # Scattering2D acts as a powerful, fixed feature extractor
        self.scattering = Scattering2D(J=4, L=8, shape=self.input_shape)
        
        # Dynamically calculate the output dimension of the scattering transform
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, self.input_shape[0], self.input_shape[1])
            scattering_output = self.scattering(dummy_input)
            scattering_output_dim = scattering_output.view(1, -1).size(1)
            
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(scattering_output_dim, 1024), nn.ReLU(), nn.BatchNorm1d(1024), nn.Dropout(0.4),
            nn.Linear(1024, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Dropout(0.4),
            nn.Linear(512, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x is expected to be the spectrogram image tensor
        x = self.scattering(x)
        x = self.classifier(x)
        return x

def create_scattering_wave_model(config: dict) -> AudioClassifier:
    """Factory function to create an instance of the AudioClassifier."""
    model = AudioClassifier(input_shape=tuple(config['image_size']))
    return model