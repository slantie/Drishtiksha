# Server/src/ml/architectures/mel_spectrogram_cnn.py

import torch
import torch.nn as nn
from kymatio.torch import Scattering2D
from typing import Tuple

class MelSpectrogramCNN(nn.Module):
    def __init__(self, input_shape: Tuple[int, int] = (256, 256)):
        super().__init__()
        self.input_shape = input_shape
        # Kymatio Scattering2D acts as a fixed feature extractor
        self.scattering = Scattering2D(J=4, L=8, shape=self.input_shape)

        # Dynamically calculate the flattened size after scattering
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, self.input_shape[0], self.input_shape[1])
            scattering_output = self.scattering(dummy_input)
            scattering_output_dim = scattering_output.view(1, -1).size(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(scattering_output_dim, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 1) # Binary output for BCEWithLogitsLoss
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.scattering(x)
        x = self.classifier(x)
        return x

def create_mel_spectrogram_model(config: dict) -> MelSpectrogramCNN:
    """Factory function to create an instance of the MelSpectrogramCNN model."""
    # The model architecture is fixed, so config is just for potential future use.
    model = MelSpectrogramCNN(input_shape=(256, 256))
    return model