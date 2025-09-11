# Server/src/ml/architectures/stft_spectrogram_cnn.py

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

class STFTSpectrogramCNN(nn.Module):
    """A CNN for classifying stacked wideband/narrowband spectrogram images."""
    def __init__(self, input_shape: Tuple[int, int, int], num_classes: int):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(16), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(32), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(64), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(128), nn.MaxPool2d(2),
        )
        conv_output_size = self._get_conv_output_size(input_shape)
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_output_size, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def _get_conv_output_size(self, shape: Tuple[int, int, int]) -> int:
        with torch.no_grad():
            dummy_input = torch.randn(1, *shape)
            output = self.conv_layers(dummy_input)
        return int(np.prod(output.size()[1:]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

def create_stft_spectrogram_model(config: dict) -> STFTSpectrogramCNN:
    """Factory function to create an instance of the STFTSpectrogramCNN model."""
    input_shape = (3, config['img_height'], config['img_width'])
    num_classes = 2 # real, fake
    model = STFTSpectrogramCNN(input_shape, num_classes)
    return model