# src/ml/architectures/color_cues_lstm.py

import torch
import torch.nn as nn
from typing import Dict, Any

class LSTMClassifier(nn.Module):
    """
    The PyTorch architecture for the ColorCues LSTM model.
    This model is designed to process sequences of color histogram features
    extracted from video frames.
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, dropout: float = 0.5):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)
        # Note: Sigmoid is not used here because BCEWithLogitsLoss is preferred for training stability.
        # The sigmoid function is applied to the output logits during inference.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the forward pass for the classifier."""
        # The input 'x' is expected to be a tensor of histograms.
        # It's flattened before being passed to the LSTM.
        batch_size, seq_len, _, _ = x.shape
        x_flattened = x.view(batch_size, seq_len, -1)
        
        # Get the last hidden state from the LSTM
        h_lstm, _ = self.lstm(x_flattened)
        out = self.dropout(h_lstm[:, -1, :])
        
        # Pass through fully connected layers
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        logits = self.fc2(out)
        return logits

def create_color_cues_model(config: Dict[str, Any]) -> LSTMClassifier:
    """Factory function to instantiate the ColorCues LSTMClassifier model."""
    input_size = config['histogram_bins'] * config['histogram_bins']
    model = LSTMClassifier(
        input_size=input_size,
        hidden_size=config['hidden_size'],
        dropout=config['dropout']
    )
    return model