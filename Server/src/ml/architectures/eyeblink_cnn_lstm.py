# src/ml/architectures/eyeblink_cnn_lstm.py

import timm
import torch
import torch.nn as nn
from typing import Dict, Any

class EyeblinkCnnLstm(nn.Module):
    """
    PyTorch implementation of the Xception + LSTM model for eye blink-based
    deepfake detection. This is a direct port of the Keras model architecture.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # 1. Load pre-trained Xception model as the CNN base.
        # This is equivalent to Keras: `Xception(weights="imagenet", include_top=False, ...)`
        # We use timm to easily get the feature extractor part of the model.
        # Temporarily suppress timm logging to avoid pretrained weight download messages.
        import logging
        timm_logger = logging.getLogger('timm')
        original_level = timm_logger.getEffectiveLevel()
        timm_logger.setLevel(logging.WARNING)
        
        try:
            self.cnn_base = timm.create_model(
                config['base_model_name'], 
                pretrained=config.get('pretrained', True), 
                features_only=True
            )
        finally:
            # Restore original logging level
            timm_logger.setLevel(original_level)
        
        # In PyTorch, we can set requires_grad to False to freeze layers.
        for param in self.cnn_base.parameters():
            param.requires_grad = False

        # Determine the number of output features from the CNN base.
        # For Xception, this is 2048.
        num_features = self.cnn_base.feature_info.channels(-1)

        # 2. LSTM layer to process sequences of features.
        # Equivalent to Keras: `LSTM(128)`
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=config['lstm_hidden_size'],
            num_layers=1,  # As per the original Keras model
            batch_first=True,
            bidirectional=False # As per the original Keras model
        )

        # 3. Dropout and final classification layers.
        # Equivalent to Keras: `Dropout(0.3)` and `Dense(1, activation="sigmoid")`
        self.dropout = nn.Dropout(config.get('dropout_rate', 0.3))
        self.fc = nn.Linear(config['lstm_hidden_size'], 1)
        # Note: The sigmoid activation is applied to the output logits during inference,
        # not in the model's forward pass, to work with BCEWithLogitsLoss during training.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch_size, sequence_length, channels, height, width)
        batch_size, seq_len, c, h, w = x.shape
        
        # Reshape to process all frames through the CNN at once.
        # This is how TimeDistributed works under the hood.
        x = x.view(batch_size * seq_len, c, h, w)
        
        # Get features from the CNN base. We take the last feature map.
        features = self.cnn_base(x)[-1]
        
        # Global Average Pooling.
        # Equivalent to Keras: `TimeDistributed(GlobalAveragePooling2D())`
        # Shape: (batch_size * seq_len, num_features, feature_h, feature_w) -> (batch_size * seq_len, num_features)
        pooled_features = features.mean(dim=[-1, -2])
        
        # Reshape back to a sequence for the LSTM.
        # Shape: (batch_size, seq_len, num_features)
        seq_features = pooled_features.view(batch_size, seq_len, -1)
        
        # Pass through LSTM. We only need the output of the last time step.
        lstm_out, (h_n, c_n) = self.lstm(seq_features)
        
        # Get the last hidden state.
        last_hidden_state = h_n[-1]
        
        # Apply dropout and the final classification layer.
        x = self.dropout(last_hidden_state)
        logits = self.fc(x)
        
        return logits

def create_eyeblink_model(config: Dict[str, Any]) -> EyeblinkCnnLstm:
    """Factory function to create an instance of the EyeblinkCnnLstm model."""
    model = EyeblinkCnnLstm(config)
    return model