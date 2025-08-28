# src/ml/architectures/siglip_lstm.py

import torch
import torch.nn as nn
from transformers import AutoModel
from typing import Dict, Any

# REFACTOR: This single class now handles the architecture for V1, V3, and V4 models.
class SiglipLSTMClassifier(nn.Module):
    """
    The unified PyTorch architecture for all SigLIP+LSTM models.
    
    This class dynamically constructs the appropriate classifier head based on the
    provided configuration, supporting both legacy models (V1, V3) and the newer
    V4 model with dropout.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # Load the pre-trained SigLIP vision backbone, shared by all versions.
        self.siglip_backbone = AutoModel.from_pretrained(config['base_model_path']).vision_model
            
        backbone_hidden_size = self.siglip_backbone.config.hidden_size
        lstm_hidden_size = config['lstm_hidden_size']
        
        # V4 and later models include dropout in their config for regularization.
        dropout_rate = config.get('dropout_rate', 0.0)

        self.lstm = nn.LSTM(
            input_size=backbone_hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=config['lstm_num_layers'],
            batch_first=True,
            bidirectional=True,
            # Apply dropout between LSTM layers if configured and num_layers > 1
            dropout=dropout_rate if config['lstm_num_layers'] > 1 else 0
        )
        
        num_classes = config.get('num_classes', 1)
        
        # --- Conditional Classifier Head Construction ---
        # If dropout_rate is specified in the config, build the more complex V4 head.
        if dropout_rate > 0:
            # V4 Classifier Head with Dropout and an extra layer
            self.video_classifier = nn.Sequential(
                nn.Linear(lstm_hidden_size * 2, lstm_hidden_size), # *2 for bidirectional
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(lstm_hidden_size, num_classes)
            )
            # A simple image classifier for single-frame inputs
            self.image_classifier = nn.Sequential(
                nn.Linear(backbone_hidden_size, 512),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(512, num_classes)
            )
        else:
            # Legacy Classifier Head (V1, V3) - a single linear layer
            self.video_classifier = nn.Linear(lstm_hidden_size * 2, num_classes)
            self.image_classifier = nn.Linear(backbone_hidden_size, num_classes)

    def forward(self, pixel_values: torch.Tensor, num_frames_per_video: int = 1) -> torch.Tensor:
        # Extract features from the vision backbone
        features = self.siglip_backbone(pixel_values=pixel_values).pooler_output
        
        if num_frames_per_video > 1:
            # Video input: reshape for LSTM
            batch_size = pixel_values.shape[0] // num_frames_per_video
            features = features.view(batch_size, num_frames_per_video, -1)
            
            # Pass through LSTM
            _, (hidden, _) = self.lstm(features)
            
            # Concatenate the final forward and backward hidden states
            final_hidden_state = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
            
            logits = self.video_classifier(final_hidden_state)
        else:
            # Single image input
            logits = self.image_classifier(features)
            
        return logits

def create_siglip_lstm_model(config: Dict[str, Any]) -> SiglipLSTMClassifier:
    """Factory function to instantiate the unified SiglipLSTMClassifier model."""
    model = SiglipLSTMClassifier(config)
    return model