# src/ml/architectures/siglip_lstm.py

import torch
import torch.nn as nn
from transformers import AutoModel
from typing import Dict, Any

class SiglipLSTMClassifier(nn.Module):
    """
    The PyTorch architecture for the SigLIP+LSTM model.
    This class defines the model structure, which is necessary for both
    training and for loading pre-trained weights during inference.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # Load the pre-trained SigLIP vision model as the backbone
        self.siglip_backbone = AutoModel.from_pretrained(config['base_model_path']).vision_model
            
        backbone_hidden_size = self.siglip_backbone.config.hidden_size
        lstm_hidden_size = config['lstm_hidden_size']
        # Get dropout rate from config, default to 0.5 if not present for older models
        dropout_rate = config.get('dropout_rate', 0.5)

        # LSTM layer for processing temporal sequences of features
        self.lstm = nn.LSTM(
            input_size=backbone_hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=config['lstm_num_layers'],
            batch_first=True,
            bidirectional=True,
            # Add dropout to LSTM itself if more than one layer
            dropout=dropout_rate if config['lstm_num_layers'] > 1 else 0
        )
        
        num_classes = config.get('num_classes', 1)
        
        # A classifier head for video-level predictions (sequences of frames)
        self.video_classifier = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, lstm_hidden_size), # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(lstm_hidden_size, num_classes)
        )
        
        # A separate classifier head for single-image predictions
        self.image_classifier = nn.Sequential(
            nn.Linear(backbone_hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

    def forward(self, pixel_values: torch.Tensor, num_frames_per_video: int = 1) -> torch.Tensor:
        """
        Defines the forward pass of the model.
        It can operate in two modes: video (sequence) or image (single frame).
        """
        # Extract features from the SigLIP backbone
        features = self.siglip_backbone(pixel_values=pixel_values).pooler_output
        
        if num_frames_per_video > 1:
            # Video mode: reshape features and pass through LSTM
            batch_size = pixel_values.shape[0] // num_frames_per_video
            features = features.view(batch_size, num_frames_per_video, -1)
            lstm_out, (hidden, cell) = self.lstm(features)
            # Concatenate the final forward and backward hidden states
            final_hidden_state = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
            logits = self.video_classifier(final_hidden_state)
        else:
            # Image mode: pass features directly to the image classifier
            logits = self.image_classifier(features)
            
        return logits

def create_lstm_model(config: Dict[str, Any]) -> SiglipLSTMClassifier:
    """Factory function to instantiate the SiglipLSTMClassifier model."""
    model = SiglipLSTMClassifier(config)
    return model