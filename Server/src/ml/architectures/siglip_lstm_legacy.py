# src/ml/architectures/siglip_lstm_legacy.py

import torch
import torch.nn as nn
from transformers import AutoModel
from typing import Dict, Any

class SiglipLSTMLegacyClassifier(nn.Module):
    """
    The LEGACY PyTorch architecture for the SigLIP+LSTM models (V1 and V3).
    This class uses a simple nn.Linear layer for classification, matching the
    original saved model weights.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        self.siglip_backbone = AutoModel.from_pretrained(config['base_model_path']).vision_model
            
        backbone_hidden_size = self.siglip_backbone.config.hidden_size
        
        self.lstm = nn.LSTM(
            input_size=backbone_hidden_size,
            hidden_size=config['lstm_hidden_size'],
            num_layers=config['lstm_num_layers'],
            batch_first=True,
            bidirectional=True
        )
        
        num_classes = config.get('num_classes', 1)
        
        # Original simple classifier head
        self.video_classifier = nn.Linear(config['lstm_hidden_size'] * 2, num_classes) # *2 for bidirectional
        self.image_classifier = nn.Linear(backbone_hidden_size, num_classes)

    def forward(self, pixel_values: torch.Tensor, num_frames_per_video: int = 1) -> torch.Tensor:
        features = self.siglip_backbone(pixel_values=pixel_values).pooler_output
        
        if num_frames_per_video > 1:
            batch_size = pixel_values.shape[0] // num_frames_per_video
            features = features.view(batch_size, num_frames_per_video, -1)
            lstm_out, (hidden, _) = self.lstm(features)
            final_hidden_state = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
            logits = self.video_classifier(final_hidden_state)
        else:
            logits = self.image_classifier(features)
            
        return logits

def create_legacy_lstm_model(config: Dict[str, Any]) -> SiglipLSTMLegacyClassifier:
    """Factory function to instantiate the legacy SiglipLSTMClassifier model."""
    model = SiglipLSTMLegacyClassifier(config)
    return model