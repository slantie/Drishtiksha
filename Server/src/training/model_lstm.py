# src/training/model_lstm.py

import torch
import torch.nn as nn
from transformers import AutoModel

class SiglipLSTMClassifier(nn.Module):
    """A classifier using a SigLIP backbone and an LSTM for temporal features."""
    def __init__(self, config):
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
        
        num_classes = config.get('num_classes', 1) # Default to 1 for BCEWithLogitsLoss
        self.video_classifier = nn.Linear(config['lstm_hidden_size'] * 2, num_classes)
        self.image_classifier = nn.Linear(backbone_hidden_size, num_classes)

    def forward(self, pixel_values, num_frames_per_video=1):
        features = self.siglip_backbone(pixel_values=pixel_values).pooler_output
        
        if num_frames_per_video > 1:
            batch_size = pixel_values.shape[0] // num_frames_per_video
            features = features.view(batch_size, num_frames_per_video, -1)
            lstm_out, (hidden, cell) = self.lstm(features)
            final_hidden_state = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
            logits = self.video_classifier(final_hidden_state)
        else:
            logits = self.image_classifier(features)
            
        return logits

def create_lstm_model(config):
    """Factory function to create the LSTM model."""
    model = SiglipLSTMClassifier(config)
    return model