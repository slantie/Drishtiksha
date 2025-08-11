# /home/dell-pc-03/Deepfake/deepfake-detection/Raj/src/model_lstm.py

import torch
import torch.nn as nn
from transformers import AutoModel

class SiglipLSTMClassifier(nn.Module):
    """
    A dual-use classifier.
    - For video, it uses a SigLIP backbone to get frame features, then an LSTM to model time.
    - For a single image, it bypasses the LSTM and uses a simple linear head.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 1. Load the SigLIP vision backbone from your local files
        self.siglip_backbone = AutoModel.from_pretrained(config['base_model_path']).vision_model
        
        # You can freeze the backbone to train only the new layers faster
        # for param in self.siglip_backbone.parameters():
        #    param.requires_grad = False
            
        # 2. Define the temporal (video) head
        backbone_hidden_size = self.siglip_backbone.config.hidden_size
        self.lstm = nn.LSTM(
            input_size=backbone_hidden_size,
            hidden_size=config['lstm_hidden_size'],
            num_layers=config['lstm_num_layers'],
            batch_first=True,
            bidirectional=True
        )
        self.video_classifier = nn.Linear(config['lstm_hidden_size'] * 2, config['num_classes']) # *2 for bidirectional
        
        # 3. Define the simple image-only head
        self.image_classifier = nn.Linear(backbone_hidden_size, config['num_classes'])

    def forward(self, pixel_values, num_frames_per_video=1):
        # Get feature vectors for every frame from the SigLIP backbone
        # The output is a feature vector for each image (pooler_output)
        features = self.siglip_backbone(pixel_values=pixel_values).pooler_output
        
        # --- PATH 1: Video Classification (a sequence of frames) ---
        if num_frames_per_video > 1:
            # Reshape features from a flat list of frames into batches of sequences
            # (batch_size * num_frames, feature_size) -> (batch_size, num_frames, feature_size)
            batch_size = pixel_values.shape[0] // num_frames_per_video
            features = features.view(batch_size, num_frames_per_video, -1)
            
            # Pass the sequence of features through the LSTM
            lstm_out, (hidden, cell) = self.lstm(features)
            
            # We use the final hidden state to make the prediction for the whole sequence
            # Concatenate the final forward and backward hidden states
            final_hidden_state = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
            
            # Final classification for the video
            logits = self.video_classifier(final_hidden_state)
            
        # --- PATH 2: Single Image Classification ---
        else:
            # Bypass the LSTM and use the simple image classifier head
            logits = self.image_classifier(features)
            
        return logits

def create_lstm_model(config):
    """Factory function to create the LSTM model."""
    model = SiglipLSTMClassifier(config)
    return model