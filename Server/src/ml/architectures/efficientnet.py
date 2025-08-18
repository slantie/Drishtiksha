# src/ml/architectures/efficientnet.py

from functools import partial
import timm
from torch import nn
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.pooling import AdaptiveAvgPool2d

# Define a mapping for available encoders to simplify model creation
encoder_params = {
    "tf_efficientnet_b7_ns": {
        "features": 2560,
        "init_op": partial(timm.create_model, "tf_efficientnet_b7_ns", pretrained=True)
    },
}

class DeepFakeClassifier(nn.Module):
    """
    Classifier based on the EfficientNet architecture.
    This model takes image features from the encoder and passes them through
    a linear layer to produce a final prediction.
    """
    def __init__(self, encoder, dropout_rate=0.0):
        super().__init__()
        if encoder not in encoder_params:
            raise ValueError(f"Encoder '{encoder}' is not supported.")
            
        self.encoder = encoder_params[encoder]["init_op"]()
        self.avg_pool = AdaptiveAvgPool2d((1, 1))
        self.dropout = Dropout(dropout_rate)
        self.fc = Linear(encoder_params[encoder]["features"], 1)

    def forward(self, x):
        # Forward pass through the EfficientNet encoder
        x = self.encoder.forward_features(x)
        # Global average pooling and flattening
        x = self.avg_pool(x).flatten(1)
        # Dropout for regularization
        x = self.dropout(x)
        # Final classification layer
        x = self.fc(x)
        return x

def create_efficientnet_model(encoder: str) -> DeepFakeClassifier:
    """Factory function to create an instance of the DeepFakeClassifier."""
    model = DeepFakeClassifier(encoder=encoder)
    return model