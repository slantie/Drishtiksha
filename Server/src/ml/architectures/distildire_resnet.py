# src/ml/architectures/distildire_resnet.py

import torch
import torch.nn as nn
import torchvision.models as TVM
from collections import OrderedDict
from typing import Dict, Any

class DistilDIRE(torch.nn.Module):
    """
    The student detector model for Distil-DIRE.
    This is a ResNet-50 model with a modified first convolutional layer
    to accept a 6-channel input (3 for RGB image, 3 for DIRE noise map).
    """
    def __init__(self):
        super(DistilDIRE, self).__init__()
    
        # Define the model using a standard ResNet-50 backbone
        student = TVM.resnet50()
        self.student_backbone = nn.Sequential(
            OrderedDict([*(list(student.named_children())[:-2])])
        )
        
        # Modify the first layer to accept 6 channels instead of 3
        self.student_backbone.conv1 = nn.Conv2d(
            6, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Define the classifier head
        self.student_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(2048, 1)
        )

    def forward(self, img_and_eps: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for the detector.

        Args:
            img_and_eps (torch.Tensor): A 6-channel tensor containing the
                                        concatenated image and its epsilon map.
        
        Returns:
            A dictionary containing the output logit and the feature map.
        """
        feature = self.student_backbone(img_and_eps) 
        logit = self.student_head(feature)
        return {'logit': logit, 'feature': feature}

def create_distildire_model(config: Dict[str, Any]) -> DistilDIRE:
    """Factory function to create an instance of the DistilDIRE model."""
    model = DistilDIRE()
    return model