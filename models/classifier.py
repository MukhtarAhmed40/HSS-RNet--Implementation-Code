"""
Classification layer with mean pooling and softmax.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationLayer(nn.Module):
    """
    Mean pooling over time + fully connected layer + softmax.
    
    h_bar = (1/T) sum_t h_tilde_t
    y_hat = softmax(W_c h_bar + b_c)
    """
    
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, input_dim) gated representations
        Returns:
            logits: (B, num_classes)
            probs: (B, num_classes)
        """
        # Mean pooling
        h_bar = x.mean(dim=1)  # (B, input_dim)
        
        # Classification
        logits = self.fc(h_bar)
        probs = F.softmax(logits, dim=-1)
        
        return logits, probs
