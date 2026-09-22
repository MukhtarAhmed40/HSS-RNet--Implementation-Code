"""
Loss functions for HSS-RNet.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedCrossEntropyLoss(nn.Module):
    """
    Weighted categorical cross-entropy for class imbalance.
    
    L = -(1/N) sum_i sum_c w_c y_{i,c} log(y_hat_{i,c})
    """
    
    def __init__(self, class_weights: torch.Tensor = None):
        super().__init__()
        self.register_buffer("class_weights", class_weights)
        
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: (B, C) unnormalized logits
            targets: (B,) class indices
        Returns:
            loss: scalar
        """
        return F.cross_entropy(
            logits,
            targets,
            weight=self.class_weights,
        )


class FocalLoss(nn.Module):
    """Focal loss for hard example mining."""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
