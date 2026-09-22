"""
HSS-RNet: Hybrid Selective State-Space Network for Network Intrusion Detection.
Main model integrating Mamba backbone, BiGRU refinement, and TAAG.
"""

import torch
import torch.nn as nn

from .mamba_backbone import MambaBlock
from .structured_ssm import StructuredSSM
from .bigru_refinement import BiGRURefinement
from .taag import TrafficAwareAdaptiveGating, AdaptiveGating
from .classifier import ClassificationLayer


class FeatureEmbedding(nn.Module):
    """Feature embedding module."""
    
    def __init__(self, input_dim: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class HSSRNet(nn.Module):
    """
    HSS-RNet architecture:
        X -> Phi_emb -> Phi_SSM -> Phi_BiGRU -> Phi_TAAG -> Phi_cls -> Y_hat
    
    Components:
        - Feature embedding
        - Selective state-space (Mamba) backbone
        - BiGRU refinement module
        - Traffic-Aware Adaptive Gating (TAAG)
        - Classification layer
    """
    
    def __init__(
        self,
        input_dim: int = 11,
        d_model: int = 128,
        d_state: int = 256,
        d_conv: int = 4,
        expand: int = 2,
        bigru_hidden: int = 128,
        num_classes: int = 2,
        dropout: float = 0.3,
        num_mamba_layers: int = 2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_classes = num_classes
        
        # Feature embedding
        self.embedding = FeatureEmbedding(input_dim, d_model, dropout)
        
        # Mamba backbone (stacked blocks)
        self.mamba_blocks = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand)
            for _ in range(num_mamba_layers)
        ])
        self.mamba_norm = nn.LayerNorm(d_model)
        
        # BiGRU refinement
        self.bigru = BiGRURefinement(d_model, bigru_hidden)
        bigru_output_dim = 2 * bigru_hidden
        
        # Project Mamba output for TAAG context
        self.mamba_context_proj = nn.Linear(d_model, bigru_output_dim)
        
        # TAAG
        self.taag = TrafficAwareAdaptiveGating(
            hidden_dim=bigru_output_dim,
            context_dim=bigru_output_dim,
        )
        
        # Classification
        self.classifier = ClassificationLayer(bigru_output_dim, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        return_features: bool = False,
    ):
        """
        Args:
            x: (B, T, input_dim) input traffic sequences
            mask: (B, T) padding mask (1=valid, 0=padding)
            return_features: whether to return intermediate features
        Returns:
            logits: (B, num_classes)
            probs: (B, num_classes)
            features: dict (optional)
        """
        # Feature embedding
        h_emb = self.embedding(x)  # (B, T, d_model)
        
        # Mamba backbone
        h_mamba = h_emb
        for block in self.mamba_blocks:
            h_mamba = block(h_mamba)
        h_mamba = self.mamba_norm(h_mamba)  # (B, T, d_model)
        
        # BiGRU refinement
        h_rnn = self.bigru(h_mamba)  # (B, T, 2H)
        
        # Prepare Mamba context for TAAG
        mamba_context = self.mamba_context_proj(h_mamba)  # (B, T, 2H)
        
        # TAAG
        h_gated = self.taag(h_rnn, mamba_context)  # (B, T, 2H)
        h_gated = self.dropout(h_gated)
        
        # Apply mask if provided
        if mask is not None:
            h_gated = h_gated * mask.unsqueeze(-1)
        
        # Classification
        logits, probs = self.classifier(h_gated)
        
        if return_features:
            features = {
                "embedding": h_emb,
                "mamba": h_mamba,
                "bigru": h_rnn,
                "gated": h_gated,
            }
            return logits, probs, features
        
        return logits, probs


class S_SSM(nn.Module):
    """
    S-SSM: Structured State-Space Model architecture.
    Uses fixed DPLR state transitions instead of selective Mamba.
    """
    
    def __init__(
        self,
        input_dim: int = 11,
        d_model: int = 128,
        d_state: int = 256,
        bigru_hidden: int = 128,
        num_classes: int = 2,
        dropout: float = 0.3,
        num_ssm_layers: int = 2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Feature embedding
        self.embedding = FeatureEmbedding(input_dim, d_model, dropout)
        
        # Structured SSM backbone
        self.ssm_layers = nn.ModuleList([
            StructuredSSM(d_model, d_state)
            for _ in range(num_ssm_layers)
        ])
        self.ssm_norm = nn.LayerNorm(d_model)
        
        # BiGRU refinement
        self.bigru = BiGRURefinement(d_model, bigru_hidden)
        bigru_output_dim = 2 * bigru_hidden
        
        # Standard adaptive gating
        self.gating = AdaptiveGating(bigru_output_dim)
        
        # Classification
        self.classifier = ClassificationLayer(bigru_output_dim, num_classes)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        return_features: bool = False,
    ):
        """Forward pass for S-SSM."""
        # Feature embedding
        h_emb = self.embedding(x)
        
        # Structured SSM backbone
        h_ssm = h_emb
        for layer in self.ssm_layers:
            h_ssm = layer(h_ssm) + h_ssm  # Residual
        h_ssm = self.ssm_norm(h_ssm)
        
        # BiGRU refinement
        h_rnn = self.bigru(h_ssm)
        
        # Adaptive gating
        h_gated = self.gating(h_rnn)
        h_gated = self.dropout(h_gated)
        
        # Apply mask
        if mask is not None:
            h_gated = h_gated * mask.unsqueeze(-1)
        
        # Classification
        logits, probs = self.classifier(h_gated)
        
        if return_features:
            features = {
                "embedding": h_emb,
                "ssm": h_ssm,
                "bigru": h_rnn,
                "gated": h_gated,
            }
            return logits, probs, features
        
        return logits, probs
