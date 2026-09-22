"""
Traffic-Aware Adaptive Gating (TAAG) mechanism.
Dynamically emphasizes informative temporal patterns and suppresses noise.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveGating(nn.Module):
    """
    Standard adaptive gating for S-SSM.
    
    g_t = sigma(W_g h_t^rnn + b_g)
    h_tilde_t = g_t * h_t^rnn
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W_g = nn.Linear(hidden_dim, hidden_dim)
        self.b_g = nn.Parameter(torch.zeros(hidden_dim))
        
    def forward(self, h_rnn: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_rnn: (B, T, 2H) BiGRU refined representations
        Returns:
            h_gated: (B, T, 2H)
        """
        g_t = torch.sigmoid(self.W_g(h_rnn) + self.b_g)
        return g_t * h_rnn


class TrafficAwareAdaptiveGating(nn.Module):
    """
    Traffic-Aware Adaptive Gating (TAAG) for HSS-RNet.
    
    g_t = sigma(W_g h_t^rnn + W_tau tau_t + b_g)
    h_tilde_t = g_t * h_t^rnn
    
    where tau_t = phi(h_t^mamba) is traffic-context from Mamba backbone.
    """
    
    def __init__(self, hidden_dim: int, context_dim: int = None):
        super().__init__()
        self.hidden_dim = hidden_dim
        context_dim = context_dim or hidden_dim
        
        # Gate computation
        self.W_g = nn.Linear(hidden_dim, hidden_dim)
        self.W_tau = nn.Linear(context_dim, hidden_dim)
        self.b_g = nn.Parameter(torch.zeros(hidden_dim))
        
        # Context projection phi
        self.phi = nn.Sequential(
            nn.Linear(context_dim, context_dim),
            nn.SiLU(),
            nn.Linear(context_dim, context_dim),
        )
        
    def forward(
        self,
        h_rnn: torch.Tensor,
        h_mamba: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            h_rnn: (B, T, 2H) BiGRU refined representations
            h_mamba: (B, T, 2H) Mamba backbone outputs (context)
        Returns:
            h_gated: (B, T, 2H)
        """
        # Traffic-aware context
        tau_t = self.phi(h_mamba)  # (B, T, 2H)
        
        # Gate computation
        g_t = torch.sigmoid(
            self.W_g(h_rnn) + self.W_tau(tau_t) + self.b_g
        )
        
        return g_t * h_rnn
