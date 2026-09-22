"""
Bidirectional GRU refinement module.
Recovers fine-grained local patterns oversmoothed by linear SSM dynamics.
"""

import torch
import torch.nn as nn


class BiGRURefinement(nn.Module):
    """
    Single-layer bidirectional GRU for local temporal refinement.
    
    Forward GRU:
        z_t^f = sigma(W_z^f y_t + U_z^f h_{t-1}^f + b_z^f)
        r_t^f = sigma(W_r^f y_t + U_r^f h_{t-1}^f + b_r^f)
        h_tilde_t^f = tanh(W_h^f y_t + U_h^f (r_t^f * h_{t-1}^f) + b_h^f)
        h_t^f = (1 - z_t^f) * h_{t-1}^f + z_t^f * h_tilde_t^f
    
    Output: h_t^mn = [h_t^f; h_t^b] in R^{2H}
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = 2 * hidden_dim
        
        self.bigru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        
        self.layer_norm = nn.LayerNorm(self.output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim) backbone outputs
        Returns:
            h_mn: (batch, seq_len, 2*hidden_dim) refined representations
        """
        output, _ = self.bigru(x)  # (B, T, 2*H)
        output = self.layer_norm(output)
        return output
