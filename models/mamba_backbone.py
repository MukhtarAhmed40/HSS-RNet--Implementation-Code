"""
Selective State-Space (Mamba) backbone for HSS-RNet.
Implements input-dependent state transitions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SelectiveSSM(nn.Module):
    """
    Selective State-Space Model (Mamba-style).
    
    Input-dependent parameters:
        Delta_t = softplus(f_Delta(z_t))
        B_t = f_B(z_t)
        C_t = f_C(z_t)
    
    Discretization:
        A_bar_t = exp(Delta_t * A_M)
        B_bar_t = Delta_t * B_t
        C_bar_t = C_t
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 256,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.dt_rank = dt_rank or max(1, d_model // 16)
        
        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # 1D convolution for local patterns
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True,
        )
        
        # SSM parameters
        # A_M: (d_inner, d_state)
        A_init = torch.arange(1, d_state + 1, dtype=torch.float32)
        A_init = A_init.unsqueeze(0).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A_init))
        
        # D parameter (skip connection)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Input-dependent projections
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        # Initialize dt bias
        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        
        # Initialize dt bias for uniform distribution
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        )
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            y: (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape
        
        # Input projection and split
        xz = self.in_proj(x)  # (B, T, 2*d_inner)
        x_ssm, z = xz.chunk(2, dim=-1)  # Each (B, T, d_inner)
        
        # Convolution for local patterns
        x_conv = x_ssm.transpose(1, 2)  # (B, d_inner, T)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]  # Causal conv
        x_conv = x_conv.transpose(1, 2)  # (B, T, d_inner)
        x_conv = F.silu(x_conv)
        
        # Compute input-dependent parameters
        x_dbl = self.x_proj(x_conv)  # (B, T, dt_rank + 2*d_state)
        dt, B_t, C_t = x_dbl.split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        
        # Delta_t = softplus(dt_proj(dt))
        dt = self.dt_proj(dt)  # (B, T, d_inner)
        dt = F.softplus(dt)  # (B, T, d_inner)
        
        # A_M = -exp(A_log) (ensure negative for stability)
        A = -torch.exp(self.A_log)  # (d_inner, d_state)
        
        # Discretize: A_bar_t = exp(Delta_t * A)
        # dt: (B, T, d_inner), A: (d_inner, d_state)
        # A_bar_t: (B, T, d_inner, d_state)
        A_bar = torch.exp(
            dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)
        )  # (B, T, d_inner, d_state)
        
        # B_bar_t = Delta_t * B_t
        # B_t: (B, T, d_state), dt: (B, T, d_inner)
        B_bar = dt.unsqueeze(-1) * B_t.unsqueeze(2)  # (B, T, d_inner, d_state)
        
        # C_bar_t = C_t
        C_bar = C_t  # (B, T, d_state)
        
        # Selective scan (sequential for simplicity; parallel scan in production)
        h = torch.zeros(batch, self.d_inner, self.d_state, device=x.device)
        ys = []
        
        for t in range(seq_len):
            # h_t = A_bar_t h_{t-1} + B_bar_t x_t
            h = A_bar[:, t] * h + B_bar[:, t] * x_conv[:, t].unsqueeze(-1)
            # y_t = C_bar_t h_t
            y_t = (h * C_bar[:, t].unsqueeze(1)).sum(dim=-1)  # (B, d_inner)
            ys.append(y_t)
        
        y_ssm = torch.stack(ys, dim=1)  # (B, T, d_inner)
        
        # Add skip connection
        y_ssm = y_ssm + self.D.unsqueeze(0).unsqueeze(0) * x_conv
        
        # Gated output
        y = y_ssm * F.silu(z)
        
        # Output projection
        y = self.out_proj(y)
        
        return y


class MambaBlock(nn.Module):
    """Complete Mamba block with normalization."""
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 256,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state, d_conv, expand)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Residual connection."""
        return x + self.ssm(self.norm(x))
