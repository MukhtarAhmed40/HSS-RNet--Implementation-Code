"""
Structured State-Space Model (S-SSM) backbone.
Implements DPLR parameterization with HiPPO initialization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class HiPPOInit:
    """HiPPO-based initialization for state-transition matrix."""
    
    @staticmethod
    def make_hippo(n: int) -> torch.Tensor:
        """Create HiPPO-LegS matrix."""
        A = torch.zeros(n, n)
        for i in range(n):
            for j in range(n):
                if i > j:
                    A[i, j] = math.sqrt(2 * i + 1) * math.sqrt(2 * j + 1)
                elif i == j:
                    A[i, j] = i + 1
        return A
    
    @staticmethod
    def make_hippo_diagonal(n: int) -> torch.Tensor:
        """Diagonal HiPPO approximation."""
        return torch.arange(1, n + 1, dtype=torch.float32)


class DPLRParameterization(nn.Module):
    """
    Diagonal-Plus-Low-Rank (DPLR) parameterization of state-transition matrix.
    A = Lambda - P Q^H
    """
    
    def __init__(self, d_state: int, rank: int = 1):
        super().__init__()
        self.d_state = d_state
        self.rank = rank
        
        # Diagonal component (HiPPO initialization)
        hippo_diag = HiPPOInit.make_hippo_diagonal(d_state)
        self.Lambda_real = nn.Parameter(hippo_diag.clone())
        self.Lambda_imag = nn.Parameter(torch.zeros(d_state))
        
        # Low-rank correction P, Q
        self.P = nn.Parameter(torch.randn(d_state, rank) / math.sqrt(d_state))
        self.Q = nn.Parameter(torch.randn(d_state, rank) / math.sqrt(d_state))
        
    def forward(self) -> torch.Tensor:
        """Construct the full state-transition matrix A_c."""
        # Diagonal matrix
        Lambda = torch.diag(torch.complex(self.Lambda_real, self.Lambda_imag))
        
        # Low-rank correction: P Q^H
        P_complex = torch.complex(self.P, torch.zeros_like(self.P))
        Q_complex = torch.complex(self.Q, torch.zeros_like(self.Q))
        low_rank = P_complex @ Q_complex.conj().T
        
        A_c = Lambda - low_rank
        return A_c


class StructuredSSM(nn.Module):
    """
    Structured State-Space Model with DPLR parameterization.
    
    Continuous-time formulation:
        dh/dt = A_c h(t) + B_c x(t)
        y(t)  = C_c h(t)
    
    Discretized via ZOH:
        A_bar = exp(Delta * A_c)
        B_bar = A_c^{-1} (A_bar - I) B_c
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 256,
        d_output: int = None,
        rank: int = 1,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_output = d_output or d_model
        self.rank = rank
        
        # DPLR state-transition matrix
        self.dplr = DPLRParameterization(d_state, rank)
        
        # Input projection B_c: (d_state, d_model)
        self.B_c = nn.Parameter(torch.randn(d_state, d_model) / math.sqrt(d_model))
        
        # Output projection C_c: (d_output, d_state)
        self.C_c = nn.Parameter(torch.randn(self.d_output, d_state) / math.sqrt(d_state))
        
        # Learnable step size
        dt = torch.exp(
            torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        )
        self.dt = nn.Parameter(dt)
        
        # Dimensionality reduction for efficiency
        self.x_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(self.d_output, d_model, bias=False)
        
    def _discretize(self):
        """Discretize the continuous-time system using ZOH."""
        A_c = self.dplr()  # (N, N) complex
        
        # Use real part for stable discretization
        A_c_real = A_c.real
        
        # A_bar = exp(Delta * A_c)
        dt = F.softplus(self.dt)  # (d_model,)
        dt_mean = dt.mean()
        
        # Discretized transition: exp(dt * A)
        A_bar = torch.matrix_exp(dt_mean * A_c_real)  # (N, N)
        
        # B_bar = A_c^{-1} (A_bar - I) B_c
        # Use pseudo-inverse for numerical stability
        I = torch.eye(self.d_state, device=A_c_real.device)
        
        try:
            A_inv = torch.linalg.pinv(A_c_real + 1e-6 * I)
            B_bar = A_inv @ (A_bar - I) @ self.B_c  # (N, d_model)
        except Exception:
            # Fallback: B_bar ≈ dt * B_c
            B_bar = dt_mean * self.B_c
        
        C_bar = self.C_c  # (d_output, N)
        
        return A_bar, B_bar, C_bar
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            y: (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape
        
        # Project input
        x_proj = self.x_proj(x)  # (B, T, d_model)
        
        # Discretize
        A_bar, B_bar, C_bar = self._discretize()
        
        # Sequential recurrence
        h = torch.zeros(batch, self.d_state, device=x.device, dtype=A_bar.dtype)
        outputs = []
        
        for t in range(seq_len):
            # h_t = A_bar h_{t-1} + B_bar x_t
            h = A_bar @ h.T  # (N, B)
            h = h.T + x_proj[:, t, :] @ B_bar.T  # (B, N)
            
            # y_t = C_bar h_t
            y_t = h @ C_bar.T  # (B, d_output)
            outputs.append(y_t)
        
        y = torch.stack(outputs, dim=1)  # (B, T, d_output)
        y = self.out_proj(y)  # (B, T, d_model)
        
        return y
