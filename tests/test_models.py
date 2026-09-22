"""Unit tests for model components."""

import pytest
import torch
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.hss_rnet import HSSRNet, S_SSM
from models.mamba_backbone import MambaBlock, SelectiveSSM
from models.structured_ssm import StructuredSSM
from models.bigru_refinement import BiGRURefinement
from models.taag import TrafficAwareAdaptiveGating, AdaptiveGating


class TestHSSRNet:
    """Tests for HSS-RNet model."""
    
    def test_forward_pass(self):
        """Test forward pass shape."""
        batch_size, seq_len, input_dim = 4, 30, 11
        model = HSSRNet(input_dim=input_dim, num_classes=2)
        
        x = torch.randn(batch_size, seq_len, input_dim)
        logits, probs = model(x)
        
        assert logits.shape == (batch_size, 2)
        assert probs.shape == (batch_size, 2)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(batch_size), atol=1e-5)
    
    def test_with_mask(self):
        """Test forward pass with padding mask."""
        batch_size, seq_len, input_dim = 4, 30, 11
        model = HSSRNet(input_dim=input_dim, num_classes=2)
        
        x = torch.randn(batch_size, seq_len, input_dim)
        mask = torch.ones(batch_size, seq_len)
        mask[:, 20:] = 0  # Last 10 timesteps are padding
        
        logits, probs = model(x, mask)
        
        assert logits.shape == (batch_size, 2)
    
    def test_gradient_flow(self):
        """Test that gradients flow through all components."""
        model = HSSRNet(input_dim=11, num_classes=2)
        x = torch.randn(2, 30, 11)
        logits, _ = model(x)
        loss = logits.sum()
        loss.backward()
        
        # Check gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


class TestMambaBlock:
    """Tests for Mamba block."""
    
    def test_output_shape(self):
        """Test output shape preservation."""
        d_model = 64
        block = MambaBlock(d_model)
        x = torch.randn(2, 30, d_model)
        y = block(x)
        assert y.shape == x.shape


class TestBiGRU:
    """Tests for BiGRU refinement."""
    
    def test_output_dim(self):
        """Test output dimension is 2*hidden."""
        input_dim, hidden_dim = 128, 64
        bigru = BiGRURefinement(input_dim, hidden_dim)
        x = torch.randn(2, 30, input_dim)
        y = bigru(x)
        assert y.shape == (2, 30, 2 * hidden_dim)


class TestTAAG:
    """Tests for TAAG."""
    
    def test_gating_output(self):
        """Test gating output shape and range."""
        hidden_dim = 128
        taag = TrafficAwareAdaptiveGating(hidden_dim)
        
        h_rnn = torch.randn(2, 30, hidden_dim)
        h_mamba = torch.randn(2, 30, hidden_dim)
        
        y = taag(h_rnn, h_mamba)
        assert y.shape == (2, 30, hidden_dim)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
