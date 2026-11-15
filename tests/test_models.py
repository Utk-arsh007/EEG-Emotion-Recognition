"""
Unit tests for Model Architectures: CNN, BiLSTM, and Proposed Hybrid CNN-BiLSTM-Attention.
Verifies forward pass, output shape (batch_size, 4), and backward gradient propagation.
"""

import unittest
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.models import get_model


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch required for local model unit tests")
class TestModelArchitectures(unittest.TestCase):

    def setUp(self):
        self.batch_size = 8
        self.timesteps = 256
        self.channels = 32
        self.num_outputs = 4
        
        # Synthetic batch: (8, 256, 32)
        self.dummy_input = torch.randn(self.batch_size, self.timesteps, self.channels)
        self.dummy_targets = torch.randn(self.batch_size, self.num_outputs)

    def test_cnn_forward_and_backward(self):
        """Test Model A: CNN forward pass and loss backward pass."""
        model = get_model("cnn", backend="torch", in_channels=self.channels, timesteps=self.timesteps)
        model.train()
        
        preds = model(self.dummy_input)
        self.assertEqual(preds.shape, (self.batch_size, self.num_outputs))
        
        # Test backward pass
        criterion = nn.HuberLoss()
        loss = criterion(preds, self.dummy_targets)
        loss.backward()
        
        # Check gradients exist
        has_grads = any(p.grad is not None and torch.sum(torch.abs(p.grad)) > 0 for p in model.parameters())
        self.assertTrue(has_grads, "Gradients must propagate through CNN layers")

    def test_bilstm_forward_and_backward(self):
        """Test Model B: BiLSTM forward pass and loss backward pass."""
        model = get_model("bilstm", backend="torch", in_channels=self.channels, timesteps=self.timesteps)
        model.train()
        
        preds = model(self.dummy_input)
        self.assertEqual(preds.shape, (self.batch_size, self.num_outputs))
        
        criterion = nn.HuberLoss()
        loss = criterion(preds, self.dummy_targets)
        loss.backward()
        
        has_grads = any(p.grad is not None and torch.sum(torch.abs(p.grad)) > 0 for p in model.parameters())
        self.assertTrue(has_grads, "Gradients must propagate through BiLSTM layers")

    def test_proposed_hybrid_attention_forward_and_backward(self):
        """Test Model C: Proposed Hybrid CNN-BiLSTM-Attention model."""
        model = get_model("hybrid", backend="torch", in_channels=self.channels, timesteps=self.timesteps)
        model.train()
        
        preds, attn_weights = model(self.dummy_input, return_attention=True)
        self.assertEqual(preds.shape, (self.batch_size, self.num_outputs))
        self.assertEqual(attn_weights.shape, (self.batch_size, 64), "Attention weights must span sequence length")
        
        # Check attention weights sum to 1.0 (softmax property)
        weights_sum = torch.sum(attn_weights, dim=-1)
        np.testing.assert_allclose(weights_sum.detach().numpy(), 1.0, atol=1e-4)

        criterion = nn.HuberLoss()
        loss = criterion(preds, self.dummy_targets)
        loss.backward()
        
        has_grads = any(p.grad is not None and torch.sum(torch.abs(p.grad)) > 0 for p in model.parameters())
        self.assertTrue(has_grads, "Gradients must propagate through Hybrid Attention layers")

    def test_ablation_hybrid_no_attention(self):
        """Test ablation variant: Hybrid model without attention mechanism."""
        model = get_model("ablation", backend="torch", in_channels=self.channels, timesteps=self.timesteps)
        model.eval()
        with torch.no_grad():
            preds = model(self.dummy_input)
        self.assertEqual(preds.shape, (self.batch_size, self.num_outputs))


if __name__ == "__main__":
    unittest.main()
