"""
Temporal Attention Mechanism for EEG Feature Learning
Implements Bahdanau-style self-attention over sequential time-steps:
Given hidden sequence H = [h_1, ..., h_T]:
  u_t = tanh(W * h_t + b)
  alpha_t = softmax(u_t^T * v)
  c = sum(alpha_t * h_t)
"""

from typing import Optional, Tuple
import numpy as np

# PyTorch Implementation
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class TorchTemporalAttention(nn.Module):
        """
        PyTorch implementation of temporal attention layer.
        Input shape: (batch_size, time_steps, hidden_dim)
        Output shape: (batch_size, hidden_dim), (batch_size, time_steps)
        """
        def __init__(self, hidden_dim: int, attention_dim: Optional[int] = None):
            super().__init__()
            att_dim = attention_dim or hidden_dim
            self.W = nn.Linear(hidden_dim, att_dim, bias=True)
            self.v = nn.Linear(att_dim, 1, bias=False)

        def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            # h: (B, T, D)
            u = torch.tanh(self.W(h))          # (B, T, att_dim)
            scores = self.v(u).squeeze(-1)      # (B, T)
            alpha = F.softmax(scores, dim=-1)   # (B, T)
            context = torch.bmm(alpha.unsqueeze(1), h).squeeze(1)  # (B, D)
            return context, alpha

except ImportError:
    TorchTemporalAttention = None


# TensorFlow / Keras Implementation (used in Google Colab T4)
try:
    import tensorflow as tf
    from tensorflow.keras import layers, initializers, regularizers

    @tf.keras.utils.register_keras_serializable(package="CustomLayers", name="KerasTemporalAttention")
    class KerasTemporalAttention(layers.Layer):
        """
        Keras custom layer for temporal Bahdanau-style self-attention.
        Input shape: (batch_size, time_steps, features)
        Output shape: (batch_size, features)
        """
        def __init__(self, attention_dim: int = 64, l2_reg: float = 1e-4, **kwargs):
            super().__init__(**kwargs)
            self.attention_dim = attention_dim
            self.l2_reg = l2_reg

        @classmethod
        def from_config(cls, config):
            return cls(**config)

        def build(self, input_shape):
            hidden_dim = input_shape[-1]
            self.W = self.add_weight(
                name="att_weight",
                shape=(hidden_dim, self.attention_dim),
                initializer="glorot_uniform",
                regularizer=regularizers.l2(self.l2_reg),
                trainable=True
            )
            self.b = self.add_weight(
                name="att_bias",
                shape=(self.attention_dim,),
                initializer="zeros",
                trainable=True
            )
            self.v = self.add_weight(
                name="att_context_vector",
                shape=(self.attention_dim, 1),
                initializer="glorot_uniform",
                regularizer=regularizers.l2(self.l2_reg),
                trainable=True
            )
            super().build(input_shape)

        def call(self, inputs, return_attention_weights=False):
            # inputs shape: (B, T, D)
            # u = tanh(H * W + b) -> (B, T, att_dim)
            u = tf.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
            # scores = u * v -> (B, T, 1)
            scores = tf.tensordot(u, self.v, axes=1)
            # alpha = softmax(scores) along time dimension -> (B, T, 1)
            alpha = tf.nn.softmax(scores, axis=1)
            # context = sum(alpha * inputs) -> (B, D)
            context = tf.reduce_sum(alpha * inputs, axis=1)

            if return_attention_weights:
                return context, tf.squeeze(alpha, axis=-1)
            return context

        def get_config(self):
            config = super().get_config()
            config.update({
                "attention_dim": self.attention_dim,
                "l2_reg": self.l2_reg
            })
            return config

except ImportError:
    KerasTemporalAttention = None
