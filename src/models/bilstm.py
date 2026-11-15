"""
Model B: Bidirectional Long Short-Term Memory (BiLSTM) for Multi-Output EEG Regression
Learns forward and backward temporal dependencies across EEG time-steps
and predicts Valence, Arousal, Dominance, and Liking.
"""

from typing import Tuple, List, Optional
import numpy as np

# PyTorch Implementation
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class TorchBiLSTM(nn.Module):
        """
        Bidirectional LSTM for EEG multi-output regression (PyTorch).
        Input shape: (batch_size, timesteps, channels) -> (B, 256, 32)
        Output shape: (batch_size, 4)
        """
        def __init__(
            self,
            in_channels: int = 32,
            hidden_units: int = 64,
            num_layers: int = 2,
            num_outputs: int = 4,
            dropout_rate: float = 0.3
        ):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=in_channels,
                hidden_size=hidden_units,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout_rate if num_layers > 1 else 0.0
            )
            # Bidirectional produces 2 * hidden_units output
            self.fc1 = nn.Linear(hidden_units * 2, 128)
            self.drop1 = nn.Dropout(dropout_rate)
            self.fc2 = nn.Linear(128, 64)
            self.drop2 = nn.Dropout(0.2)
            self.out = nn.Linear(64, num_outputs)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x shape: (B, T, C)
            lstm_out, (hn, cn) = self.lstm(x)  # lstm_out: (B, T, 2*H)
            
            # Take the representation from the last timestep or concatenate both directions
            # hn has shape (num_layers * 2, B, H). Last layer forward and backward:
            last_forward = hn[-2, :, :]
            last_backward = hn[-1, :, :]
            h_cat = torch.cat([last_forward, last_backward], dim=-1)  # (B, 2*H)
            
            x = self.drop1(F.relu(self.fc1(h_cat)))
            x = self.drop2(F.relu(self.fc2(x)))
            out = self.out(x)
            return out

except ImportError:
    TorchBiLSTM = None


# TensorFlow / Keras Implementation (for Google Colab T4)
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, regularizers

    def build_keras_bilstm(
        input_shape: Tuple[int, int] = (256, 32),
        num_outputs: int = 4,
        lstm_units: int = 64,
        dropout_rate: float = 0.3,
        l2_reg: float = 1e-4
    ) -> tf.keras.Model:
        """
        Builds a Keras BiLSTM regression model for EEG time-series.
        Input shape: (256, 32)
        Output: (4,) continuous regression values
        """
        inputs = layers.Input(shape=input_shape, name="eeg_input")

        # Layer 1: BiLSTM (return sequences)
        x = layers.Bidirectional(
            layers.LSTM(
                lstm_units,
                return_sequences=True,
                kernel_regularizer=regularizers.l2(l2_reg),
                dropout=0.2
            ),
            name="bilstm_1"
        )(inputs)

        # Layer 2: BiLSTM (last timestep)
        x = layers.Bidirectional(
            layers.LSTM(
                lstm_units,
                return_sequences=False,
                kernel_regularizer=regularizers.l2(l2_reg),
                dropout=0.2
            ),
            name="bilstm_2"
        )(x)

        # Dense Head
        x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(l2_reg), name="dense1")(x)
        x = layers.Dropout(dropout_rate, name="drop1")(x)
        x = layers.Dense(64, activation="relu", name="dense2")(x)
        x = layers.Dropout(0.2, name="drop2")(x)
        outputs = layers.Dense(num_outputs, activation="linear", name="affective_outputs")(x)

        model = models.Model(inputs=inputs, outputs=outputs, name="ModelB_BiLSTM")
        return model

except ImportError:
    build_keras_bilstm = None
