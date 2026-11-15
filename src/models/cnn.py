"""
Model A: Convolutional Neural Network (CNN) for Multi-Output EEG Regression
Extracts spatial-temporal feature representations from 32-channel EEG windows
and predicts Valence, Arousal, Dominance, and Liking.
"""

from typing import Tuple, List, Optional
import numpy as np

# PyTorch Implementation
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class TorchCNN(nn.Module):
        """
        1D CNN for EEG multi-output regression (PyTorch).
        Input shape: (batch_size, timesteps, channels) -> (B, 256, 32)
        Output shape: (batch_size, 4)
        """
        def __init__(
            self,
            in_channels: int = 32,
            timesteps: int = 256,
            num_outputs: int = 4,
            dropout_rate: float = 0.3
        ):
            super().__init__()
            # Conv1d expects (batch_size, channels, timesteps)
            self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=5, padding=2)
            self.bn1 = nn.BatchNorm1d(32)
            self.pool1 = nn.MaxPool1d(kernel_size=2)
            self.drop1 = nn.Dropout(0.2)

            self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm1d(64)
            self.pool2 = nn.MaxPool1d(kernel_size=2)
            self.drop2 = nn.Dropout(0.2)

            self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm1d(128)
            self.pool3 = nn.AdaptiveAvgPool1d(1)
            self.drop3 = nn.Dropout(dropout_rate)

            self.fc1 = nn.Linear(128, 128)
            self.drop4 = nn.Dropout(dropout_rate)
            self.fc2 = nn.Linear(128, 64)
            self.out = nn.Linear(64, num_outputs)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x shape: (B, T, C) -> transpose to (B, C, T)
            x = x.transpose(1, 2)
            
            x = self.drop1(self.pool1(F.relu(self.bn1(self.conv1(x)))))
            x = self.drop2(self.pool2(F.relu(self.bn2(self.conv2(x)))))
            x = self.drop3(self.pool3(F.relu(self.bn3(self.conv3(x)))))
            
            x = x.squeeze(-1)  # (B, 128)
            x = self.drop4(F.relu(self.fc1(x)))
            x = F.relu(self.fc2(x))
            out = self.out(x)   # (B, 4)
            return out

except ImportError:
    TorchCNN = None


# TensorFlow / Keras Implementation (for Google Colab T4)
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, regularizers

    def build_keras_cnn(
        input_shape: Tuple[int, int] = (256, 32),
        num_outputs: int = 4,
        dropout_rate: float = 0.3,
        l2_reg: float = 1e-4
    ) -> tf.keras.Model:
        """
        Builds a Keras 1D CNN regression model for EEG time-series.
        Input shape: (256, 32)
        Output: (4,) continuous regression values
        """
        inputs = layers.Input(shape=input_shape, name="eeg_input")

        # Block 1
        x = layers.Conv1D(
            filters=32,
            kernel_size=5,
            padding="same",
            kernel_regularizer=regularizers.l2(l2_reg),
            name="conv1"
        )(inputs)
        x = layers.BatchNormalization(name="bn1")(x)
        x = layers.Activation("relu", name="relu1")(x)
        x = layers.MaxPooling1D(pool_size=2, name="pool1")(x)
        x = layers.SpatialDropout1D(0.2, name="drop1")(x)

        # Block 2
        x = layers.Conv1D(
            filters=64,
            kernel_size=3,
            padding="same",
            kernel_regularizer=regularizers.l2(l2_reg),
            name="conv2"
        )(x)
        x = layers.BatchNormalization(name="bn2")(x)
        x = layers.Activation("relu", name="relu2")(x)
        x = layers.MaxPooling1D(pool_size=2, name="pool2")(x)
        x = layers.SpatialDropout1D(0.2, name="drop2")(x)

        # Block 3
        x = layers.Conv1D(
            filters=128,
            kernel_size=3,
            padding="same",
            kernel_regularizer=regularizers.l2(l2_reg),
            name="conv3"
        )(x)
        x = layers.BatchNormalization(name="bn3")(x)
        x = layers.Activation("relu", name="relu3")(x)
        x = layers.GlobalAveragePooling1D(name="gap")(x)
        x = layers.Dropout(dropout_rate, name="drop3")(x)

        # Dense Head
        x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(l2_reg), name="dense1")(x)
        x = layers.Dropout(dropout_rate, name="drop4")(x)
        x = layers.Dense(64, activation="relu", name="dense2")(x)
        outputs = layers.Dense(num_outputs, activation="linear", name="affective_outputs")(x)

        model = models.Model(inputs=inputs, outputs=outputs, name="ModelA_CNN")
        return model

except ImportError:
    build_keras_cnn = None
