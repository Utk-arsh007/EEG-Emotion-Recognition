"""
Model C: Proposed Hybrid CNN-BiLSTM-Attention Deep Learning Architecture
Combines:
  1. 1D CNN for local temporal/spatial feature extraction across 32 EEG channels.
  2. Bidirectional LSTM for forward and backward long-term temporal dependencies.
  3. Bahdanau-style Temporal Attention mechanism to weight salient affective segments.
  4. Fully connected regression head predicting continuous Valence, Arousal, Dominance, Liking.
Includes support for ablation study (with vs. without attention).
"""

from typing import Tuple, List, Optional
import numpy as np

# PyTorch Implementation
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from src.models.attention import TorchTemporalAttention

    class TorchHybridCNNBiLSTMAttention(nn.Module):
        """
        Proposed Hybrid CNN-BiLSTM-Attention for EEG regression (PyTorch).
        Input shape: (batch_size, timesteps, channels) -> (B, 256, 32)
        Output shape: (batch_size, 4)
        """
        def __init__(
            self,
            in_channels: int = 32,
            timesteps: int = 256,
            num_outputs: int = 4,
            use_attention: bool = True,
            dropout_rate: float = 0.3
        ):
            super().__init__()
            self.use_attention = use_attention

            # Step 1: CNN Feature Extraction (Expects (B, C, T))
            self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=5, padding=2)
            self.bn1 = nn.BatchNorm1d(32)
            self.pool1 = nn.MaxPool1d(kernel_size=2)  # 256 -> 128
            self.drop1 = nn.Dropout(0.2)

            self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm1d(64)
            self.pool2 = nn.MaxPool1d(kernel_size=2)  # 128 -> 64
            self.drop2 = nn.Dropout(0.2)

            # Step 2: BiLSTM Temporal Learning
            # Input to LSTM: (B, 64 time_steps, 64 features)
            self.bilstm1 = nn.LSTM(
                input_size=64,
                hidden_size=64,
                batch_first=True,
                bidirectional=True
            )
            self.bilstm2 = nn.LSTM(
                input_size=128,  # 64*2 from first BiLSTM
                hidden_size=32,
                batch_first=True,
                bidirectional=True
            )

            # Step 3: Attention Mechanism (BiLSTM2 output has dimension 32*2 = 64)
            if self.use_attention:
                self.attention = TorchTemporalAttention(hidden_dim=64, attention_dim=32)
            else:
                self.attention = None

            # Step 4: Fully Connected Regression Head
            self.fc1 = nn.Linear(64, 64)
            self.drop3 = nn.Dropout(dropout_rate)
            self.fc2 = nn.Linear(64, 32)
            self.drop4 = nn.Dropout(0.2)
            self.out = nn.Linear(32, num_outputs)

        def forward(self, x: torch.Tensor, return_attention: bool = False):
            # x shape: (B, T, C) -> transpose to (B, C, T) for Conv1D
            x = x.transpose(1, 2)
            
            x = self.drop1(self.pool1(F.relu(self.bn1(self.conv1(x)))))
            x = self.drop2(self.pool2(F.relu(self.bn2(self.conv2(x)))))
            
            # Transpose back to (B, T_reduced, C_features) for LSTM: (B, 64, 64)
            x = x.transpose(1, 2)
            
            lstm_out1, _ = self.bilstm1(x)         # (B, 64, 128)
            lstm_out2, _ = self.bilstm2(lstm_out1)  # (B, 64, 64)

            attn_weights = None
            if self.use_attention:
                context, attn_weights = self.attention(lstm_out2)  # (B, 64)
            else:
                # Ablation fallback: Global Average Pooling over time dimension
                context = torch.mean(lstm_out2, dim=1)

            h = self.drop3(F.relu(self.fc1(context)))
            h = self.drop4(F.relu(self.fc2(h)))
            preds = self.out(h)

            if return_attention:
                return preds, attn_weights
            return preds

except ImportError:
    TorchHybridCNNBiLSTMAttention = None


# TensorFlow / Keras Implementation (for Google Colab T4)
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, regularizers
    from src.models.attention import KerasTemporalAttention

    def build_keras_hybrid(
        input_shape: Tuple[int, int] = (256, 32),
        num_outputs: int = 4,
        use_attention: bool = True,
        dropout_rate: float = 0.3,
        l2_reg: float = 1e-4
    ) -> tf.keras.Model:
        """
        Builds the proposed Keras Hybrid CNN-BiLSTM-Attention architecture.
        
        Args:
            input_shape: (timesteps, channels), e.g. (256, 32)
            num_outputs: 4 continuous outputs
            use_attention: True for proposed model, False for ablation baseline
            dropout_rate: dropout probability
            l2_reg: L2 weight penalty
            
        Returns:
            tf.keras.Model
        """
        inputs = layers.Input(shape=input_shape, name="eeg_input")

        # --- Phase 1: CNN Spatial-Local Feature Extraction ---
        # Block 1
        x = layers.Conv1D(
            filters=32,
            kernel_size=5,
            padding="same",
            kernel_regularizer=regularizers.l2(l2_reg),
            name="cnn_conv1"
        )(inputs)
        x = layers.BatchNormalization(name="cnn_bn1")(x)
        x = layers.Activation("relu", name="cnn_relu1")(x)
        x = layers.MaxPooling1D(pool_size=2, name="cnn_pool1")(x)  # 256 -> 128
        x = layers.Dropout(0.2, name="cnn_drop1")(x)

        # Block 2
        x = layers.Conv1D(
            filters=64,
            kernel_size=3,
            padding="same",
            kernel_regularizer=regularizers.l2(l2_reg),
            name="cnn_conv2"
        )(x)
        x = layers.BatchNormalization(name="cnn_bn2")(x)
        x = layers.Activation("relu", name="cnn_relu2")(x)
        x = layers.MaxPooling1D(pool_size=2, name="cnn_pool2")(x)  # 128 -> 64
        x = layers.Dropout(0.2, name="cnn_drop2")(x)

        # --- Phase 2: BiLSTM Temporal Dependency Learning ---
        x = layers.Bidirectional(
            layers.LSTM(64, return_sequences=True, dropout=0.25, kernel_regularizer=regularizers.l2(l2_reg)),
            name="bilstm_layer1"
        )(x)
        x = layers.Bidirectional(
            layers.LSTM(32, return_sequences=True, dropout=0.25, kernel_regularizer=regularizers.l2(l2_reg)),
            name="bilstm_layer2"
        )(x)

        # --- Phase 3: Temporal Attention Mechanism ---
        if use_attention:
            context = KerasTemporalAttention(attention_dim=32, l2_reg=l2_reg, name="temporal_attention")(x)
        else:
            context = layers.GlobalAveragePooling1D(name="ablation_gap")(x)

        # --- Phase 4: Dense Regression Head ---
        dense1 = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(l2_reg), name="dense_1")(context)
        drop_dense1 = layers.Dropout(dropout_rate, name="drop_dense1")(dense1)
        dense2 = layers.Dense(32, activation="relu", name="dense_2")(drop_dense1)
        drop_dense2 = layers.Dropout(0.2, name="drop_dense2")(dense2)

        # Continuous 4-Output Predictions [Valence, Arousal, Dominance, Liking]
        outputs = layers.Dense(num_outputs, activation="linear", name="affective_predictions")(drop_dense2)

        model_name = "Proposed_Hybrid_CNN_BiLSTM_Attention" if use_attention else "Ablation_Hybrid_No_Attention"
        model = models.Model(inputs=inputs, outputs=outputs, name=model_name)
        return model

except ImportError:
    build_keras_hybrid = None
