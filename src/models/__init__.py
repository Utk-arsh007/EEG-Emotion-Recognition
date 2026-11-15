"""
Model Factory and Registry Module
Allows seamless creation of CNN, BiLSTM, and Proposed Hybrid CNN-BiLSTM-Attention architectures
across both TensorFlow/Keras and PyTorch backends.
"""

from typing import Any, Optional, Dict
from src.config import CONFIG

from src.models.attention import TorchTemporalAttention, KerasTemporalAttention
from src.models.cnn import TorchCNN, build_keras_cnn
from src.models.bilstm import TorchBiLSTM, build_keras_bilstm
from src.models.hybrid_model import TorchHybridCNNBiLSTMAttention, build_keras_hybrid


def get_model(
    model_name: str = "hybrid",
    backend: str = "torch",
    in_channels: int = 32,
    timesteps: int = 256,
    num_outputs: int = 4,
    use_attention: bool = True,
    dropout_rate: float = 0.3,
    l2_reg: float = 1e-4
) -> Any:
    """
    Factory function to instantiate models.
    
    Args:
        model_name: 'cnn', 'bilstm', 'hybrid', or 'ablation' (hybrid without attention)
        backend: 'torch' (for local execution/testing) or 'keras'/'tf' (for Google Colab T4 GPU)
    """
    model_key = model_name.lower().strip()
    backend_key = backend.lower().strip()

    if backend_key in ["torch", "pytorch"]:
        if model_key == "cnn":
            return TorchCNN(
                in_channels=in_channels,
                timesteps=timesteps,
                num_outputs=num_outputs,
                dropout_rate=dropout_rate
            )
        elif model_key == "bilstm":
            return TorchBiLSTM(
                in_channels=in_channels,
                hidden_units=64,
                num_layers=2,
                num_outputs=num_outputs,
                dropout_rate=dropout_rate
            )
        elif model_key in ["hybrid", "proposed"]:
            return TorchHybridCNNBiLSTMAttention(
                in_channels=in_channels,
                timesteps=timesteps,
                num_outputs=num_outputs,
                use_attention=True,
                dropout_rate=dropout_rate
            )
        elif model_key in ["ablation", "hybrid_no_attention"]:
            return TorchHybridCNNBiLSTMAttention(
                in_channels=in_channels,
                timesteps=timesteps,
                num_outputs=num_outputs,
                use_attention=False,
                dropout_rate=dropout_rate
            )
        else:
            raise ValueError(f"Unknown model name '{model_name}'. Choose from 'cnn', 'bilstm', 'hybrid', 'ablation'")

    elif backend_key in ["keras", "tf", "tensorflow"]:
        if build_keras_cnn is None:
            raise ImportError("TensorFlow/Keras is not installed in the current environment. Use backend='torch' or run in Google Colab.")

        input_shape = (timesteps, in_channels)
        if model_key == "cnn":
            return build_keras_cnn(input_shape=input_shape, num_outputs=num_outputs, dropout_rate=dropout_rate, l2_reg=l2_reg)
        elif model_key == "bilstm":
            return build_keras_bilstm(input_shape=input_shape, num_outputs=num_outputs, dropout_rate=dropout_rate, l2_reg=l2_reg)
        elif model_key in ["hybrid", "proposed"]:
            return build_keras_hybrid(input_shape=input_shape, num_outputs=num_outputs, use_attention=True, dropout_rate=dropout_rate, l2_reg=l2_reg)
        elif model_key in ["ablation", "hybrid_no_attention"]:
            return build_keras_hybrid(input_shape=input_shape, num_outputs=num_outputs, use_attention=False, dropout_rate=dropout_rate, l2_reg=l2_reg)
        else:
            raise ValueError(f"Unknown model name '{model_name}'. Choose from 'cnn', 'bilstm', 'hybrid', 'ablation'")

    else:
        raise ValueError(f"Unsupported backend '{backend}'. Choose 'torch' or 'keras'")
