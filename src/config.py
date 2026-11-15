"""
EEG Emotion Recognition - Centralized Configuration Module
Provides typed dataclass representations and helper accessors for all hyperparameters,
paths, and split strategies.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import os
import yaml


@dataclass
class PathConfig:
    local_raw_dir: str = "./data/raw"
    local_processed_dir: str = "./data/processed"
    sample_data_dir: str = "./sample_data"
    checkpoint_dir: str = "./models/checkpoints"
    results_dir: str = "./results"
    reports_dir: str = "./reports"
    
    # Google Colab / Google Drive Paths
    colab_drive_root: str = "/content/drive/MyDrive"
    colab_raw_dir: str = "/content/drive/MyDrive/DEAP/data_preprocessed_python"
    colab_processed_dir: str = "/content/drive/MyDrive/DEAP/processed"
    colab_checkpoint_dir: str = "/content/drive/MyDrive/DEAP/checkpoints"


@dataclass
class DatasetConfig:
    num_participants: int = 32
    trials_per_participant: int = 40
    total_channels: int = 40
    eeg_channels_count: int = 32
    sampling_rate: int = 128  # Hz
    raw_samples_per_trial: int = 8064
    baseline_seconds: float = 3.0
    trial_seconds: float = 60.0
    
    baseline_samples: int = 384  # 3.0 * 128
    trial_samples: int = 7680    # 60.0 * 128

    channel_names: List[str] = field(default_factory=lambda: [
        "Fp1", "AF3", "F3", "F7", "FC5", "FC1", "C3", "T7",
        "CP5", "CP1", "P3", "P7", "PO3", "O1", "Oz", "Pz",
        "Fp2", "AF4", "Fz", "F4", "F8", "FC6", "FC2", "Cz",
        "C4", "T8", "CP6", "CP2", "P4", "P8", "PO4", "O2"
    ])
    
    target_names: List[str] = field(default_factory=lambda: [
        "Valence", "Arousal", "Dominance", "Liking"
    ])
    target_min: float = 1.0
    target_max: float = 9.0


@dataclass
class PreprocessingConfig:
    baseline_strategy: str = "subtract"  # 'subtract', 'discard', 'none'
    filter_enabled: bool = True
    filter_order: int = 4
    filter_lowcut: float = 4.0   # Hz
    filter_highcut: float = 45.0  # Hz
    window_size_seconds: float = 2.0
    window_stride_seconds: float = 1.0
    scaler_type: str = "standard"  # 'standard' or 'minmax'
    clip_outliers_sigma: float = 5.0

    @property
    def window_samples(self) -> int:
        return int(self.window_size_seconds * 128)

    @property
    def stride_samples(self) -> int:
        return int(self.window_stride_seconds * 128)


@dataclass
class DataSplitConfig:
    # 70% Train (22 subjects), 15% Val (5 subjects), 15% Test (5 subjects)
    train_participants: List[int] = field(default_factory=lambda: list(range(1, 23)))
    val_participants: List[int] = field(default_factory=lambda: list(range(23, 28)))
    test_participants: List[int] = field(default_factory=lambda: list(range(28, 33)))


@dataclass
class TrainingConfig:
    batch_size: int = 64
    epochs: int = 50
    learning_rate: float = 0.001
    loss: str = "huber"  # 'huber' or 'mse'
    huber_delta: float = 1.0
    optimizer: str = "adam"
    reduce_lr_patience: int = 4
    reduce_lr_factor: float = 0.5
    min_lr: float = 1e-6
    early_stopping_patience: int = 10
    use_mixed_precision: bool = False
    random_seed: int = 42


@dataclass
class AppConfig:
    paths: PathConfig = field(default_factory=PathConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    data_split: DataSplitConfig = field(default_factory=DataSplitConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """Load configuration from YAML file or return defaults."""
    cfg = AppConfig()
    
    if config_path is None:
        # Check standard locations
        possible_paths = [
            Path("configs/config.yaml"),
            Path("../configs/config.yaml"),
            Path("../../configs/config.yaml")
        ]
        for p in possible_paths:
            if p.exists():
                config_path = str(p)
                break
                
    if config_path and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
            
        if "paths" in raw:
            for k, v in raw["paths"].items():
                if hasattr(cfg.paths, k):
                    setattr(cfg.paths, k, v)
                    
        if "dataset" in raw:
            for k, v in raw["dataset"].items():
                if hasattr(cfg.dataset, k):
                    setattr(cfg.dataset, k, v)
                    
        if "preprocessing" in raw:
            prep = raw["preprocessing"]
            if "baseline_strategy" in prep:
                cfg.preprocessing.baseline_strategy = prep["baseline_strategy"]
            if "bandpass_filter" in prep:
                bf = prep["bandpass_filter"]
                cfg.preprocessing.filter_enabled = bf.get("enabled", True)
                cfg.preprocessing.filter_order = bf.get("order", 4)
                cfg.preprocessing.filter_lowcut = bf.get("lowcut", 4.0)
                cfg.preprocessing.filter_highcut = bf.get("highcut", 45.0)
            if "windowing" in prep:
                win = prep["windowing"]
                cfg.preprocessing.window_size_seconds = win.get("window_size_seconds", 2.0)
                cfg.preprocessing.window_stride_seconds = win.get("window_stride_seconds", 1.0)
            if "normalization" in prep:
                norm = prep["normalization"]
                cfg.preprocessing.scaler_type = norm.get("scaler_type", "standard")
                cfg.preprocessing.clip_outliers_sigma = norm.get("clip_outliers_sigma", 5.0)
                
        if "data_split" in raw:
            ds = raw["data_split"]
            if "train_participants" in ds:
                cfg.data_split.train_participants = ds["train_participants"]
            if "val_participants" in ds:
                cfg.data_split.val_participants = ds["val_participants"]
            if "test_participants" in ds:
                cfg.data_split.test_participants = ds["test_participants"]
                
        if "training" in raw:
            tr = raw["training"]
            for k, v in tr.items():
                if hasattr(cfg.training, k):
                    setattr(cfg.training, k, v)
                    
    return cfg


# Global singleton configuration instance
CONFIG = load_config()
