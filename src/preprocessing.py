"""
EEG Preprocessing Pipeline for DEAP Dataset
Includes:
- Baseline handling (mean baseline subtraction and segment extraction)
- Zero-phase Butterworth bandpass filtering (4 - 45 Hz)
- Zero-leakage standardizer (fit exclusively on training participants)
- Outlier clipping and signal validation
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from scipy import signal

from src.config import CONFIG, AppConfig


class EEGPreprocessor:
    """
    Robust EEG preprocessor adhering strictly to signal processing standards
    and zero data leakage principles.
    """

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or CONFIG
        self.fs = self.config.dataset.sampling_rate
        self.baseline_samples = self.config.dataset.baseline_samples
        self.trial_samples = self.config.dataset.trial_samples
        self.filter_enabled = self.config.preprocessing.filter_enabled
        self.filter_order = self.config.preprocessing.filter_order
        self.lowcut = self.config.preprocessing.filter_lowcut
        self.highcut = self.config.preprocessing.filter_highcut
        self.clip_sigma = self.config.preprocessing.clip_outliers_sigma

        # Precompute Butterworth filter coefficients
        if self.filter_enabled:
            nyquist = 0.5 * self.fs
            low = self.lowcut / nyquist
            high = self.highcut / nyquist
            self.b, self.a = signal.butter(self.filter_order, [low, high], btype='bandpass')
        else:
            self.b, self.a = None, None

    def remove_baseline(
        self,
        trial_data: np.ndarray,
        strategy: str = "subtract"
    ) -> np.ndarray:
        """
        Processes the 3-second baseline and 60-second trial segment.
        
        Args:
            trial_data: shape (32, 8064) or (trials, 32, 8064)
            strategy:
                'subtract': subtract mean of first 384 samples from the 7680 trial samples
                'discard': simply drop the first 384 samples
                'none': keep all 8064 samples
                
        Returns:
            processed_data: shape (32, 7680) or (trials, 32, 7680)
        """
        is_3d = (trial_data.ndim == 3)
        if not is_3d:
            trial_data = np.expand_dims(trial_data, axis=0)

        # trial_data shape: (N, 32, 8064)
        if strategy == "subtract":
            baseline_seg = trial_data[:, :, :self.baseline_samples]  # (N, 32, 384)
            active_seg = trial_data[:, :, self.baseline_samples:]     # (N, 32, 7680)
            baseline_mean = np.mean(baseline_seg, axis=-1, keepdims=True)  # (N, 32, 1)
            result = active_seg - baseline_mean
        elif strategy == "discard":
            result = trial_data[:, :, self.baseline_samples:]
        elif strategy == "none":
            result = trial_data
        else:
            raise ValueError(f"Unknown baseline strategy: {strategy}")

        if not is_3d:
            result = np.squeeze(result, axis=0)
        return result

    def bandpass_filter(self, eeg_data: np.ndarray) -> np.ndarray:
        """
        Applies a zero-phase Butterworth bandpass filter along the time dimension (axis=-1).
        Preserves phase alignment using filtfilt.
        
        Args:
            eeg_data: shape (..., samples)
        """
        if not self.filter_enabled:
            return eeg_data

        # scipy.signal.filtfilt performs forward-backward filtering (zero phase distortion)
        filtered = signal.filtfilt(self.b, self.a, eeg_data, axis=-1)
        return filtered.astype(np.float32)

    def clip_outliers(
        self,
        eeg_data: np.ndarray,
        mean: Optional[float] = None,
        std: Optional[float] = None
    ) -> np.ndarray:
        """
        Clips signal values beyond +/- clip_sigma standard deviations.
        """
        if mean is None:
            mean = np.mean(eeg_data)
        if std is None:
            std = np.std(eeg_data)
        if std < 1e-8:
            return eeg_data

        lower_bound = mean - self.clip_sigma * std
        upper_bound = mean + self.clip_sigma * std
        return np.clip(eeg_data, lower_bound, upper_bound)

    def preprocess_trial(
        self,
        raw_trial_eeg: np.ndarray,
        baseline_strategy: str = "subtract"
    ) -> np.ndarray:
        """
        Full single-trial preprocessing:
        1. Baseline subtraction / extraction
        2. Bandpass filtering (4-45 Hz)
        3. Outlier clipping
        
        Args:
            raw_trial_eeg: shape (32, 8064)
            
        Returns:
            preprocessed: shape (32, 7680)
        """
        # Step 1: Baseline removal
        active_eeg = self.remove_baseline(raw_trial_eeg, strategy=baseline_strategy)
        
        # Step 2: Zero-phase bandpass filtering
        filtered_eeg = self.bandpass_filter(active_eeg)
        
        # Step 3: Outlier clipping
        clean_eeg = self.clip_outliers(filtered_eeg)
        
        return clean_eeg


class ZeroLeakageStandardizer:
    """
    StandardScaler that computes global or channel-wise mean and std
    STRICTLY on the training participant dataset.
    Never exposes validation or test sets during fitting.
    """

    def __init__(self, per_channel: bool = True):
        self.per_channel = per_channel
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.is_fitted: bool = False

    def fit(self, train_data: np.ndarray) -> "ZeroLeakageStandardizer":
        """
        Fit on training array of shape (samples, timesteps, channels) or (samples, channels, timesteps).
        
        Args:
            train_data: np.ndarray of shape (N, T, C) where C is channel count (32).
        """
        if train_data.ndim != 3:
            raise ValueError(f"Expected 3D array (N, T, C), got shape {train_data.shape}")

        if self.per_channel:
            # Axis 0 (samples) and Axis 1 (timesteps) collapsed, keeping Axis 2 (channels)
            self.mean_ = np.mean(train_data, axis=(0, 1), keepdims=True, dtype=np.float64).astype(np.float32)
            self.std_ = np.std(train_data, axis=(0, 1), keepdims=True, dtype=np.float64).astype(np.float32)
            # Prevent division by zero
            self.std_[self.std_ < 1e-6] = 1.0
        else:
            self.mean_ = np.mean(train_data, dtype=np.float64).astype(np.float32)
            self.std_ = np.std(train_data, dtype=np.float64).astype(np.float32)
            if self.std_ < 1e-6:
                self.std_ = 1.0

        self.is_fitted = True
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Standardize data using previously computed training parameters."""
        if not self.is_fitted:
            raise RuntimeError("Standardizer must be fitted on training data before transform!")
        return (data - self.mean_) / self.std_

    def fit_transform(self, train_data: np.ndarray) -> np.ndarray:
        return self.fit(train_data).transform(train_data)

    def save(self, filepath: str) -> None:
        """Save fitted mean and std to file."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted standardizer.")
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        np.savez_compressed(
            filepath,
            mean=self.mean_,
            std=self.std_,
            per_channel=self.per_channel
        )

    def load(self, filepath: str) -> "ZeroLeakageStandardizer":
        """Load fitted parameters from file."""
        data = np.load(filepath)
        self.mean_ = data["mean"]
        self.std_ = data["std"]
        self.per_channel = bool(data["per_channel"])
        self.is_fitted = True
        return self
