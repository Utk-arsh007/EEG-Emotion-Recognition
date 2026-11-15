"""
Unit tests for EEG signal preprocessing, filtering, windowing, and zero-leakage scaling.
"""

import unittest
import numpy as np
from pathlib import Path

from src.config import CONFIG
from src.preprocessing import EEGPreprocessor, ZeroLeakageStandardizer
from src.windowing import EEGWindowSegmenter, build_split_datasets
from sample_data.generate_sample_eeg import create_sample_dataset


class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        self.prep = EEGPreprocessor()
        self.segmenter = EEGWindowSegmenter()

    def test_baseline_removal_subtract(self):
        """Test that mean baseline (first 384 samples) is subtracted from the trial."""
        fs = 128
        raw_trial = np.ones((32, 8064), dtype=np.float32)
        # Give baseline an offset of 5.0
        raw_trial[:, :384] = 5.0
        # Give trial segment values of 10.0
        raw_trial[:, 384:] = 10.0
        
        result = self.prep.remove_baseline(raw_trial, strategy="subtract")
        
        self.assertEqual(result.shape, (32, 7680), "Trial length must be 7680 samples after baseline removal")
        # 10.0 - 5.0 baseline = 5.0
        np.testing.assert_allclose(result, 5.0, atol=1e-5)

    def test_bandpass_filter_attenuates_out_of_band(self):
        """Test that frequencies outside 4-45 Hz (e.g. 1 Hz slow drift and 60 Hz hum) are attenuated."""
        fs = 128
        t = np.linspace(0, 10.0, 10 * fs, endpoint=False)
        
        # 1 Hz slow wave (should be attenuated by highpass at 4Hz)
        low_freq_sig = np.sin(2 * np.pi * 1.0 * t)
        # 10 Hz alpha wave (should pass through 4-45Hz band)
        in_band_sig = np.sin(2 * np.pi * 10.0 * t)
        # 60 Hz noise (should be attenuated by lowpass at 45Hz)
        high_freq_sig = np.sin(2 * np.pi * 60.0 * t)
        
        combined = (low_freq_sig + in_band_sig + high_freq_sig).astype(np.float32)
        filtered = self.prep.bandpass_filter(combined)
        
        # FFT power ratio
        fft_orig = np.abs(np.fft.rfft(combined))
        fft_filt = np.abs(np.fft.rfft(filtered))
        freqs = np.fft.rfftfreq(len(t), 1.0/fs)
        
        idx_1hz = np.argmin(np.abs(freqs - 1.0))
        idx_10hz = np.argmin(np.abs(freqs - 10.0))
        idx_60hz = np.argmin(np.abs(freqs - 60.0))
        
        # 10 Hz power should remain largely intact
        self.assertGreater(fft_filt[idx_10hz] / fft_orig[idx_10hz], 0.85)
        # 1 Hz power should be significantly attenuated
        self.assertLess(fft_filt[idx_1hz] / fft_orig[idx_1hz], 0.30)
        # 60 Hz power should be significantly attenuated
        self.assertLess(fft_filt[idx_60hz] / fft_orig[idx_60hz], 0.20)

    def test_window_segmentation_dimensions(self):
        """Verify windowing produces correct (num_windows, 256, 32) tensors and label mappings."""
        trial_eeg = np.random.randn(32, 7680).astype(np.float32)
        trial_labels = np.array([6.5, 7.2, 5.0, 8.0], dtype=np.float32)
        
        # With trial length 7680, window 256, stride 128:
        # num_windows = (7680 - 256) // 128 + 1 = 59
        windows, w_labels, metadata = self.segmenter.segment_trial(
            trial_eeg, trial_labels, participant_id=1, trial_index=0
        )
        
        self.assertEqual(windows.shape, (59, 256, 32))
        self.assertEqual(w_labels.shape, (59, 4))
        self.assertEqual(len(metadata), 59)
        # Every window label must match the trial label
        for i in range(59):
            np.testing.assert_allclose(w_labels[i], trial_labels)

    def test_zero_leakage_standardizer(self):
        """Verify standardizer computes parameters on train only and does not alter when applied to test."""
        scaler = ZeroLeakageStandardizer(per_channel=True)
        rng = np.random.default_rng(123)
        
        # Fake train: mean ~10, std ~5
        X_train = rng.normal(loc=10.0, scale=5.0, size=(100, 256, 32)).astype(np.float32)
        # Fake test: mean ~20, std ~8
        X_test = rng.normal(loc=20.0, scale=8.0, size=(20, 256, 32)).astype(np.float32)
        
        X_train_norm = scaler.fit_transform(X_train)
        self.assertTrue(scaler.is_fitted)
        
        # Train should have zero mean and unit variance per channel
        train_means = np.mean(X_train_norm, axis=(0, 1))
        train_stds = np.std(X_train_norm, axis=(0, 1))
        np.testing.assert_allclose(train_means, 0.0, atol=1e-1)
        np.testing.assert_allclose(train_stds, 1.0, atol=1e-1)
        
        # Transform test set: test must use train mean & std
        X_test_norm = scaler.transform(X_test)
        # Check that test is NOT zero mean because it was shifted by train mean (10), not test mean (20)
        test_means = np.mean(X_test_norm, axis=(0, 1))
        self.assertTrue(np.all(test_means > 1.0), "Test set must not be centered on its own mean (leakage check)!")


if __name__ == "__main__":
    unittest.main()
