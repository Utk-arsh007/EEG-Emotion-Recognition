"""
EEG Emotion Inference and Prediction Engine
Applies the exact same signal preprocessing, zero-leakage standardizer,
and trained Hybrid CNN-BiLSTM-Attention model to predict continuous affective dimensions:
Valence, Arousal, Dominance, and Liking.
Maps results to Russell's Circumplex Affective Space.
"""

import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np

from src.config import CONFIG, AppConfig
from src.preprocessing import EEGPreprocessor, ZeroLeakageStandardizer
from src.windowing import EEGWindowSegmenter
from src.models import get_model


class EEGPredictor:
    """
    End-to-end inference engine for 32-channel EEG emotion recognition.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        scaler_path: Optional[str] = None,
        backend: str = "torch",
        config: Optional[AppConfig] = None
    ):
        self.config = config or CONFIG
        self.backend = backend.lower()
        self.preprocessor = EEGPreprocessor(self.config)
        self.segmenter = EEGWindowSegmenter(self.config)
        
        # Load scaler
        s_path = scaler_path or "./data/processed/eeg_scaler.npz"
        self.scaler = ZeroLeakageStandardizer(per_channel=True)
        if os.path.exists(s_path):
            self.scaler.load(s_path)
        else:
            print(f"[Warning] Scaler file {s_path} not found. Operating with identity scaling until fitted.")

        # Load model checkpoint
        ckpt = checkpoint_path or "./models/checkpoints/best_hybrid_torch.pt"
        self.model = None
        self._load_model(ckpt)

    def _load_model(self, ckpt_path: str):
        if not os.path.exists(ckpt_path):
            # Fallback search
            alt_paths = [
                "./models/checkpoints/best_hybrid_torch.pt",
                "../models/checkpoints/best_hybrid_torch.pt"
            ]
            for ap in alt_paths:
                if os.path.exists(ap):
                    ckpt_path = ap
                    break

        if self.backend in ["torch", "pytorch"]:
            import torch
            self.model = get_model("hybrid", backend="torch", in_channels=32, timesteps=256)
            if os.path.exists(ckpt_path):
                self.model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
                print(f"[OK] Loaded PyTorch checkpoint from {ckpt_path}")
            else:
                print(f"[Warning] Checkpoint {ckpt_path} not found. Running with initialized weights.")
            self.model.eval()
        else:
            import tensorflow as tf
            from src.models.attention import KerasTemporalAttention
            if os.path.exists(ckpt_path):
                custom_objects = {"KerasTemporalAttention": KerasTemporalAttention}
                self.model = tf.keras.models.load_model(ckpt_path, custom_objects=custom_objects, compile=False)
                print(f"[OK] Loaded Keras checkpoint from {ckpt_path}")
            else:
                self.model = get_model("hybrid", backend="keras", in_channels=32, timesteps=256)

    def _map_circumplex_quadrant(self, valence: float, arousal: float) -> Dict[str, str]:
        """
        Maps (Valence, Arousal) continuous coordinates (1.0 - 9.0)
        to Russell's 1980 Circumplex Model of Affect.
        Threshold midpoint is 5.0.
        """
        is_high_val = (valence >= 5.0)
        is_high_aro = (arousal >= 5.0)

        if is_high_val and is_high_aro:
            quadrant = "HVHA (High Valence, High Arousal)"
            state = "Joy / Excitement / Alert Pleasure"
            color = "#16a34a"  # Green
        elif (not is_high_val) and is_high_aro:
            quadrant = "LVHA (Low Valence, High Arousal)"
            state = "Anger / Distress / Anxiety / Frustration"
            color = "#dc2626"  # Red
        elif (not is_high_val) and (not is_high_aro):
            quadrant = "LVLA (Low Valence, Low Arousal)"
            state = "Sadness / Depression / Boredom"
            color = "#475569"  # Slate Gray
        else:
            quadrant = "HVLA (High Valence, Low Arousal)"
            state = "Calmness / Serenity / Relaxation / Contentment"
            color = "#2563eb"  # Blue

        return {
            "quadrant": quadrant,
            "emotional_state": state,
            "theme_color": color
        }

    def predict_trial(
        self,
        trial_eeg: np.ndarray,
        return_trajectory: bool = True
    ) -> Dict[str, Any]:
        """
        Full prediction pipeline for a single EEG trial.
        
        Args:
            trial_eeg: numpy array of shape (40, 8064), (32, 8064), or (32, 7680)
            return_trajectory: whether to include window-by-window predictions
            
        Returns:
            Dictionary with 4 continuous predictions, quadrant, confidence, and trajectory.
        """
        # Ensure channel selection (first 32 EEG channels)
        if trial_eeg.shape[0] > 32:
            trial_eeg = trial_eeg[:32, :]

        # Preprocess: baseline removal if length is 8064
        if trial_eeg.shape[1] >= self.config.dataset.raw_samples_per_trial:
            clean_trial = self.preprocessor.preprocess_trial(trial_eeg)
        else:
            clean_trial = self.preprocessor.clip_outliers(self.preprocessor.bandpass_filter(trial_eeg))

        # Windowing: segment into (W, 256, 32)
        dummy_lbl = np.zeros(4, dtype=np.float32)
        windows, _, _ = self.segmenter.segment_trial(clean_trial, dummy_lbl, participant_id=0, trial_index=0)

        # Standardize using training distribution
        if self.scaler.is_fitted:
            norm_windows = self.scaler.transform(windows)
        else:
            norm_windows = windows

        # Inference
        if self.backend in ["torch", "pytorch"]:
            import torch
            with torch.no_grad():
                tensor_x = torch.from_numpy(norm_windows).float()
                preds = self.model(tensor_x, return_attention=False)
                win_preds = preds.cpu().numpy()
        else:
            win_preds = self.model.predict(norm_windows, batch_size=32, verbose=0)

        # Average over all windows in trial and bound to [1.0, 9.0]
        mean_preds = np.clip(np.mean(win_preds, axis=0), 1.0, 9.0)
        std_preds = np.std(win_preds, axis=0)

        valence = float(mean_preds[0])
        arousal = float(mean_preds[1])
        dominance = float(mean_preds[2])
        liking = float(mean_preds[3])

        affect_info = self._map_circumplex_quadrant(valence, arousal)

        result = {
            "predictions": {
                "Valence": round(valence, 2),
                "Arousal": round(arousal, 2),
                "Dominance": round(dominance, 2),
                "Liking": round(liking, 2)
            },
            "uncertainty_std": {
                "Valence": round(float(std_preds[0]), 3),
                "Arousal": round(float(std_preds[1]), 3),
                "Dominance": round(float(std_preds[2]), 3),
                "Liking": round(float(std_preds[3]), 3)
            },
            "circumplex": affect_info,
            "windows_analyzed": len(windows),
            "disclaimer": "Academic & Research Demonstration Only. Not intended for clinical or psychiatric diagnosis."
        }

        if return_trajectory:
            result["trajectory"] = {
                "window_indices": list(range(len(win_preds))),
                "valence": [round(float(v), 2) for v in np.clip(win_preds[:, 0], 1.0, 9.0)],
                "arousal": [round(float(a), 2) for a in np.clip(win_preds[:, 1], 1.0, 9.0)],
                "dominance": [round(float(d), 2) for d in np.clip(win_preds[:, 2], 1.0, 9.0)],
                "liking": [round(float(l), 2) for l in np.clip(win_preds[:, 3], 1.0, 9.0)]
            }

        return result

    def predict_from_file(self, file_path: str, trial_idx: int = 0) -> Dict[str, Any]:
        """Loads a .dat or .npy EEG file and predicts emotional dimensions."""
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if p.suffix == ".dat":
            with open(p, "rb") as f:
                content = pickle.load(f, encoding="latin1")
            raw_data = np.asarray(content["data"], dtype=np.float32)
            if raw_data.ndim == 3:
                trial_eeg = raw_data[trial_idx]
            else:
                trial_eeg = raw_data
        elif p.suffix == ".npy":
            raw_data = np.load(p)
            if raw_data.ndim == 3:
                trial_eeg = raw_data[trial_idx]
            else:
                trial_eeg = raw_data
        else:
            raise ValueError(f"Unsupported file format {p.suffix}. Expected .dat or .npy")

        return self.predict_trial(trial_eeg)
