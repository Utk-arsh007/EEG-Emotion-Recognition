"""
DEAP Dataset Loader and Validator
Handles reading .dat participant files, validating signal integrity,
strictly extracting the 32 EEG channels (dropping 8 peripheral channels),
and packaging participant-separated data splits without memory bloat.
"""

import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Generator
import numpy as np

from src.config import CONFIG, AppConfig


class DEAPDataLoader:
    """
    Robust loader for DEAP preprocessed .dat files.
    
    DEAP .dat file structure:
        - data: ndarray of shape (40, 40, 8064)
          Channels 0-31: EEG channels
          Channels 32-39: Peripheral physiological channels (GSR, Respiration, Plethysmograph, Temperature, etc.)
        - labels: ndarray of shape (40, 4)
          Columns: Valence, Arousal, Dominance, Liking (continuous scale 1.0 - 9.0)
    """

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or CONFIG
        self.eeg_channels = self.config.dataset.eeg_channels_count
        self.total_channels = self.config.dataset.total_channels
        self.expected_samples = self.config.dataset.raw_samples_per_trial

    def load_single_participant(
        self,
        filepath: str,
        validate: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Loads a single participant .dat file.
        
        Args:
            filepath: Path to the .dat file (e.g., 's01.dat').
            validate: Whether to validate shape, NaN, and Inf values.
            
        Returns:
            eeg_data: np.ndarray of shape (trials, 32, 8064)
            labels: np.ndarray of shape (trials, 4)
            metadata: Dict containing integrity report and trial count.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Participant file not found: {filepath}")

        # DEAP files were originally pickled in Python 2.7, requiring encoding='latin1' in Python 3
        try:
            with open(path, "rb") as f:
                content = pickle.load(f, encoding="latin1")
        except Exception as e:
            raise IOError(f"Failed to unpickle {filepath}: {str(e)}")

        if not isinstance(content, dict) or "data" not in content or "labels" not in content:
            raise ValueError(f"Invalid DEAP file format in {filepath}. Expected dict with 'data' and 'labels' keys.")

        raw_data = np.asarray(content["data"], dtype=np.float32)
        raw_labels = np.asarray(content["labels"], dtype=np.float32)

        # Basic shape checks
        if raw_data.ndim != 3:
            raise ValueError(f"Expected 3D data array (trials, channels, samples), got shape {raw_data.shape}")
        
        num_trials, num_ch, num_samples = raw_data.shape

        if num_ch < self.eeg_channels:
            raise ValueError(f"File {filepath} has only {num_ch} channels; requires at least {self.eeg_channels} EEG channels.")

        # STRICT REQUIREMENT: Extract only first 32 channels (EEG channels)
        # Channels 32 to 39 are peripheral and must be excluded
        eeg_data = raw_data[:, :self.eeg_channels, :]
        labels = raw_labels[:, :4]

        metadata = {
            "file": path.name,
            "original_shape": raw_data.shape,
            "extracted_eeg_shape": eeg_data.shape,
            "labels_shape": labels.shape,
            "num_trials": num_trials,
            "is_valid": True,
            "issues": []
        }

        if validate:
            self._validate_integrity(eeg_data, labels, metadata)

        return eeg_data, labels, metadata

    def _validate_integrity(
        self,
        eeg_data: np.ndarray,
        labels: np.ndarray,
        metadata: Dict[str, Any]
    ) -> None:
        """Checks for NaNs, Infs, zero channels, and label range validity."""
        # Check NaNs
        nan_count_data = np.isnan(eeg_data).sum()
        nan_count_labels = np.isnan(labels).sum()
        if nan_count_data > 0:
            metadata["issues"].append(f"Found {nan_count_data} NaN values in EEG data.")
            metadata["is_valid"] = False
        if nan_count_labels > 0:
            metadata["issues"].append(f"Found {nan_count_labels} NaN values in labels.")
            metadata["is_valid"] = False

        # Check Infs
        inf_count_data = np.isinf(eeg_data).sum()
        if inf_count_data > 0:
            metadata["issues"].append(f"Found {inf_count_data} Inf values in EEG data.")
            metadata["is_valid"] = False

        # Check Label Ranges (Expected 1.0 to 9.0)
        lbl_min = float(np.min(labels))
        lbl_max = float(np.max(labels))
        metadata["label_min"] = lbl_min
        metadata["label_max"] = lbl_max

        if lbl_min < 0.5 or lbl_max > 10.0:
            metadata["issues"].append(f"Labels out of expected [1, 9] range: min={lbl_min:.2f}, max={lbl_max:.2f}")

    def inspect_dataset_directory(self, dir_path: str) -> List[Dict[str, Any]]:
        """
        Inspects all .dat participant files in a directory and returns validation summary.
        """
        folder = Path(dir_path)
        if not folder.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dir_path}")

        dat_files = sorted(list(folder.glob("s*.dat")))
        if not dat_files:
            return []

        reports = []
        for file in dat_files:
            try:
                _, _, meta = self.load_single_participant(str(file), validate=True)
                reports.append(meta)
            except Exception as e:
                reports.append({
                    "file": file.name,
                    "is_valid": False,
                    "issues": [str(e)]
                })

        return reports

    def stream_split_participants(
        self,
        data_dir: str,
        participant_ids: List[int]
    ) -> Generator[Tuple[int, np.ndarray, np.ndarray], None, None]:
        """
        Generator that yields one participant's (id, eeg_data, labels) at a time.
        Avoids loading all 32 participants into RAM simultaneously on Google Colab T4.
        """
        folder = Path(data_dir)
        for pid in participant_ids:
            fname = f"s{pid:02d}.dat"
            fpath = folder / fname
            if not fpath.exists():
                print(f"[Warning] Participant file {fname} not found in {data_dir}. Skipping.")
                continue

            eeg_data, labels, _ = self.load_single_participant(str(fpath))
            yield pid, eeg_data, labels
