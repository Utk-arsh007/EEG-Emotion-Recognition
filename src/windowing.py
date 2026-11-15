"""
EEG Windowing and Segmentation Module
Divides continuous 60-second trial signals into overlapping temporal windows,
preserves participant and trial metadata, and creates model-ready datasets.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Generator
import numpy as np

from src.config import CONFIG, AppConfig
from src.preprocessing import EEGPreprocessor, ZeroLeakageStandardizer
from src.data_loader import DEAPDataLoader


class EEGWindowSegmenter:
    """
    Segments EEG trials into time-series windows with configurable size and overlap.
    Standard output shape per window: (window_samples, 32 channels) -> (timesteps, channels).
    """

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or CONFIG
        self.window_samples = self.config.preprocessing.window_samples
        self.stride_samples = self.config.preprocessing.stride_samples
        self.fs = self.config.dataset.sampling_rate

    def segment_trial(
        self,
        trial_eeg: np.ndarray,
        trial_labels: np.ndarray,
        participant_id: int,
        trial_index: int
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, int]]]:
        """
        Segments a single preprocessed EEG trial.
        
        Args:
            trial_eeg: shape (32, num_samples), e.g. (32, 7680)
            trial_labels: shape (4,) -> [Valence, Arousal, Dominance, Liking]
            participant_id: int (e.g. 1 to 32)
            trial_index: int (e.g. 0 to 39)
            
        Returns:
            windows: shape (num_windows, window_samples, 32) -> (W, 256, 32)
            window_labels: shape (num_windows, 4)
            metadata: list of dicts with participant, trial, and window indices
        """
        num_channels, total_samples = trial_eeg.shape
        if total_samples < self.window_samples:
            raise ValueError(f"Trial length {total_samples} is shorter than window length {self.window_samples}")

        # Compute number of sliding windows
        num_windows = (total_samples - self.window_samples) // self.stride_samples + 1
        
        windows = np.zeros((num_windows, self.window_samples, num_channels), dtype=np.float32)
        window_labels = np.zeros((num_windows, 4), dtype=np.float32)
        metadata = []

        for w_idx in range(num_windows):
            start = w_idx * self.stride_samples
            end = start + self.window_samples
            
            # Slice: (32, window_samples) -> transpose to (window_samples, 32) for Conv1D / LSTM
            window_slice = trial_eeg[:, start:end].T
            windows[w_idx] = window_slice
            window_labels[w_idx] = trial_labels
            
            metadata.append({
                "participant_id": participant_id,
                "trial_index": trial_index,
                "window_index": w_idx,
                "start_sample": start,
                "end_sample": end
            })

        return windows, window_labels, metadata

    def process_participant(
        self,
        eeg_trials: np.ndarray,
        labels: np.ndarray,
        participant_id: int,
        preprocessor: Optional[EEGPreprocessor] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, int]]]:
        """
        Preprocesses and segments all trials for one participant.
        
        Args:
            eeg_trials: shape (num_trials, 32, 8064)
            labels: shape (num_trials, 4)
            participant_id: int
            preprocessor: EEGPreprocessor instance
            
        Returns:
            all_windows: shape (total_windows, window_samples, 32)
            all_labels: shape (total_windows, 4)
            all_metadata: list of metadata dicts
        """
        prep = preprocessor or EEGPreprocessor(self.config)
        num_trials = eeg_trials.shape[0]

        window_list = []
        label_list = []
        meta_list = []

        for t_idx in range(num_trials):
            raw_trial = eeg_trials[t_idx]  # (32, 8064)
            # Step 1: Preprocess (baseline removal + bandpass filter + clip)
            clean_trial = prep.preprocess_trial(raw_trial)  # (32, 7680)
            
            # Step 2: Segment into windows
            wins, w_lbls, metas = self.segment_trial(
                clean_trial,
                labels[t_idx],
                participant_id=participant_id,
                trial_index=t_idx
            )
            
            window_list.append(wins)
            label_list.append(w_lbls)
            meta_list.extend(metas)

        if not window_list:
            return (
                np.empty((0, self.window_samples, 32), dtype=np.float32),
                np.empty((0, 4), dtype=np.float32),
                []
            )

        return (
            np.concatenate(window_list, axis=0),
            np.concatenate(label_list, axis=0),
            meta_list
        )


def build_split_datasets(
    data_dir: str,
    output_dir: str,
    config: Optional[AppConfig] = None,
    save_to_disk: bool = True,
    compressed: bool = False
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Builds model-ready datasets for Train, Validation, and Test splits
    with strictly zero data leakage:
      1. Loads each participant independently.
      2. Preprocesses and windows signals.
      3. Fits ZeroLeakageStandardizer ONLY on train windows.
      4. Standardizes val and test windows using the train standardizer.
      5. Saves .npz archives and standardizer to disk (uncompressed by default for fast, low-RAM saving).
      
    Returns:
        dict with keys: 'train', 'val', 'test' -> (X, y)
    """
    cfg = config or CONFIG
    loader = DEAPDataLoader(cfg)
    preprocessor = EEGPreprocessor(cfg)
    segmenter = EEGWindowSegmenter(cfg)

    train_pids = cfg.data_split.train_participants
    val_pids = cfg.data_split.val_participants
    test_pids = cfg.data_split.test_participants

    print(f"=== Building Split Datasets (Strict Participant Isolation) ===")
    print(f"Train Participants ({len(train_pids)}): {train_pids}")
    print(f"Val Participants   ({len(val_pids)}): {val_pids}")
    print(f"Test Participants  ({len(test_pids)}): {test_pids}")

    def collect_split(pids: List[int], name: str):
        X_parts, y_parts = [], []
        for pid in pids:
            possible_fnames = [f"s{pid:02d}.dat", f"s{pid}.dat", f"S{pid:02d}.dat", f"S{pid}.dat"]
            fpath = None
            for cand in possible_fnames:
                cand_path = Path(data_dir) / cand
                if cand_path.exists():
                    fpath = cand_path
                    break
            if fpath is None:
                continue
            eeg, lbls, _ = loader.load_single_participant(str(fpath))
            wins, w_lbls, _ = segmenter.process_participant(eeg, lbls, pid, preprocessor)
            X_parts.append(wins)
            y_parts.append(w_lbls)
            print(f"  [{name}] Processed participant {pid:02d} ({fpath.name}): {len(wins)} windows", flush=True)

        if not X_parts:
            return np.empty((0, cfg.preprocessing.window_samples, 32), dtype=np.float32), np.empty((0, 4), dtype=np.float32)
        return np.concatenate(X_parts, axis=0), np.concatenate(y_parts, axis=0)

    # 1. Process Train split
    print("\n[1/3] Processing Training participants...")
    X_train_raw, y_train = collect_split(train_pids, "Train")
    print(f"  -> Raw Train shape: X={X_train_raw.shape}, y={y_train.shape}")

    # 2. Process Validation split
    print("\n[2/3] Processing Validation participants...")
    X_val_raw, y_val = collect_split(val_pids, "Val")
    print(f"  -> Raw Val shape:   X={X_val_raw.shape}, y={y_val.shape}")

    # 3. Process Test split
    print("\n[3/3] Processing Test participants...")
    X_test_raw, y_test = collect_split(test_pids, "Test")
    print(f"  -> Raw Test shape:  X={X_test_raw.shape}, y={y_test.shape}")

    # 4. Zero Data Leakage Normalization
    print("\n[4/4] Applying Zero-Leakage Normalization (Fit on Train ONLY)...")
    scaler = ZeroLeakageStandardizer(per_channel=True)
    if len(X_train_raw) > 0:
        X_train = scaler.fit_transform(X_train_raw)
        X_val = scaler.transform(X_val_raw) if len(X_val_raw) > 0 else X_val_raw
        X_test = scaler.transform(X_test_raw) if len(X_test_raw) > 0 else X_test_raw
    else:
        X_train, X_val, X_test = X_train_raw, X_val_raw, X_test_raw

    out_dict = {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test)
    }

    if save_to_disk:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        scaler_file = str(out_p / "eeg_scaler.npz")
        if scaler.is_fitted:
            scaler.save(scaler_file)
            print(f"[OK] Saved fitted scaler to {scaler_file}")

        save_fn = np.savez_compressed if compressed else np.savez
        for split_name, (X_s, y_s) in out_dict.items():
            split_file = out_p / f"{split_name}_data.npz"
            save_fn(split_file, X=X_s, y=y_s)
            print(f"[OK] Saved {split_name} split to {split_file} (X={X_s.shape}, y={y_s.shape})")

    return out_dict
