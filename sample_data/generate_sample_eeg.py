"""
Synthetic Sample EEG Generator for DEAP Format
Generates realistic, physically plausible multi-channel EEG signals
matching the exact DEAP preprocessed Python (.dat) format:
  - Dictionary with keys 'data' and 'labels'
  - data: shape (trials, 40, 8064), float32/float64
  - labels: shape (trials, 4), float32/float64 (Valence, Arousal, Dominance, Liking in [1, 9])
"""

import os
import pickle
import numpy as np
from pathlib import Path


def generate_eeg_signal(
    duration_sec: float = 63.0,
    sampling_rate: int = 128,
    num_channels: int = 40,
    seed: int = 42
) -> np.ndarray:
    """
    Generate synthetic multi-channel EEG signals composed of:
    - Theta waves (4-8 Hz)
    - Alpha waves (8-13 Hz)
    - Beta waves (14-30 Hz)
    - Pink/1/f noise + Gaussian sensor noise
    """
    rng = np.random.default_rng(seed)
    total_samples = int(duration_sec * sampling_rate)
    t = np.linspace(0, duration_sec, total_samples, endpoint=False)
    
    signals = np.zeros((num_channels, total_samples), dtype=np.float32)
    
    for ch in range(num_channels):
        # 1/f baseline noise
        white_noise = rng.standard_normal(total_samples)
        pink_filter = np.cumsum(white_noise) * 0.05
        pink_filter -= np.mean(pink_filter)
        
        # Theta band oscillation (4-8 Hz)
        theta_freq = rng.uniform(4.0, 7.5)
        theta_amp = rng.uniform(1.0, 3.0)
        theta = theta_amp * np.sin(2 * np.pi * theta_freq * t + rng.uniform(0, 2*np.pi))
        
        # Alpha band oscillation (8-13 Hz) - prominent in parietal/occipital
        alpha_freq = rng.uniform(8.5, 12.5)
        alpha_amp = rng.uniform(2.0, 6.0) if ch in [10, 11, 12, 13, 14, 15, 28, 29, 30, 31] else rng.uniform(0.5, 2.5)
        alpha = alpha_amp * np.sin(2 * np.pi * alpha_freq * t + rng.uniform(0, 2*np.pi))
        
        # Beta band oscillation (14-30 Hz) - active thinking / arousal
        beta_freq = rng.uniform(15.0, 25.0)
        beta_amp = rng.uniform(0.5, 2.5)
        beta = beta_amp * np.sin(2 * np.pi * beta_freq * t + rng.uniform(0, 2*np.pi))
        
        # Sensor noise
        sensor_noise = rng.normal(0, 0.8, total_samples)
        
        signals[ch] = pink_filter + theta + alpha + beta + sensor_noise
        
    return signals


def generate_participant_dat(
    output_path: str,
    num_trials: int = 40,
    seed: int = 42,
    include_peripheral: bool = True
) -> None:
    """Generate and save a single participant .dat file conforming to DEAP specifications."""
    rng = np.random.default_rng(seed)
    num_channels = 40 if include_peripheral else 32
    samples_per_trial = 8064
    
    data = np.zeros((num_trials, num_channels, samples_per_trial), dtype=np.float32)
    labels = np.zeros((num_trials, 4), dtype=np.float32)
    
    for trial_idx in range(num_trials):
        trial_seed = seed * 1000 + trial_idx
        data[trial_idx] = generate_eeg_signal(
            duration_sec=63.0,
            sampling_rate=128,
            num_channels=num_channels,
            seed=trial_seed
        )
        
        # Realistic continuous labels distributed around mean ~5.2, std ~1.8, bounded in [1.0, 9.0]
        # Dimensions: Valence, Arousal, Dominance, Liking
        raw_vals = rng.normal(loc=5.3, scale=1.7, size=4)
        labels[trial_idx] = np.clip(raw_vals, 1.0, 9.0)
        
    out_dict = {
        "data": data,
        "labels": labels
    }
    
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(out_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    print(f"[OK] Generated {output_path} | data shape: {data.shape} | labels shape: {labels.shape}")


def create_sample_dataset(
    output_dir: str = "./sample_data",
    participant_ids: list = None,
    trials_per_participant: int = 5
) -> None:
    """Create sample DEAP files for local debugging and demonstration."""
    if participant_ids is None:
        participant_ids = [1, 28]  # One training (s01), one testing (s28)
        
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for pid in participant_ids:
        filename = f"s{pid:02d}.dat"
        filepath = out_dir / filename
        generate_participant_dat(
            str(filepath),
            num_trials=trials_per_participant,
            seed=pid
        )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate synthetic DEAP .dat files")
    parser.add_argument("--output_dir", type=str, default="./sample_data")
    parser.add_argument("--trials", type=int, default=10, help="Number of trials per participant (default 10 for quick tests, 40 for full DEAP spec)")
    parser.add_argument("--participants", type=int, nargs="+", default=[1, 2, 23, 28], help="Participant IDs to generate")
    args = parser.parse_args()
    
    create_sample_dataset(
        output_dir=args.output_dir,
        participant_ids=args.participants,
        trials_per_participant=args.trials
    )
