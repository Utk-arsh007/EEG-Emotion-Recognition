# Sample EEG Data Directory

This folder contains synthetic DEAP-formatted `.dat` files for offline verification, unit testing, and FastAPI demonstration without requiring the full 3 GB DEAP dataset download.

### File Format:
- Python pickle file (`.dat`)
- Dictionary keys:
  - `"data"`: `numpy.ndarray` with shape `(trials, 40, 8064)`
  - `"labels"`: `numpy.ndarray` with shape `(trials, 4)` (Valence, Arousal, Dominance, Liking)

To regenerate or create more sample files:
```bash
python sample_data/generate_sample_eeg.py --output_dir ./sample_data --trials 10 --participants 1 2 23 28
```
