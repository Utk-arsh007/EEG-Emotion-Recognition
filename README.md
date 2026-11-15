# EEG-Based Emotion Recognition Using a Hybrid CNN-BiLSTM Deep Learning Architecture

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow 2.12+](https://img.shields.io/badge/TensorFlow-2.12%2B-orange.svg)](https://www.tensorflow.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end, reproducible deep learning system for human affective state prediction from multi-channel electroencephalogram (EEG) signals using the **DEAP dataset**. The system formulates emotion recognition as a continuous **multi-output regression problem**, predicting four affective dimensions: **Valence, Arousal, Dominance, and Liking** (1.0–9.0 scale).

---

## Architecture Overview

The proposed model combines:
1. **Convolutional Neural Networks (CNN):** Extracts spatial hierarchies and localized frequency patterns across 32 EEG electrode channels.
2. **Bidirectional Long Short-Term Memory (BiLSTM):** Captures bidirectional temporal dependencies and sequential dynamics in brainwave activity.
3. **Temporal Bahdanau-Style Self-Attention:** Dynamically weights emotionally salient time intervals within each EEG epoch.
4. **Fully Connected Regression Head:** Jointly outputs continuous affective coordinates.

```text
       Input 32-Channel EEG Window [Batch, 256 TimeSteps, 32 Channels]
                                     |
                                     v
                 +---------------------------------------+
                 |  Phase 1: CNN Feature Extraction      |
                 |  - Conv1D (32 filters, k=5) + BN + MP |
                 |  - Conv1D (64 filters, k=3) + BN + MP |
                 +---------------------------------------+
                                     |
                                     v (Batch, 64 TimeSteps, 64 Features)
                 +---------------------------------------+
                 |  Phase 2: BiLSTM Temporal Learning    |
                 |  - BiLSTM Layer 1 (64 units x 2)      |
                 |  - BiLSTM Layer 2 (32 units x 2)      |
                 +---------------------------------------+
                                     |
                                     v (Batch, 64 TimeSteps, 64 Features)
                 +---------------------------------------+
                 |  Phase 3: Temporal Attention Layer    |
                 |  - Score: u_t = tanh(W * h_t + b)     |
                 |  - Weight: alpha_t = softmax(u_t^T v) |
                 |  - Context: c = sum(alpha_t * h_t)    |
                 +---------------------------------------+
                                     |
                                     v Context Vector [Batch, 64]
                 +---------------------------------------+
                 |  Phase 4: Dense Regression Head       |
                 |  - Dense (64, ReLU) + Dropout (0.3)   |
                 |  - Dense (32, ReLU) + Dropout (0.2)   |
                 |  - Dense (4, Linear)                  |
                 +---------------------------------------+
                                     |
                  +------------------+------------------+
                  |                  |                  |                  |
                  v                  v                  v                  v
               Valence            Arousal           Dominance            Liking
              [1.0 - 9.0]        [1.0 - 9.0]       [1.0 - 9.0]        [1.0 - 9.0]
```

---

## Key Features

- **Strict 32-Channel EEG Selection:** Excludes peripheral physiological channels (channels 32–39), keeping only the 32 10–20 standard EEG electrode channels (`Fp1`, `AF3`, `F3`, `F7`, `FC5`, ..., `O2`).
- **Zero Data Leakage:** Participant-independent split ($s01–s22$ Train [70%], $s23–s27$ Val [15%], $s28–s32$ Test [15%]). All normalizers and standardizers are fitted strictly on the training participant cohort.
- **Robust Signal Preprocessing:**
  - 3-second pre-trial baseline extraction and mean subtraction.
  - Zero-phase 4th order Butterworth bandpass filtering (4.0–45.0 Hz), preserving Theta (4–8 Hz), Alpha (8–13 Hz), Beta (14–30 Hz), and Gamma (30–45 Hz) rhythms.
  - Sliding temporal windowing (2.0s window, 1.0s stride = 50% overlap).
- **Multi-Model Benchmark & Ablation Study:**
  - **Model A:** 1D Convolutional Neural Network (CNN).
  - **Model B:** Bidirectional LSTM (BiLSTM).
  - **Model C (Proposed):** Hybrid CNN-BiLSTM with Attention.
  - **Ablation Variant:** Hybrid CNN-BiLSTM without Attention (Global Average Pooling).
- **FastAPI + Modern Web Demonstration UI:**
  - Real-time prediction dashboard with interactive 2D Russell's Circumplex plot.
  - Preloaded sample trial loader and custom `.dat`/`.npy` file drag-and-drop.
  - Ethical and academic non-diagnostic disclaimer.

---

## Directory Structure

```text
EEG-Emotion-Recognition/
├── configs/
│   └── config.yaml                     # Centralized hyperparameters & paths
├── notebooks/
│   ├── 01_dataset_inspection.ipynb     # DEAP dataset inspection & channel validation
│   ├── 02_preprocessing.ipynb          # Filtering, baseline handling & zero-leakage split
│   ├── 03_cnn_model.ipynb              # Model A (CNN) training on Colab T4
│   ├── 04_bilstm_model.ipynb           # Model B (BiLSTM) training on Colab T4
│   ├── 05_hybrid_model.ipynb           # Model C (Proposed Hybrid Attention) training
│   └── 06_evaluation.ipynb             # Multi-metric comparative evaluation & plots
├── src/
│   ├── __init__.py
│   ├── config.py                       # Dataclass configuration engine
│   ├── data_loader.py                  # DEAP .dat reader, channel slicer & integrity checker
│   ├── preprocessing.py                # Butterworth filter & zero-leakage standardizer
│   ├── windowing.py                    # Temporal window segmenter & split builder
│   ├── models/
│   │   ├── __init__.py                 # Unified model factory (Keras & PyTorch)
│   │   ├── attention.py                # Bahdanau-style temporal self-attention layer
│   │   ├── cnn.py                      # Model A: 1D CNN implementation
│   │   ├── bilstm.py                   # Model B: BiLSTM implementation
│   │   └── hybrid_model.py             # Model C: Hybrid CNN-BiLSTM-Attention & Ablation
│   ├── train.py                        # Modular training pipeline with EarlyStopping & LR decay
│   ├── evaluate.py                     # Metric calculation (MAE, MSE, RMSE, R², Pearson r)
│   ├── predict.py                      # End-to-end single trial predictor & Circumplex mapper
│   └── utils.py                        # Vector SVG and Matplotlib visualization utilities
├── backend/
│   ├── main.py                         # FastAPI web server & prediction endpoints
│   ├── schemas.py                      # Pydantic request/response schemas
│   └── requirements.txt
├── frontend/
│   ├── index.html                      # Modern scientific user interface
│   ├── style.css                       # Dark-mode styling, responsive grid & animations
│   └── script.js                       # Circumplex 2D canvas & API communication
├── sample_data/
│   ├── generate_sample_eeg.py          # Synthetic DEAP-compliant participant generator
│   └── README.md
├── models/
│   └── checkpoints/                    # Saved weights (.pt / .keras)
├── results/
│   ├── figures/                        # Generated vector curves, scatter & bar charts
│   └── metrics/                        # CSV & JSON evaluation tables
├── reports/
│   ├── project_report.md               # 21-section comprehensive academic project report
│   └── figures/
├── presentation/
│   └── presentation_slides.md          # 15-slide viva defense presentation deck
├── tests/
│   ├── test_data_loader.py             # Channel slicing & split integrity tests
│   ├── test_preprocessing.py           # Filter attenuation & leakage verification
│   ├── test_models.py                  # Forward/backward gradient flow tests
│   └── test_api.py                     # FastAPI endpoint validation tests
├── colab_train.py                      # One-command Colab T4 GPU automated benchmark runner
├── requirements.txt
└── README.md
```

---

## Quickstart & Execution Guide

### 1. Local Environment Setup

Clone the repository and install dependencies:
```bash
git clone https://github.com/your-username/EEG-Emotion-Recognition.git
cd EEG-Emotion-Recognition

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Unit Test Suite
To verify dataset loading, channel slicing, zero-leakage standardizer, and model architectures:
```bash
python -m unittest discover tests
```
*All 17 tests verify data integrity, filter attenuation, tensor shapes, gradient flow, and API endpoints.*

### 3. Generate Sample Dataset (Offline Development)
To test and demonstrate the pipeline without downloading the full 3 GB DEAP dataset:
```bash
python sample_data/generate_sample_eeg.py --output_dir ./sample_data --trials 5 --participants 1 2 23 28
```

### 4. Build Preprocessed Splits
```bash
python -c "
from src.windowing import build_split_datasets
from src.config import CONFIG
build_split_datasets(data_dir='./sample_data', output_dir='./data/processed', config=CONFIG, save_to_disk=True)
"
```

### 5. Launch the Demonstration Web Interface
```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your web browser. You can select preloaded sample trials or upload custom `.dat` files, visualize the 4 continuous ratings, and view the real-time position on Russell's 2D Circumplex affective plot.

---

## Google Colab T4 GPU Training Guide

For full training on the 32-participant DEAP dataset:

1. **Upload Dataset to Google Drive:**
   Place your DEAP preprocessed Python folder (`data_preprocessed_python`) containing `s01.dat` to `s32.dat` into:
   ```text
   /content/drive/MyDrive/DEAP/data_preprocessed_python/
   ```

2. **Open Google Colab:**
   - Select **Runtime > Change runtime type > T4 GPU**.
   - Mount Google Drive:
     ```python
     from google.colab import drive
     drive.mount('/content/drive')
     ```

3. **Clone & Configure Project:**
   ```bash
   !git clone https://github.com/your-username/EEG-Emotion-Recognition.git
   %cd EEG-Emotion-Recognition
   !pip install -r requirements.txt
   ```

4. **Run Preprocessing Pipeline (Notebook 02 or Command Line):**
   ```python
   from src.windowing import build_split_datasets
   from src.config import CONFIG

   build_split_datasets(
       data_dir='/content/drive/MyDrive/DEAP/data_preprocessed_python',
       output_dir='/content/drive/MyDrive/DEAP/processed',
       config=CONFIG,
       save_to_disk=True
   )
   ```

5. **Train All Models with One Command:**
   ```bash
   !python colab_train.py --data_dir /content/drive/MyDrive/DEAP/processed --backend keras --epochs 40 --batch_size 64
   ```

---

## Experimental Benchmark (Pipeline Verification Run)

> **Metric Provenance Disclosure:**  
> The metrics reported in the tables below were obtained during the **offline software verification run** using synthetic DEAP-conforming signals across the deterministic participant split ($s01–s22$ Train, $s23–s27$ Val, $s28–s32$ Test). This test empirically verifies that gradient backpropagation, loss functions, learning rate scheduling, and regression metric computations execute correctly without errors.
>
> When you run the training pipeline with your actual 32-participant DEAP dataset on Google Colab T4 GPU, the script will automatically evaluate the model on the real test participants and overwrite the metric files in `results/metrics/`.

| Model Architecture | MAE (Mean) | MSE (Mean) | RMSE (Mean) | Overall Pearson $r$ | Parameters | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model A: 1D CNN** | 1.2237 | 2.4306 | 1.4723 | -0.0228 | ~128,452 | Verified |
| **Model B: BiLSTM** | **1.0528** | **1.7904** | **1.3257** | -0.0376 | ~184,836 | Verified |
| **Ablation: Hybrid (No Attention)** | 1.1143 | 2.5148 | 1.4965 | +0.0515 | ~142,308 | Verified |
| **Model C: Proposed Hybrid CNN-BiLSTM-Attention** | 1.1766 | 2.1939 | 1.4757 | **+0.0206** | ~146,436 | Verified |

### Per-Dimension Verification Results (Proposed Hybrid Model)

| Affective Dimension | MAE | MSE | RMSE | Pearson $r$ | $p$-value |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Valence** | 0.8628 | 1.8734 | 1.3687 | +0.0303 | 0.644 |
| **Arousal** | 1.1401 | 1.7928 | 1.3389 | +0.1810 | 0.0053 |
| **Dominance** | 1.2535 | 2.7145 | 1.6476 | -0.1122 | 0.0854 |
| **Liking** | 1.4499 | 2.3949 | 1.5475 | -0.0168 | 0.797 |
| **Average** | **1.1766** | **2.1939** | **1.4757** | **+0.0206** | — |

*Theoretical Note:* In empirical neurophysiological literature (e.g. Koelstra et al., 2012; Zheng & Lu, 2015), **Arousal** typically achieves higher classification and correlation scores compared to Valence due to strong beta-band (14–30 Hz) power modulation and autonomic activation across central and parietal electrodes. Valence decoding relies on subtler frontal hemispheric asymmetry (FAA) which exhibits greater inter-subject variance.

---

## Viva Voce Defense Preparation (Q&A)

### Q1: Why formulate emotion recognition as regression instead of binary classification?
**Answer:** Real-world emotions are continuous, non-discrete, and dynamic. Discretizing continuous ratings (e.g. $> 5.0$ as High, $\le 5.0$ as Low) discards critical gradations (a rating of 5.1 is treated identically to 9.0 and oppositely to 4.9) and introduces arbitrary threshold bias. Regression directly preserves the nuanced 1.0–9.0 affective continuum.

### Q2: Why use 32 channels instead of all 40 channels in DEAP?
**Answer:** In the DEAP dataset, channels 0–31 represent genuine EEG electrode sites adhering to the International 10–20 montage, whereas channels 32–39 are peripheral physiological modalities (GSR, respiration, plethysmograph, skin temperature). Mixing peripheral modalities with EEG confuses spatial cortical feature extractors due to disparate physical dimensions, units, and latencies.

### Q3: How do you guarantee zero data leakage?
**Answer:** We enforce strict participant-level splitting. Participants $s01–s22$ constitute training, $s23–s27$ validation, and $s28–s32$ testing. Normalization scalers (`ZeroLeakageStandardizer`) calculate mean and standard deviation exclusively from training subjects and transform test subjects post-hoc. Overlapping temporal windows never span across train and test sets.

### Q4: What is the mathematical contribution of the Attention mechanism?
**Answer:** Rather than flattening or simple average pooling over the BiLSTM temporal representations $H = [h_1, \dots, h_T]$, the Bahdanau self-attention layer learns an alignment vector:
$$u_t = \tanh(W h_t + b), \quad \alpha_t = \frac{\exp(u_t^\top v)}{\sum_k \exp(u_k^\top v)}, \quad c = \sum_{t=1}^T \alpha_t h_t$$
This allows the network to assign higher weights $\alpha_t$ to emotionally salient brainwave bursts (e.g., sudden onset of musical dissonance or crescendos) while ignoring baseline noise.

---

## License & Citation

This project is licensed under the MIT License. Academic works referencing this implementation should cite:
- **DEAP Dataset:** Koelstra et al., "DEAP: A Database for Emotion Analysis using Physiological Signals", *IEEE Transactions on Affective Computing*, 2012.
- **Russell Circumplex Model:** Russell, J. A., "A circumplex model of affect", *Journal of Personality and Social Psychology*, 1980.
