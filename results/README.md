# Results & Metrics Directory

This directory stores experimental results, benchmark tables, and visualization graphics.

> **Metric Provenance & Colab Execution Note:**  
> The initial CSV/JSON metrics and SVG plots currently stored in this directory were generated during the **offline pipeline verification run** on synthetic DEAP-conforming signals to validate backpropagation, windowing, and metric computations across independent test subjects ($s28–s32$).
>
> When you train on your actual 32-participant DEAP dataset using the provided Google Colab script (`colab_train.py`), the script will evaluate the trained models on the real human test subjects and automatically overwrite these files with the final DEAP benchmark metrics.

### Contents:
- `metrics/`:
  - `hybrid_test_metrics.csv` & `.json`: Performance metrics (MAE, MSE, RMSE, R², Pearson r) for the proposed Hybrid Attention model on the independent test cohort ($s28–s32$).
  - `cnn_test_metrics.csv`: Performance metrics for Model A (CNN).
  - `bilstm_test_metrics.csv`: Performance metrics for Model B (BiLSTM).
  - `ablation_test_metrics.csv`: Performance metrics for the ablation model (Hybrid without Attention).
  - `overall_model_comparison.json`: Cross-architecture summary table.
- `figures/`:
  - `cross_model_comparison.svg`: Grouped bar chart comparing MAE and RMSE across all 4 architectures.
  - `hybrid_actual_vs_predicted.svg`: 4-quadrant scatter plots comparing ground-truth ratings with predicted ratings against ideal $y=x$ lines.
  - `hybrid_error_distribution.svg`: Residual error histograms for Valence, Arousal, Dominance, and Liking.
  - `hybrid_training_curves.svg`: Training and validation loss curves across epochs.
