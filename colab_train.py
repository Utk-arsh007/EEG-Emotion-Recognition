"""
Google Colab Automated Multi-Model Training & Benchmark Runner
Runs complete training and comparative benchmarking on Google Colab T4 GPU
for all 4 model variants:
  1. Model A: CNN
  2. Model B: BiLSTM
  3. Model C: Proposed Hybrid CNN-BiLSTM-Attention
  4. Ablation: Hybrid without Attention

Usage in Google Colab:
  !python colab_train.py --data_dir /content/drive/MyDrive/DEAP/processed --backend keras --epochs 40
"""

import os
import sys
import argparse
from pathlib import Path
import json
import numpy as np

from src.train import run_training_pipeline
from src.utils import plot_model_comparison


def main():
    parser = argparse.ArgumentParser(description="Colab Benchmark Runner")
    parser.add_argument("--data_dir", type=str, default="./data/processed")
    parser.add_argument("--backend", type=str, default="keras", choices=["keras", "torch"])
    parser.add_argument("--model", type=str, default="all", choices=["all", "cnn", "bilstm", "hybrid", "ablation"])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--checkpoint_dir", type=str, default="./models/checkpoints")
    parser.add_argument("--output_dir", type=str, default="./results")
    args = parser.parse_args()

    models_to_run = ["cnn", "bilstm", "hybrid", "ablation"] if args.model == "all" else [args.model]
    comparison_summary = {}

    print("==================================================================")
    print("      EEG EMOTION RECOGNITION - ARCHITECTURAL BENCHMARK           ")
    print(f" Models: {models_to_run} | Backend: {args.backend} | Epochs: {args.epochs}")
    print("==================================================================")

    for m_name in models_to_run:
        print(f"\n>>> Running pipeline for: {m_name.upper()} <<<")
        try:
            res = run_training_pipeline(
                model_name=m_name,
                backend=args.backend,
                data_dir=args.data_dir,
                results_dir=args.output_dir,
                checkpoint_dir=args.checkpoint_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr
            )
            overall = res["evaluation"]["metrics"]["overall"]
            comparison_summary[m_name.upper()] = {
                "MAE": round(overall["MAE_mean"], 4),
                "RMSE": round(overall["RMSE_mean"], 4),
                "R2": round(overall["R2_mean"], 4),
                "Pearson_r": round(overall["Pearson_r_mean"], 4)
            }
        except Exception as e:
            print(f"[Error] Failed training {m_name}: {e}")

    # Generate comparison plot and summary table
    if comparison_summary:
        fig_path = os.path.join(args.output_dir, "figures", "cross_model_comparison.png")
        plot_model_comparison(comparison_summary, save_path=fig_path)
        print(f"\n[OK] Comparative benchmark chart saved to {fig_path}")

        summary_file = os.path.join(args.output_dir, "metrics", "overall_model_comparison.json")
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(comparison_summary, f, indent=2)
        print(f"[OK] Comparison summary saved to {summary_file}")


if __name__ == "__main__":
    main()
