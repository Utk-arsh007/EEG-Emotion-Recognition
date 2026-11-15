"""
Model Evaluation and Metrics Module
Calculates MAE, MSE, RMSE, R^2, and Pearson Correlation Coefficient
for all 4 continuous affective dimensions (Valence, Arousal, Dominance, Liking)
on the independent test participants.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import CONFIG
from src.utils import plot_actual_vs_predicted, plot_error_distributions


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Computes regression metrics per dimension and overall averages.
    
    Args:
        y_true: shape (N, 4)
        y_pred: shape (N, 4)
        target_names: list of target names
        
    Returns:
        dictionary with per-target metrics and overall summary
    """
    targets = target_names or CONFIG.dataset.target_names
    num_targets = y_true.shape[1]

    per_target = {}
    mae_list, mse_list, rmse_list, r2_list, pearson_list = [], [], [], [], []

    for i in range(num_targets):
        t_name = targets[i]
        yt = y_true[:, i]
        yp = y_pred[:, i]

        mae = float(mean_absolute_error(yt, yp))
        mse = float(mean_squared_error(yt, yp))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(yt, yp))

        # Pearson correlation
        if np.std(yt) > 1e-6 and np.std(yp) > 1e-6:
            r, p_val = stats.pearsonr(yt, yp)
            r = float(r)
            p_val = float(p_val)
        else:
            r, p_val = 0.0, 1.0

        per_target[t_name] = {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2,
            "Pearson_r": r,
            "p_value": p_val
        }

        mae_list.append(mae)
        mse_list.append(mse)
        rmse_list.append(rmse)
        r2_list.append(r2)
        pearson_list.append(r)

    overall = {
        "MAE_mean": float(np.mean(mae_list)),
        "MSE_mean": float(np.mean(mse_list)),
        "RMSE_mean": float(np.mean(rmse_list)),
        "R2_mean": float(np.mean(r2_list)),
        "Pearson_r_mean": float(np.mean(pearson_list))
    }

    return {
        "per_target": per_target,
        "overall": overall
    }


def format_metrics_table(metrics_dict: Dict[str, Any]) -> pd.DataFrame:
    """Formats metrics dictionary into a clean pandas DataFrame."""
    rows = []
    per_target = metrics_dict["per_target"]
    for t_name, m in per_target.items():
        rows.append({
            "Affective Dimension": t_name,
            "MAE": round(m["MAE"], 4),
            "MSE": round(m["MSE"], 4),
            "RMSE": round(m["RMSE"], 4),
            "R² Score": round(m["R2"], 4),
            "Pearson r": round(m["Pearson_r"], 4),
            "p-value": f"{m['p_value']:.2e}"
        })

    overall = metrics_dict["overall"]
    rows.append({
        "Affective Dimension": "OVERALL MEAN",
        "MAE": round(overall["MAE_mean"], 4),
        "MSE": round(overall["MSE_mean"], 4),
        "RMSE": round(overall["RMSE_mean"], 4),
        "R² Score": round(overall["R2_mean"], 4),
        "Pearson r": round(overall["Pearson_r_mean"], 4),
        "p-value": "N/A"
    })

    return pd.DataFrame(rows)


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Hybrid_Attention",
    backend: str = "torch",
    output_dir: str = "./results"
) -> Dict[str, Any]:
    """
    Runs full evaluation on test data, generates plots, and saves tables.
    """
    out_path = Path(output_dir)
    fig_dir = out_path / "figures"
    metrics_dir = out_path / "metrics"
    fig_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Evaluating Model '{model_name}' on Test Cohort (N={len(X_test)}) ===")

    # Predict
    if backend in ["torch", "pytorch"]:
        import torch
        model.eval()
        with torch.no_grad():
            tensor_X = torch.from_numpy(X_test).float()
            if hasattr(model, "forward") and "return_attention" in model.forward.__code__.co_varnames:
                preds = model(tensor_X, return_attention=False)
            else:
                preds = model(tensor_X)
            y_pred = preds.cpu().numpy()
    else:
        y_pred = model.predict(X_test, batch_size=64)

    # Calculate metrics
    results = compute_metrics(y_test, y_pred)
    df_table = format_metrics_table(results)

    print("\n--- Model Evaluation Results ---")
    print(df_table.to_string(index=False))

    # Save CSV and JSON
    csv_file = metrics_dir / f"{model_name.lower()}_test_metrics.csv"
    json_file = metrics_dir / f"{model_name.lower()}_test_metrics.json"
    df_table.to_csv(csv_file, index=False)
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Generate Actual vs Predicted and Error Distribution Plots
    scatter_path = str(fig_dir / f"{model_name.lower()}_actual_vs_predicted.png")
    err_path = str(fig_dir / f"{model_name.lower()}_error_distribution.png")
    plot_actual_vs_predicted(y_test, y_pred, save_path=scatter_path)
    plot_error_distributions(y_test, y_pred, save_path=err_path)

    print(f"[OK] Saved metrics to {csv_file}")
    print(f"[OK] Saved figures to {scatter_path} and {err_path}")

    return {
        "metrics": results,
        "dataframe": df_table,
        "y_pred": y_pred
    }


if __name__ == "__main__":
    import argparse
    from src.models import get_model

    parser = argparse.ArgumentParser(description="Evaluate EEG Emotion Recognition Model")
    parser.add_argument("--test_data", type=str, default="./data/processed/test_data.npz")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="hybrid")
    parser.add_argument("--backend", type=str, default="torch")
    parser.add_argument("--output_dir", type=str, default="./results")
    args = parser.parse_args()

    if os.path.exists(args.test_data):
        td = np.load(args.test_data)
        X_test, y_test = td["X"], td["y"]
        print(f"Loaded test dataset from {args.test_data}: X={X_test.shape}, y={y_test.shape}")

        timesteps, channels = X_test.shape[1], X_test.shape[2]
        if args.backend in ["torch", "pytorch"]:
            import torch
            model = get_model(args.model_name, backend="torch", in_channels=channels, timesteps=timesteps)
            ckpt = args.model_path or f"./models/checkpoints/best_{args.model_name.lower()}_torch.pt"
            if os.path.exists(ckpt):
                model.load_state_dict(torch.load(ckpt, map_location="cpu"))
                print(f"[OK] Loaded PyTorch checkpoint from {ckpt}")
            else:
                print(f"[Warning] Checkpoint {ckpt} not found. Running with initialized weights.")
        else:
            import tensorflow as tf
            from src.models.attention import KerasTemporalAttention
            ckpt = args.model_path or f"./models/checkpoints/best_{args.model_name.lower()}_keras.keras"
            if os.path.exists(ckpt):
                model = tf.keras.models.load_model(ckpt, custom_objects={"KerasTemporalAttention": KerasTemporalAttention}, compile=False)
                print(f"[OK] Loaded Keras model from {ckpt}")
            else:
                model = get_model(args.model_name, backend="keras", in_channels=channels, timesteps=timesteps)

        evaluate_model(
            model=model,
            X_test=X_test,
            y_test=y_test,
            model_name=args.model_name,
            backend=args.backend,
            output_dir=args.output_dir
        )
    else:
        print(f"[Warning] Test data {args.test_data} not found. Exiting.")
