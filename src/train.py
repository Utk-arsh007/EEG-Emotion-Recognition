"""
Unified Training Pipeline for EEG Emotion Recognition
Supports Model A (CNN), Model B (BiLSTM), Model C (Hybrid CNN-BiLSTM-Attention),
and Ablation Variant (Hybrid without Attention).
Dual-Backend Support:
  - 'keras' / 'tf' (Primary for Google Colab T4 GPU)
  - 'torch' (For local execution and testing)
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np

from src.config import CONFIG, AppConfig
from src.models import get_model
from src.utils import plot_training_curves
from src.evaluate import evaluate_model


# -------------------------------------------------------------------------
# PyTorch Training Engine
# -------------------------------------------------------------------------
def train_torch_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 40,
    batch_size: int = 64,
    lr: float = 0.001,
    patience: int = 8,
    checkpoint_dir: str = "./models/checkpoints",
    device_str: str = "auto"
) -> Tuple[Any, Dict[str, list], str]:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    print(f"[PyTorch Engine] Training on device: {device}")

    # Build model
    timesteps = X_train.shape[1]
    channels = X_train.shape[2]
    model = get_model(model_name, backend="torch", in_channels=channels, timesteps=timesteps)
    model.to(device)

    # Convert to DataLoaders
    train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    val_ds = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6)

    os.makedirs(checkpoint_dir, exist_ok=True)
    best_ckpt_path = os.path.join(checkpoint_dir, f"best_{model_name.lower()}_torch.pt")

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_mae": [],
        "val_mae": []
    }

    best_val_loss = float("inf")
    patience_counter = 0

    start_time = time.time()
    print(f"Starting training: {epochs} epochs | Batch size: {batch_size} | Model: {model_name}")

    for epoch in range(1, epochs + 1):
        # Training loop
        model.train()
        train_loss_accum = 0.0
        train_mae_accum = 0.0

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            
            if model_name in ["hybrid", "proposed"]:
                preds = model(bx, return_attention=False)
            else:
                preds = model(bx)

            loss = criterion(preds, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss_accum += loss.item() * bx.size(0)
            train_mae_accum += torch.sum(torch.abs(preds - by)).item() / 4.0

        epoch_train_loss = train_loss_accum / len(train_ds)
        epoch_train_mae = train_mae_accum / len(train_ds)

        # Validation loop
        model.eval()
        val_loss_accum = 0.0
        val_mae_accum = 0.0

        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                if model_name in ["hybrid", "proposed"]:
                    preds = model(bx, return_attention=False)
                else:
                    preds = model(bx)

                val_loss = criterion(preds, by)
                val_loss_accum += val_loss.item() * bx.size(0)
                val_mae_accum += torch.sum(torch.abs(preds - by)).item() / 4.0

        epoch_val_loss = val_loss_accum / len(val_ds)
        epoch_val_mae = val_mae_accum / len(val_ds)

        scheduler.step(epoch_val_loss)

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["train_mae"].append(epoch_train_mae)
        history["val_mae"].append(epoch_val_mae)

        print(f"Epoch {epoch:02d}/{epochs:02d} - Train Loss: {epoch_train_loss:.4f} (MAE: {epoch_train_mae:.4f}) | Val Loss: {epoch_val_loss:.4f} (MAE: {epoch_val_mae:.4f})")

        # Early stopping and checkpoint
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"  -> [Saved Checkpoint] Best val loss improved to {best_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch} (no improvement for {patience} epochs).")
                break

    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed:.1f}s. Restoring best checkpoint weights.")
    model.load_state_dict(torch.load(best_ckpt_path))
    model.cpu()

    return model, history, best_ckpt_path


# -------------------------------------------------------------------------
# TensorFlow / Keras Training Engine (Google Colab T4)
# -------------------------------------------------------------------------
def train_keras_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 0.001,
    patience: int = 10,
    checkpoint_dir: str = "./models/checkpoints"
) -> Tuple[Any, Dict[str, list], str]:
    import tensorflow as tf

    os.makedirs(checkpoint_dir, exist_ok=True)
    best_ckpt_path = os.path.join(checkpoint_dir, f"best_{model_name.lower()}_keras.keras")

    input_shape = (X_train.shape[1], X_train.shape[2])
    model = get_model(model_name, backend="keras", in_channels=input_shape[1], timesteps=input_shape[0])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.Huber(delta=1.0),
        metrics=["mae", tf.keras.metrics.RootMeanSquaredError(name="rmse")]
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6),
        tf.keras.callbacks.ModelCheckpoint(best_ckpt_path, monitor="val_loss", save_best_only=True)
    ]

    fit_hist = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    history = {
        "train_loss": [float(x) for x in fit_hist.history["loss"]],
        "val_loss": [float(x) for x in fit_hist.history["val_loss"]],
        "train_mae": [float(x) for x in fit_hist.history["mae"]],
        "val_mae": [float(x) for x in fit_hist.history["val_mae"]]
    }

    return model, history, best_ckpt_path


# -------------------------------------------------------------------------
# Main CLI Entrypoint
# -------------------------------------------------------------------------
def run_training_pipeline(
    model_name: str = "hybrid",
    backend: str = "torch",
    data_dir: str = "./data/processed",
    results_dir: str = "./results",
    checkpoint_dir: str = "./models/checkpoints",
    epochs: int = 40,
    batch_size: int = 64,
    lr: float = 0.001
) -> Dict[str, Any]:
    """Runs complete end-to-end training and evaluation for a specified model."""
    print(f"\n=======================================================")
    print(f" TRAINING MODEL: {model_name.upper()} | BACKEND: {backend.upper()}")
    print(f"=======================================================")

    # Load data
    train_file = Path(data_dir) / "train_data.npz"
    val_file = Path(data_dir) / "val_data.npz"
    test_file = Path(data_dir) / "test_data.npz"

    if not (train_file.exists() and val_file.exists() and test_file.exists()):
        raise FileNotFoundError(f"Missing preprocessed .npz files in {data_dir}. Run preprocessing first!")

    train_data = np.load(train_file)
    val_data = np.load(val_file)
    test_data = np.load(test_file)

    X_train, y_train = train_data["X"], train_data["y"]
    X_val, y_val = val_data["X"], val_data["y"]
    X_test, y_test = test_data["X"], test_data["y"]

    print(f"Loaded datasets:")
    print(f"  Train: X={X_train.shape}, y={y_train.shape}")
    print(f"  Val:   X={X_val.shape}, y={y_val.shape}")
    print(f"  Test:  X={X_test.shape}, y={y_test.shape}")

    # Train
    if backend in ["torch", "pytorch"]:
        model, history, ckpt_path = train_torch_model(
            model_name=model_name,
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            epochs=epochs, batch_size=batch_size, lr=lr,
            checkpoint_dir=checkpoint_dir
        )
    else:
        model, history, ckpt_path = train_keras_model(
            model_name=model_name,
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            epochs=epochs, batch_size=batch_size, lr=lr,
            checkpoint_dir=checkpoint_dir
        )

    # Save training history curve
    fig_dir = Path(results_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    curve_path = str(fig_dir / f"{model_name.lower()}_training_curves.png")
    plot_training_curves(history, save_path=curve_path)
    print(f"[OK] Training curve saved to {curve_path}")

    # Evaluate on independent test set
    eval_res = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        model_name=model_name,
        backend=backend,
        output_dir=results_dir
    )

    return {
        "model": model,
        "history": history,
        "checkpoint_path": ckpt_path,
        "evaluation": eval_res
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EEG Emotion Recognition Models")
    parser.add_argument("--model", type=str, default="hybrid", choices=["cnn", "bilstm", "hybrid", "ablation"])
    parser.add_argument("--backend", type=str, default="torch", choices=["torch", "keras"])
    parser.add_argument("--data_dir", type=str, default="./data/processed")
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--checkpoint_dir", type=str, default="./models/checkpoints")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    run_training_pipeline(
        model_name=args.model,
        backend=args.backend,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
