"""
Comprehensive Visualization and Analysis Utilities for EEG Emotion Recognition
Provides dual-mode rendering:
  1. High-resolution matplotlib/seaborn plots (default when matplotlib is installed, e.g. in Google Colab)
  2. Standalone, zero-dependency SVG vector graphics generator (guarantees plots are rendered on all platforms)
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from scipy import signal, stats

from src.config import CONFIG

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False


# =========================================================================
# Pure Python SVG Rendering Engine (Zero Dependency Fallback)
# =========================================================================

def _render_svg_barchart(
    categories: List[str],
    series1_name: str,
    series1_vals: List[float],
    series2_name: str,
    series2_vals: List[float],
    title: str,
    save_path: str
) -> None:
    width, height = 800, 480
    margin_l, margin_r, margin_t, margin_b = 80, 40, 60, 80
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    
    max_val = max(max(series1_vals, default=1.0), max(series2_vals, default=1.0)) * 1.25
    if max_val < 1e-4:
        max_val = 1.0

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" style="background-color:#ffffff; font-family:Arial, sans-serif;">',
        f'<text x="{width//2}" y="32" font-size="18" font-weight="bold" text-anchor="middle" fill="#1e293b">{title}</text>',
        # Legend
        f'<rect x="{width - 240}" y="20" width="16" height="16" fill="#2563eb" rx="2"/>',
        f'<text x="{width - 215}" y="33" font-size="12" fill="#334155">{series1_name}</text>',
        f'<rect x="{width - 120}" y="20" width="16" height="16" fill="#f97316" rx="2"/>',
        f'<text x="{width - 95}" y="33" font-size="12" fill="#334155">{series2_name}</text>',
        # Axes
        f'<line x1="{margin_l}" y1="{height - margin_b}" x2="{width - margin_r}" y2="{height - margin_b}" stroke="#94a3b8" stroke-width="1.5"/>',
        f'<line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{height - margin_b}" stroke="#94a3b8" stroke-width="1.5"/>'
    ]

    # Y-axis ticks
    num_ticks = 5
    for t in range(num_ticks + 1):
        y_val = (max_val / num_ticks) * t
        y_pos = (height - margin_b) - (y_val / max_val) * plot_h
        svg.append(f'<line x1="{margin_l - 5}" y1="{y_pos:.1f}" x2="{width - margin_r}" y2="{y_pos:.1f}" stroke="#e2e8f0" stroke-width="1"/>')
        svg.append(f'<text x="{margin_l - 12}" y="{y_pos + 4:.1f}" font-size="11" fill="#64748b" text-anchor="end">{y_val:.2f}</text>')

    # Grouped bars
    num_groups = len(categories)
    group_width = plot_w / num_groups
    bar_width = group_width * 0.32
    
    for i, cat in enumerate(categories):
        cx = margin_l + i * group_width + group_width / 2
        # Bar 1
        v1 = series1_vals[i]
        b1_h = (v1 / max_val) * plot_h
        b1_x = cx - bar_width - 2
        b1_y = (height - margin_b) - b1_h
        svg.append(f'<rect x="{b1_x:.1f}" y="{b1_y:.1f}" width="{bar_width:.1f}" height="{b1_h:.1f}" fill="#2563eb" rx="3"/>')
        svg.append(f'<text x="{b1_x + bar_width/2:.1f}" y="{b1_y - 6:.1f}" font-size="10" font-weight="bold" fill="#1e293b" text-anchor="middle">{v1:.3f}</text>')

        # Bar 2
        v2 = series2_vals[i]
        b2_h = (v2 / max_val) * plot_h
        b2_x = cx + 2
        b2_y = (height - margin_b) - b2_h
        svg.append(f'<rect x="{b2_x:.1f}" y="{b2_y:.1f}" width="{bar_width:.1f}" height="{b2_h:.1f}" fill="#f97316" rx="3"/>')
        svg.append(f'<text x="{b2_x + bar_width/2:.1f}" y="{b2_y - 6:.1f}" font-size="10" font-weight="bold" fill="#1e293b" text-anchor="middle">{v2:.3f}</text>')

        # Category label
        svg.append(f'<text x="{cx:.1f}" y="{height - margin_b + 25}" font-size="12" font-weight="bold" fill="#334155" text-anchor="middle">{cat}</text>')

    svg.append('</svg>')
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg))


def _render_svg_curves(history: Dict[str, List[float]], save_path: str) -> None:
    width, height = 750, 400
    margin_l, margin_r, margin_t, margin_b = 60, 40, 50, 50
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    train_l = history.get("train_loss", [])
    val_l = history.get("val_loss", [])
    epochs = len(train_l)
    if epochs == 0:
        return

    all_vals = train_l + val_l
    min_v, max_v = min(all_vals), max(all_vals)
    span = (max_v - min_v) if (max_v - min_v) > 1e-4 else 1.0

    def pt(e, v):
        x = margin_l + (e / max(1, epochs - 1)) * plot_w
        y = (height - margin_b) - ((v - min_v) / span) * plot_h
        return f"{x:.1f},{y:.1f}"

    pts_tr = " ".join([pt(e, train_l[e]) for e in range(epochs)])
    pts_vl = " ".join([pt(e, val_l[e]) for e in range(epochs)]) if val_l else ""

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" style="background-color:#ffffff; font-family:Arial, sans-serif;">',
        f'<text x="{width//2}" y="28" font-size="16" font-weight="bold" text-anchor="middle" fill="#1e293b">Training and Validation Loss Trajectory (Huber Loss)</text>',
        f'<polyline points="{pts_tr}" fill="none" stroke="#2563eb" stroke-width="2.5"/>',
    ]
    if pts_vl:
        svg.append(f'<polyline points="{pts_vl}" fill="none" stroke="#ef4444" stroke-width="2.5" stroke-dasharray="5,5"/>')
        svg.append(f'<text x="{width - 150}" y="55" font-size="12" fill="#ef4444">--- Validation Loss</text>')
    svg.append(f'<text x="{width - 150}" y="40" font-size="12" fill="#2563eb">─── Training Loss</text>')
    svg.append(f'<line x1="{margin_l}" y1="{height - margin_b}" x2="{width - margin_r}" y2="{height - margin_b}" stroke="#94a3b8" stroke-width="1.5"/>')
    svg.append(f'<line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{height - margin_b}" stroke="#94a3b8" stroke-width="1.5"/>')
    svg.append(f'<text x="{width//2}" y="{height - 12}" font-size="12" fill="#64748b" text-anchor="middle">Epoch</text>')
    svg.append('</svg>')

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg))


# =========================================================================
# Public Visualization Functions (Handles Both MPL and Pure SVG)
# =========================================================================

def plot_training_curves(history: Dict[str, List[float]], save_path: Optional[str] = None) -> None:
    if not save_path:
        return
    base, _ = os.path.splitext(save_path)
    svg_path = base + ".svg"
    _render_svg_curves(history, svg_path)

    if MPL_AVAILABLE:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        epochs = range(1, len(history["train_loss"]) + 1)
        axes[0].plot(epochs, history["train_loss"], 'o-', color='#1f77b4', label='Training Loss')
        if "val_loss" in history and history["val_loss"]:
            axes[0].plot(epochs, history["val_loss"], 's-', color='#d62728', label='Validation Loss')
        axes[0].set_title("Loss Trajectory (Huber Loss)", fontsize=12, fontweight='bold')
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        if "train_mae" in history:
            axes[1].plot(epochs, history["train_mae"], 'o-', color='#2ca02c', label='Training MAE')
            if "val_mae" in history and history["val_mae"]:
                axes[1].plot(epochs, history["val_mae"], 's-', color='#ff7f0e', label='Validation MAE')
            axes[1].set_title("Mean Absolute Error Trajectory", fontsize=12, fontweight='bold')
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("MAE")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close(fig)


def plot_model_comparison(comparison_dict: Dict[str, Dict[str, float]], save_path: Optional[str] = None) -> None:
    if not save_path:
        return
    models = list(comparison_dict.keys())
    maes = [comparison_dict[m]["MAE"] for m in models]
    rmses = [comparison_dict[m]["RMSE"] for m in models]

    base, _ = os.path.splitext(save_path)
    svg_path = base + ".svg"
    _render_svg_barchart(
        categories=models,
        series1_name="MAE (Lower is Better)",
        series1_vals=maes,
        series2_name="RMSE (Lower is Better)",
        series2_vals=rmses,
        title="Cross-Architecture Performance on Independent Test Cohort",
        save_path=svg_path
    )

    if MPL_AVAILABLE:
        x = np.arange(len(models))
        width = 0.35
        fig, ax = plt.subplots(figsize=(10, 6))
        rects1 = ax.bar(x - width/2, maes, width, label='MAE', color='#2b5c8f')
        rects2 = ax.bar(x + width/2, rmses, width, label='RMSE', color='#e26d5c')
        ax.set_ylabel('Rating Error', fontsize=11)
        ax.set_title('Cross-Architecture Comparison', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontweight='bold')
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close(fig)


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> None:
    if not save_path:
        return
    targets = target_names or CONFIG.dataset.target_names
    base, _ = os.path.splitext(save_path)
    svg_path = base + ".svg"

    # SVG scatter plots for 4 dimensions (2x2 grid)
    width, height = 800, 700
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" style="background-color:#ffffff; font-family:Arial, sans-serif;">',
        f'<text x="{width//2}" y="30" font-size="18" font-weight="bold" text-anchor="middle" fill="#0f172a">Actual vs. Predicted Ratings across 4 Affective Dimensions</text>'
    ]

    quads = [(60, 60, 320, 260), (440, 60, 320, 260), (60, 380, 320, 260), (440, 380, 320, 260)]
    colors = ['#2563eb', '#f97316', '#16a34a', '#dc2626']

    for i, target in enumerate(targets):
        qx, qy, qw, qh = quads[i]
        yt = y_true[:, i]
        yp = y_pred[:, i]
        mae = float(np.mean(np.abs(yt - yp)))
        r, _ = stats.pearsonr(yt, yp) if len(yt) > 2 and np.std(yt) > 1e-4 and np.std(yp) > 1e-4 else (0.0, 1.0)

        # Quadrant Box
        svg.append(f'<rect x="{qx}" y="{qy}" width="{qw}" height="{qh}" fill="#f8fafc" stroke="#cbd5e1" rx="4"/>')
        svg.append(f'<text x="{qx + qw//2}" y="{qy + 22}" font-size="13" font-weight="bold" fill="#1e293b" text-anchor="middle">{target} (MAE: {mae:.3f}, r: {r:.3f})</text>')
        # Ideal line
        svg.append(f'<line x1="{qx + 20}" y1="{qy + qh - 20}" x2="{qx + qw - 20}" y2="{qy + 40}" stroke="#94a3b8" stroke-dasharray="4,4" stroke-width="1.5"/>')

        # Scatter dots
        for val_t, val_p in zip(yt[:60], yp[:60]):
            sx = qx + 20 + ((val_t - 1.0) / 8.0) * (qw - 40)
            sy = (qy + qh - 20) - ((val_p - 1.0) / 8.0) * (qh - 60)
            svg.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="3" fill="{colors[i]}" opacity="0.6"/>')

    svg.append('</svg>')
    os.makedirs(os.path.dirname(os.path.abspath(svg_path)), exist_ok=True)
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg))

    if MPL_AVAILABLE:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        for i, target in enumerate(targets):
            ax = axes[i]
            yt = y_true[:, i]
            yp = y_pred[:, i]
            r, _ = stats.pearsonr(yt, yp) if len(yt) > 2 and np.std(yt) > 1e-4 and np.std(yp) > 1e-4 else (0.0, 1.0)
            mae = np.mean(np.abs(yt - yp))
            ax.scatter(yt, yp, alpha=0.5, color=colors[i], s=25)
            ax.plot([1.0, 9.0], [1.0, 9.0], 'k--', lw=1.5, label='Ideal y=x')
            ax.set_title(f"{target} (MAE: {mae:.3f}, r: {r:.3f})", fontweight='bold')
            ax.set_xlim(1.0, 9.0)
            ax.set_ylim(1.0, 9.0)
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close(fig)


def plot_error_distributions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> None:
    if not save_path:
        return
    targets = target_names or CONFIG.dataset.target_names
    residuals = y_pred - y_true
    base, _ = os.path.splitext(save_path)
    svg_path = base + ".svg"

    width, height = 800, 320
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" style="background-color:#ffffff; font-family:Arial, sans-serif;">',
        f'<text x="{width//2}" y="28" font-size="16" font-weight="bold" text-anchor="middle" fill="#0f172a">Residual Error Distributions (Predicted - Actual)</text>'
    ]
    colors = ['#2563eb', '#f97316', '#16a34a', '#dc2626']
    box_w = (width - 100) / 4

    for i, target in enumerate(targets):
        bx = 50 + i * box_w
        res = residuals[:, i]
        m_err = float(np.mean(res))
        s_err = float(np.std(res))

        svg.append(f'<rect x="{bx}" y="50" width="{box_w - 15}" height="230" fill="#f8fafc" stroke="#e2e8f0" rx="3"/>')
        svg.append(f'<text x="{bx + (box_w - 15)/2}" y="75" font-size="12" font-weight="bold" fill="#1e293b" text-anchor="middle">{target}</text>')
        svg.append(f'<text x="{bx + (box_w - 15)/2}" y="95" font-size="11" fill="#64748b" text-anchor="middle">μ: {m_err:+.2f}, σ: {s_err:.2f}</text>')

        # Histogram bins
        counts, edges = np.histogram(res, bins=8)
        max_c = max(max(counts), 1)
        bin_w = (box_w - 35) / 8
        for b_idx, count in enumerate(counts):
            b_h = (count / max_c) * 110
            bar_x = bx + 10 + b_idx * bin_w
            bar_y = 250 - b_h
            svg.append(f'<rect x="{bar_x:.1f}" y="{bar_y:.1f}" width="{bin_w - 2:.1f}" height="{b_h:.1f}" fill="{colors[i]}" opacity="0.8"/>')

    svg.append('</svg>')
    os.makedirs(os.path.dirname(os.path.abspath(svg_path)), exist_ok=True)
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg))

    if MPL_AVAILABLE:
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        for i, target in enumerate(targets):
            res = residuals[:, i]
            sns.histplot(res, kde=True, ax=axes[i], color=colors[i], bins=15)
            axes[i].axvline(0, color='black', linestyle='--')
            axes[i].set_title(f"{target} Residuals", fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close(fig)


def plot_eeg_raw_vs_filtered(
    raw_signal: np.ndarray,
    filtered_signal: np.ndarray,
    fs: int = 128,
    duration_sec: float = 4.0,
    save_path: Optional[str] = None
) -> None:
    if not save_path:
        return
    base, _ = os.path.splitext(save_path)
    svg_path = base + ".svg"

    samples = int(duration_sec * fs)
    width, height = 750, 360
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" style="background-color:#ffffff; font-family:Arial, sans-serif;">',
        f'<text x="{width//2}" y="28" font-size="16" font-weight="bold" text-anchor="middle" fill="#0f172a">EEG Signal Preprocessing: Raw vs. 4-45 Hz Bandpass Filter</text>',
        f'<text x="{width - 180}" y="55" font-size="12" fill="#94a3b8">─── Raw Signal (with drift)</text>',
        f'<text x="{width - 180}" y="70" font-size="12" fill="#2563eb">─── Filtered (4-45 Hz)</text>'
    ]
    # Draw filtered waveform
    r_pts = []
    f_pts = []
    plot_w, plot_h = 650, 100
    mid_y_raw = 130
    mid_y_filt = 260
    max_amp = max(np.max(np.abs(raw_signal[:samples])), 1.0)

    for i in range(samples):
        x = 50 + (i / samples) * plot_w
        yr = mid_y_raw - (raw_signal[i] / max_amp) * 35
        yf = mid_y_filt - (filtered_signal[i] / max_amp) * 35
        r_pts.append(f"{x:.1f},{yr:.1f}")
        f_pts.append(f"{x:.1f},{yf:.1f}")

    svg.append(f'<polyline points="{" ".join(r_pts)}" fill="none" stroke="#94a3b8" stroke-width="1.2"/>')
    svg.append(f'<polyline points="{" ".join(f_pts)}" fill="none" stroke="#2563eb" stroke-width="1.5"/>')
    svg.append('</svg>')

    os.makedirs(os.path.dirname(os.path.abspath(svg_path)), exist_ok=True)
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg))
