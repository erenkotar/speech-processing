"""
visualization.py
----------------
Three-panel diagnostic plot for a single KNN prediction:
    1. Input waveform (annotated with predicted digit + confidence)
    2. MFCC heatmap (frame-by-frame coefficients)
    3. Top-k neighbor distances (bar chart, coloured by neighbor label)

Mirrors the style of HW2/main.py:plot_results.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot_prediction(
    audio: np.ndarray,
    sr: int,
    mfcc: np.ndarray,
    neighbors: List[Tuple[float, object]],
    prediction: object,
    confidence: float,
    save_png: str = "output_plot.png",
) -> None:
    t_axis = np.linspace(0, len(audio) / sr, len(audio))
    fig = plt.figure(figsize=(12, 8), constrained_layout=True)
    gs = gridspec.GridSpec(3, 1, figure=fig, height_ratios=[1.5, 2, 1.5])

    # ---- Panel 1: Waveform ----------------------------------------------
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(t_axis, audio, linewidth=0.5, color="#2a6ebb")
    ax1.set_title(
        f"Input waveform   |   predicted digit = {prediction}   "
        f"(confidence {confidence:.0%})",
        fontsize=11,
    )
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    if len(t_axis):
        ax1.set_xlim(t_axis[0], t_axis[-1])

    # ---- Panel 2: MFCC heatmap ------------------------------------------
    ax2 = fig.add_subplot(gs[1])
    im = ax2.imshow(
        mfcc,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        extent=[0, len(audio) / sr, 0, mfcc.shape[0]],
    )
    ax2.set_title("MFCC frames", fontsize=11)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("MFCC index")
    fig.colorbar(im, ax=ax2, pad=0.01, fraction=0.04)

    # ---- Panel 3: Top-k neighbor distances ------------------------------
    ax3 = fig.add_subplot(gs[2])
    dists = [d for d, _ in neighbors]
    labels = [str(lbl) for _, lbl in neighbors]
    positions = np.arange(len(neighbors))

    cmap = plt.get_cmap("tab10")
    colors = [cmap(int(lbl) % 10) if lbl.isdigit() else cmap(0) for lbl in labels]
    ax3.bar(positions, dists, color=colors, edgecolor="black", linewidth=0.5)
    ax3.set_xticks(positions)
    ax3.set_xticklabels([f"#{i+1}\nlabel={lbl}" for i, lbl in enumerate(labels)], fontsize=8)
    ax3.set_ylabel("Euclidean distance")
    ax3.set_title(f"Top-{len(neighbors)} nearest training samples", fontsize=11)

    fig.savefig(save_png, dpi=150)
    plt.close(fig)
    print(f"[plot]  Wrote {save_png}")
