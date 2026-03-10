"""
plot_utils.py
-------------
All visualization routines for LPC analysis:
  1. Waveform overview
  2. Single frame: raw vs windowed
  3. LPC spectrum vs FFT (one frame)
  4. LPC spectrum vs FFT (voiced AND unvoiced comparison)
  5. LPC residual
  6. LPCC heatmap over time
  7. LPC coefficients over time
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator


# ── Shared style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":      130,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size":       10,
})

COLORS = {
    "waveform":  "#4C72B0",
    "window":    "#DD8452",
    "lpc":       "#C44E52",
    "fft":       "#4C72B0",
    "residual":  "#8172B2",
    "voiced":    "#2CA02C",
    "unvoiced":  "#D62728",
}


# ─────────────────────────────────────────────
# 1. WAVEFORM OVERVIEW
# ─────────────────────────────────────────────

def plot_waveform(signal: np.ndarray, sr: int, title: str = "Speech Signal",
                  save_path: str | None = None) -> plt.Figure:
    t = np.arange(len(signal)) / sr
    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.plot(t, signal, color=COLORS["waveform"], linewidth=0.6)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.set_xlim([0, t[-1]])
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────
# 2. SINGLE FRAME: RAW vs WINDOWED
# ─────────────────────────────────────────────

def plot_frame(raw_frame: np.ndarray, windowed_frame: np.ndarray,
               sr: int, frame_idx: int, label: str = "",
               save_path: str | None = None) -> plt.Figure:
    t = np.arange(len(raw_frame)) / sr * 1000  # ms
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))

    axes[0].plot(t, raw_frame, color=COLORS["waveform"], linewidth=0.8)
    axes[0].set_title(f"Frame {frame_idx} – Raw  {label}")
    axes[0].set_xlabel("Time (ms)")
    axes[0].set_ylabel("Amplitude")

    axes[1].plot(t, windowed_frame, color=COLORS["window"], linewidth=0.8)
    axes[1].set_title(f"Frame {frame_idx} – Hamming windowed  {label}")
    axes[1].set_xlabel("Time (ms)")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────
# 3. LPC SPECTRUM vs FFT (single frame)
# ─────────────────────────────────────────────

def plot_lpc_vs_fft(
    windowed_frame: np.ndarray,
    lpc_freqs: np.ndarray,
    lpc_spectrum_dB: np.ndarray,
    sr: int,
    frame_label: str = "",
    n_fft: int = 512,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Overlay the LPC spectral envelope over the FFT magnitude spectrum.
    """
    # FFT of windowed frame
    fft_mag = np.abs(np.fft.rfft(windowed_frame, n=n_fft))
    fft_dB  = 20 * np.log10(fft_mag + 1e-10)
    freq_hz = np.fft.rfftfreq(n_fft, d=1.0 / sr)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(freq_hz, fft_dB,      color=COLORS["fft"],  linewidth=0.8,
            alpha=0.65, label="FFT magnitude")
    ax.plot(lpc_freqs * sr, lpc_spectrum_dB,
            color=COLORS["lpc"],  linewidth=2.0, label="LPC envelope")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title(f"LPC Spectral Envelope vs FFT  {frame_label}")
    ax.legend()
    ax.set_xlim([0, sr / 2])
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────
# 4. VOICED vs UNVOICED COMPARISON
# ─────────────────────────────────────────────

def plot_voiced_vs_unvoiced(
    voiced_frame:   np.ndarray,
    unvoiced_frame: np.ndarray,
    voiced_lpc_freqs:    np.ndarray,
    voiced_lpc_dB:       np.ndarray,
    unvoiced_lpc_freqs:  np.ndarray,
    unvoiced_lpc_dB:     np.ndarray,
    sr: int,
    n_fft: int = 512,
    save_path: str | None = None,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
    titles = ["Voiced Frame", "Unvoiced Frame"]
    frames      = [voiced_frame, unvoiced_frame]
    lpc_freqs   = [voiced_lpc_freqs, unvoiced_lpc_freqs]
    lpc_dBs     = [voiced_lpc_dB, unvoiced_lpc_dB]
    lpc_colors  = [COLORS["voiced"], COLORS["unvoiced"]]

    for ax, title, frame, lf, ld, lc in zip(
            axes, titles, frames, lpc_freqs, lpc_dBs, lpc_colors):
        fft_mag = np.abs(np.fft.rfft(frame, n=n_fft))
        fft_dB  = 20 * np.log10(fft_mag + 1e-10)
        freq_hz = np.fft.rfftfreq(n_fft, d=1.0 / sr)

        ax.plot(freq_hz, fft_dB, color=COLORS["fft"], linewidth=0.7,
                alpha=0.6, label="FFT")
        ax.plot(lf * sr, ld, color=lc, linewidth=2.2, label="LPC envelope")
        ax.set_title(title)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude (dB)")
        ax.legend()
        ax.set_xlim([0, sr / 2])

    plt.suptitle("Voiced vs Unvoiced: LPC Spectral Envelope Comparison", y=1.01)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────
# 5. LPC RESIDUAL
# ─────────────────────────────────────────────

def plot_residual(
    frame: np.ndarray,
    residual: np.ndarray,
    sr: int,
    frame_label: str = "",
    save_path: str | None = None,
) -> plt.Figure:
    t = np.arange(len(frame)) / sr * 1000
    fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)

    axes[0].plot(t, frame,    color=COLORS["waveform"], linewidth=0.8)
    axes[0].set_title(f"Original Frame  {frame_label}")
    axes[0].set_ylabel("Amplitude")

    axes[1].plot(t, residual, color=COLORS["residual"], linewidth=0.8)
    axes[1].set_title("LPC Prediction Residual (Excitation)")
    axes[1].set_ylabel("Amplitude")
    axes[1].set_xlabel("Time (ms)")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────
# 6. LPCC HEATMAP OVER TIME
# ─────────────────────────────────────────────

def plot_lpcc_heatmap(
    lpcc_matrix: np.ndarray,
    hop_ms: float = 10.0,
    save_path: str | None = None,
) -> plt.Figure:
    """
    2-D heatmap: x = time (frames), y = cepstral coefficient index.
    """
    num_frames, num_ceps = lpcc_matrix.shape
    time_axis = np.arange(num_frames) * hop_ms  # ms

    fig, ax = plt.subplots(figsize=(11, 4))
    im = ax.imshow(
        lpcc_matrix.T,
        aspect="auto",
        origin="lower",
        extent=[0, time_axis[-1], 0, num_ceps],
        cmap="RdBu_r",
        interpolation="nearest",
    )
    plt.colorbar(im, ax=ax, label="Coefficient value")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Cepstral coefficient index")
    ax.set_title("LPCC Matrix over Time")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────
# 7. LPC COEFFICIENTS OVER TIME
# ─────────────────────────────────────────────

def plot_lpc_coefficients_over_time(
    all_a: np.ndarray,
    hop_ms: float = 10.0,
    num_to_show: int = 5,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot the first `num_to_show` LPC coefficients as time series.
    """
    num_frames = all_a.shape[0]
    time_axis  = np.arange(num_frames) * hop_ms

    fig, ax = plt.subplots(figsize=(11, 4))
    cmap = plt.get_cmap("tab10")
    for k in range(min(num_to_show, all_a.shape[1])):
        ax.plot(time_axis, all_a[:, k], label=f"a{k+1}",
                color=cmap(k), linewidth=0.9, alpha=0.85)

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Coefficient value")
    ax.set_title(f"First {num_to_show} LPC Coefficients over Time")
    ax.legend(loc="upper right", ncol=num_to_show)
    ax.axvline(x=time_axis[num_frames // 2], color="gray",
               linestyle="--", linewidth=0.8, label="voiced/unvoiced boundary")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────
# 8. COMBINED SUMMARY FIGURE
# ─────────────────────────────────────────────

def plot_summary(
    signal:       np.ndarray,
    sr:           int,
    voiced_frame: np.ndarray,
    unvoiced_frame: np.ndarray,
    v_lpc_freqs:  np.ndarray,
    v_lpc_dB:     np.ndarray,
    u_lpc_freqs:  np.ndarray,
    u_lpc_dB:     np.ndarray,
    lpcc_matrix:  np.ndarray,
    hop_ms:       float = 10.0,
    n_fft:        int   = 512,
    save_path:    str | None = None,
) -> plt.Figure:
    """
    4-panel summary figure.
    """
    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # Panel 1: Waveform
    ax1 = fig.add_subplot(gs[0, :])
    t = np.arange(len(signal)) / sr
    ax1.plot(t, signal, color=COLORS["waveform"], linewidth=0.5)
    ax1.axvline(x=1.0, color="gray", linestyle="--", linewidth=0.9,
                label="voiced / unvoiced boundary")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.set_title("Speech Signal Overview")
    ax1.legend(fontsize=8)

    # Panel 2: Voiced LPC vs FFT
    ax2 = fig.add_subplot(gs[1, 0])
    fft_v   = np.abs(np.fft.rfft(voiced_frame, n=n_fft))
    fft_v_dB = 20 * np.log10(fft_v + 1e-10)
    freq_hz = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    ax2.plot(freq_hz, fft_v_dB, color=COLORS["fft"], linewidth=0.7,
             alpha=0.6, label="FFT")
    ax2.plot(v_lpc_freqs * sr, v_lpc_dB, color=COLORS["voiced"],
             linewidth=2, label="LPC")
    ax2.set_title("Voiced Frame — LPC Envelope")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Magnitude (dB)")
    ax2.legend(fontsize=8)
    ax2.set_xlim([0, sr / 2])

    # Panel 3: Unvoiced LPC vs FFT
    ax3 = fig.add_subplot(gs[1, 1])
    fft_u    = np.abs(np.fft.rfft(unvoiced_frame, n=n_fft))
    fft_u_dB = 20 * np.log10(fft_u + 1e-10)
    ax3.plot(freq_hz, fft_u_dB, color=COLORS["fft"], linewidth=0.7,
             alpha=0.6, label="FFT")
    ax3.plot(u_lpc_freqs * sr, u_lpc_dB, color=COLORS["unvoiced"],
             linewidth=2, label="LPC")
    ax3.set_title("Unvoiced Frame — LPC Envelope")
    ax3.set_xlabel("Frequency (Hz)")
    ax3.set_ylabel("Magnitude (dB)")
    ax3.legend(fontsize=8)
    ax3.set_xlim([0, sr / 2])

    plt.suptitle("LPC Analysis of Speech — Summary", fontsize=13, y=1.01)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig
