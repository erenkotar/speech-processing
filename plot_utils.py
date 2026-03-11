"""
plot_utils.py
-------------
Visualization routines for LPC analysis:
  1. Waveform overview
  2. LPC spectrum vs FFT (one frame)
  3. LPCC heatmap over time
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


plt.rcParams.update({
    "figure.dpi":        130,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.size":         10,
})

COLORS = {
    "waveform": "#4C72B0",
    "lpc":      "#C44E52",
    "fft":      "#4C72B0",
}


# ─────────────────────────────────────────────
# 1. WAVEFORM
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
# 2. LPC SPECTRUM vs FFT
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
    """Overlay the LPC spectral envelope over the FFT magnitude spectrum."""
    fft_mag = np.abs(np.fft.rfft(windowed_frame, n=n_fft))
    fft_dB  = 20 * np.log10(fft_mag + 1e-10)
    freq_hz = np.fft.rfftfreq(n_fft, d=1.0 / sr)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(freq_hz, fft_dB, color=COLORS["fft"], linewidth=0.8,
            alpha=0.65, label="FFT magnitude")
    ax.plot(lpc_freqs * sr, lpc_spectrum_dB,
            color=COLORS["lpc"], linewidth=2.0, label="LPC envelope")
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
# 3. LPCC HEATMAP
# ─────────────────────────────────────────────

def plot_lpcc_heatmap(
    lpcc_matrix: np.ndarray,
    hop_ms: float = 10.0,
    save_path: str | None = None,
) -> plt.Figure:
    """2-D heatmap: x = time (frames), y = cepstral coefficient index."""
    num_frames, num_ceps = lpcc_matrix.shape
    time_axis = np.arange(num_frames) * hop_ms

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
