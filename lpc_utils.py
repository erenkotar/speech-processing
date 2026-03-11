"""
lpc_utils.py
------------
Core DSP utilities for LPC analysis of speech:
  - Pre-emphasis
  - Framing & windowing
  - LPC estimation (via librosa)
  - LPC spectrum
  - LPC → Cepstral coefficients (LPCC)
"""

import numpy as np
import librosa
from scipy.signal import lfilter


# ─────────────────────────────────────────────
# 1. PRE-EMPHASIS
# ─────────────────────────────────────────────

def pre_emphasis(signal: np.ndarray, alpha: float = 0.97) -> np.ndarray:
    """
    Apply first-order high-pass pre-emphasis filter.
      y[n] = x[n] - alpha * x[n-1]
    """
    return np.append(signal[0], signal[1:] - alpha * signal[:-1])


# ─────────────────────────────────────────────
# 2. FRAMING
# ─────────────────────────────────────────────

def frame_signal(
    signal: np.ndarray,
    sr: int,
    frame_ms: float = 20.0,
    hop_ms: float = 10.0,
) -> tuple[np.ndarray, int, int]:
    """
    Slice a 1-D signal into overlapping frames.

    Returns:
        frames    : 2-D array (num_frames, frame_length)
        frame_len : frame length in samples
        hop_len   : hop length in samples
    """
    frame_len = int(sr * frame_ms / 1000.0)
    hop_len   = int(sr * hop_ms  / 1000.0)

    num_frames = 1 + (len(signal) - frame_len) // hop_len
    padded_len = max(len(signal), frame_len + (num_frames - 1) * hop_len)
    padded     = np.zeros(padded_len)
    padded[:len(signal)] = signal

    shape   = (num_frames, frame_len)
    strides = (padded.strides[0] * hop_len, padded.strides[0])
    frames  = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides).copy()

    return frames, frame_len, hop_len


# ─────────────────────────────────────────────
# 3. WINDOWING
# ─────────────────────────────────────────────

def apply_window(frames: np.ndarray) -> np.ndarray:
    """Apply Hamming window to each frame."""
    return frames * np.hamming(frames.shape[1])


# ─────────────────────────────────────────────
# 4. LPC ESTIMATION
# ─────────────────────────────────────────────

def compute_lpc(frame: np.ndarray, order: int) -> tuple[np.ndarray, float]:
    """
    Estimate LPC coefficients for a single windowed frame using librosa.

    Returns:
        a    : LPC coefficients [a1..ap] (length = order)
        gain : excitation gain G
    """
    coeffs = librosa.lpc(frame, order=order)  # [1, a1, a2, ..., ap]
    a = coeffs[1:]
    residual = lfilter(coeffs, [1.0], frame)
    gain = np.sqrt(np.mean(residual ** 2)) + 1e-10
    return a, gain


def compute_lpc_all_frames(
    frames: np.ndarray,
    order: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute LPC coefficients for every frame.

    Returns:
        all_a    : (num_frames, order)
        all_gain : (num_frames,)
    """
    num_frames = frames.shape[0]
    all_a    = np.zeros((num_frames, order))
    all_gain = np.zeros(num_frames)

    for i, frame in enumerate(frames):
        a, g = compute_lpc(frame, order)
        all_a[i]    = a
        all_gain[i] = g

    return all_a, all_gain


# ─────────────────────────────────────────────
# 5. LPC SPECTRUM
# ─────────────────────────────────────────────

def compute_lpc_spectrum(
    a: np.ndarray,
    gain: float,
    n_fft: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the LPC spectral envelope |H(f)| = G / |A(e^jω)|.

    Returns:
        freqs        : normalized frequency axis [0..0.5]
        lpc_spectrum : magnitude in dB
    """
    A_poly = np.concatenate([[1.0], a])
    A_freq = np.fft.rfft(A_poly, n=n_fft)
    H_freq = gain / (np.abs(A_freq) + 1e-10)

    freqs = np.fft.rfftfreq(n_fft)
    lpc_spectrum_dB = 20 * np.log10(H_freq + 1e-10)

    return freqs, lpc_spectrum_dB


# ─────────────────────────────────────────────
# 6. LPC → CEPSTRAL COEFFICIENTS (LPCC)
# ─────────────────────────────────────────────

def lpc_to_cepstrum(a: np.ndarray, gain: float, num_ceps: int = 13) -> np.ndarray:
    """
    Convert LPC coefficients to LPCC using the standard recursion:
        c[0] = log(G)
        c[m] = -a[m] - Σ (k/m) * c[k] * a[m-k]
    """
    p = len(a)
    c = np.zeros(num_ceps)
    c[0] = np.log(gain + 1e-10)

    for m in range(1, num_ceps):
        if m <= p:
            c[m] = -a[m - 1]
            for k in range(1, m):
                c[m] -= (k / m) * c[k] * a[m - k - 1]
        else:
            for k in range(1, p + 1):
                if m - k - 1 >= 0:
                    c[m] -= (k / m) * c[k] * a[m - k - 1]

    return c


def compute_lpcc_all_frames(
    all_a: np.ndarray,
    all_gain: np.ndarray,
    num_ceps: int = 13,
) -> np.ndarray:
    """Compute LPCC matrix for all frames. Returns (num_frames, num_ceps)."""
    return np.array([
        lpc_to_cepstrum(a, g, num_ceps)
        for a, g in zip(all_a, all_gain)
    ])
