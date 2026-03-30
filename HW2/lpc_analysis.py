"""
lpc_analysis.py
---------------
Core LPC analysis routines:
  - Frame windowing
  - Autocorrelation (lag method)
  - Levinson-Durbin recursion
  - LPC spectral envelope estimation

Reference: Rabiner & Schafer, "Introduction to Digital Speech Processing",
           Ch. 6 (Linear Predictive Analysis of Speech).
"""

import numpy as np


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------

def apply_window(frame: np.ndarray, window_type: str = "hamming") -> np.ndarray:
    """Apply a window function to a speech frame."""
    N = len(frame)
    if window_type == "hamming":
        w = np.hamming(N)
    elif window_type == "hanning":
        w = np.hanning(N)
    elif window_type == "rectangular":
        w = np.ones(N)
    else:
        raise ValueError(f"Unknown window type: {window_type}")
    return frame * w


# ---------------------------------------------------------------------------
# Autocorrelation (lag method)
# ---------------------------------------------------------------------------

def autocorrelation(frame: np.ndarray, p: int) -> np.ndarray:
    """
    Compute biased autocorrelation estimates R(0), R(1), ..., R(p).

    Parameters
    ----------
    frame : windowed speech frame (length N)
    p     : LPC order

    Returns
    -------
    R : array of shape (p+1,)
    """
    N = len(frame)
    R = np.zeros(p + 1)
    for k in range(p + 1):
        R[k] = np.dot(frame[: N - k], frame[k:])
    return R


# ---------------------------------------------------------------------------
# Levinson-Durbin recursion
# ---------------------------------------------------------------------------

def levinson_durbin(R: np.ndarray, p: int):
    """
    Solve the Yule-Walker equations R*a = r via the Levinson-Durbin
    recursion.  Runs in O(p^2) time.

    Parameters
    ----------
    R : autocorrelation vector R(0..p), shape (p+1,)
    p : LPC order

    Returns
    -------
    a    : LPC predictor coefficients [a_1, ..., a_p]
    gain : square root of the final prediction error (excitation gain G)
    k_all: reflection coefficients (useful for diagnostics)
    """
    if R[0] < 1e-12:
        return np.zeros(p), 0.0, np.zeros(p)

    a = np.zeros(p)
    e = R[0]          # initial prediction error = signal power
    k_all = np.zeros(p)

    for i in range(p):
        # Reflection coefficient for stage i
        k_i = (R[i + 1] - np.dot(a[:i], R[i:0:-1])) / e
        k_all[i] = k_i

        # Update coefficients (order-update formula)
        a_prev = a[:i].copy()
        a[i] = k_i
        a[:i] -= k_i * a_prev[::-1]

        # Update prediction error
        e *= 1.0 - k_i ** 2
        if e <= 0:
            e = 1e-12
            break

    gain = np.sqrt(e)
    return a, gain, k_all


# ---------------------------------------------------------------------------
# Full frame analysis
# ---------------------------------------------------------------------------

def lpc_analyze_frame(frame: np.ndarray, p: int, window_type: str = "hamming"):
    """
    Perform complete LPC analysis of one speech frame.

    Returns
    -------
    a    : predictor coefficients [a_1 .. a_p]
    gain : excitation gain G
    k    : reflection coefficients
    """
    windowed = apply_window(frame, window_type)
    R = autocorrelation(windowed, p)
    a, gain, k = levinson_durbin(R, p)
    return a, gain, k


# ---------------------------------------------------------------------------
# LPC spectral envelope
# ---------------------------------------------------------------------------

def lpc_spectral_envelope(a: np.ndarray, gain: float, n_fft: int = 512) -> np.ndarray:
    """
    Compute the magnitude spectrum of the all-pole synthesis filter:

        |H(e^jω)| = G / |A(e^jω)|

    Parameters
    ----------
    a     : LPC coefficients [a_1 .. a_p]
    gain  : excitation gain G
    n_fft : FFT size (returns n_fft//2 + 1 points)

    Returns
    -------
    H : one-sided magnitude spectrum (linear scale)
    """
    # A(z) = 1 - a_1 z^-1 - ... - a_p z^-p  →  [1, -a_1, ..., -a_p]
    A_poly = np.concatenate([[1.0], -a])
    A_freq = np.fft.rfft(A_poly, n=n_fft)
    H = gain / (np.abs(A_freq) + 1e-12)
    return H
