"""
lpc_utils.py
------------
Core DSP utilities for LPC analysis of speech:
  - Pre-emphasis
  - Framing & windowing
  - LPC estimation via Levinson-Durbin (and librosa wrapper)
  - LPC residual
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

    Args:
        signal: 1-D float array
        alpha:  pre-emphasis coefficient (typically 0.95–0.97)

    Returns:
        Pre-emphasized signal of same length.
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

    Args:
        signal:   1-D float array (samples)
        sr:       sample rate in Hz
        frame_ms: frame length in milliseconds (10–20 ms recommended)
        hop_ms:   hop (step) size in milliseconds

    Returns:
        frames      : 2-D array of shape (num_frames, frame_length)
        frame_len   : frame length in samples
        hop_len     : hop length in samples
    """
    frame_len = int(sr * frame_ms / 1000.0)
    hop_len   = int(sr * hop_ms  / 1000.0)

    # Zero-pad so the last frame is full
    num_frames = 1 + (len(signal) - frame_len) // hop_len
    padded_len = frame_len + (num_frames - 1) * hop_len
    padded     = np.zeros(padded_len)
    padded[:len(signal)] = signal

    # Build frame matrix efficiently with strides
    shape   = (num_frames, frame_len)
    strides = (padded.strides[0] * hop_len, padded.strides[0])
    frames  = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides).copy()

    return frames, frame_len, hop_len


# ─────────────────────────────────────────────
# 3. WINDOWING
# ─────────────────────────────────────────────

def apply_window(frames: np.ndarray, window_type: str = "hamming") -> np.ndarray:
    """
    Multiply each frame by a window function to reduce spectral leakage.

    Args:
        frames:      2-D array (num_frames, frame_len)
        window_type: 'hamming' | 'hanning' | 'blackman' | 'rectangular'

    Returns:
        Windowed frames, same shape as input.
    """
    frame_len = frames.shape[1]

    windows = {
        "hamming":     np.hamming(frame_len),
        "hanning":     np.hanning(frame_len),
        "blackman":    np.blackman(frame_len),
        "rectangular": np.ones(frame_len),
    }

    if window_type not in windows:
        raise ValueError(f"Unknown window type '{window_type}'. "
                         f"Choose from: {list(windows.keys())}")

    return frames * windows[window_type]


# ─────────────────────────────────────────────
# 4. LPC ESTIMATION (Levinson-Durbin)
# ─────────────────────────────────────────────

def levinson_durbin(r: np.ndarray, order: int) -> tuple[np.ndarray, float]:
    """
    Levinson-Durbin recursion to solve the Yule-Walker equations:
        R·a = -r[1:p+1]
    where R is the Toeplitz autocorrelation matrix.

    Args:
        r:     autocorrelation sequence r[0..order]
        order: LPC order p

    Returns:
        a_coeffs: LPC coefficients [a1, a2, ..., ap]  (without leading 1)
        gain:     prediction error power G
    """
    a = np.zeros(order)
    e = r[0]  # prediction error (starts as total signal power)

    for i in range(order):
        # Reflection coefficient k_i
        k = -np.dot(a[:i], r[i - 1::-1][:i]) if i > 0 else 0.0
        k = (k - r[i + 1]) / (e + 1e-10)

        # Update coefficients
        a_new = a.copy()
        a_new[i] = k
        if i > 0:
            a_new[:i] += k * a[i - 1::-1]

        a = a_new
        e *= 1.0 - k ** 2

    return a, max(e, 1e-10)


def compute_lpc(frame: np.ndarray, order: int, method: str = "librosa") -> tuple[np.ndarray, float]:
    """
    Estimate LPC coefficients for a single (windowed) frame.

    Args:
        frame:  1-D float array (one windowed frame)
        order:  LPC order p (typically 10–16 for speech)
        method: 'librosa' (recommended) or 'levinson' (manual implementation)

    Returns:
        a:    LPC coefficients [a1..ap] (length = order)
        gain: excitation gain G
    """
    if method == "librosa":
        # librosa.lpc returns [1, a1, a2, ..., ap]
        coeffs = librosa.lpc(frame, order=order)
        a = coeffs[1:]          # drop leading '1'
        # Gain = sqrt of prediction error power
        residual = lfilter(coeffs, [1.0], frame)
        gain = np.sqrt(np.mean(residual ** 2)) + 1e-10
        return a, gain

    elif method == "levinson":
        # Compute normalized autocorrelation
        r = np.array([np.dot(frame[i:], frame[:len(frame) - i]) / len(frame)
                      for i in range(order + 1)])
        return levinson_durbin(r, order)

    else:
        raise ValueError("method must be 'librosa' or 'levinson'")


def compute_lpc_all_frames(
    frames: np.ndarray,
    order: int,
    method: str = "librosa",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute LPC coefficients for every frame.

    Returns:
        all_a    : (num_frames, order)  – LPC coefficient matrix
        all_gain : (num_frames,)        – per-frame gain
    """
    num_frames = frames.shape[0]
    all_a    = np.zeros((num_frames, order))
    all_gain = np.zeros(num_frames)

    for i, frame in enumerate(frames):
        a, g = compute_lpc(frame, order, method)
        all_a[i]    = a
        all_gain[i] = g

    return all_a, all_gain


# ─────────────────────────────────────────────
# 5. LPC RESIDUAL
# ─────────────────────────────────────────────

def compute_residual(frame: np.ndarray, a: np.ndarray) -> np.ndarray:
    """
    Compute the LPC prediction residual (error signal):
        e[n] = x[n] + a1*x[n-1] + ... + ap*x[n-p]

    The residual approximates the glottal excitation for voiced speech
    and noise for unvoiced speech.

    Args:
        frame: 1-D windowed frame
        a:     LPC coefficients [a1..ap]

    Returns:
        residual: 1-D array, same length as frame
    """
    coeffs = np.concatenate([[1.0], a])   # [1, a1, ..., ap]
    return lfilter(coeffs, [1.0], frame)


# ─────────────────────────────────────────────
# 6. LPC SPECTRUM
# ─────────────────────────────────────────────

def compute_lpc_spectrum(
    a: np.ndarray,
    gain: float,
    n_fft: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the LPC spectral envelope |H(f)| = G / |A(e^jω)|
    where A(z) = 1 + a1*z^-1 + ... + ap*z^-p

    Args:
        a:     LPC coefficients [a1..ap]
        gain:  excitation gain G
        n_fft: FFT size for frequency resolution

    Returns:
        freqs:        frequency axis (0 .. sr/2) — normalized [0..0.5] here,
                      caller scales by sr
        lpc_spectrum: magnitude spectrum in dB (one-sided, n_fft//2 + 1 bins)
    """
    # Full A polynomial: [1, a1, a2, ..., ap]
    A_poly = np.concatenate([[1.0], a])

    # Frequency response of 1/A(z) using FFT of zero-padded polynomial
    A_freq = np.fft.rfft(A_poly, n=n_fft)
    H_freq = gain / (np.abs(A_freq) + 1e-10)

    freqs = np.fft.rfftfreq(n_fft)   # normalized [0, 0.5]
    lpc_spectrum_dB = 20 * np.log10(H_freq + 1e-10)

    return freqs, lpc_spectrum_dB


# ─────────────────────────────────────────────
# 7. LPC → CEPSTRAL COEFFICIENTS (LPCC)
# ─────────────────────────────────────────────

def lpc_to_cepstrum(a: np.ndarray, gain: float, num_ceps: int = 13) -> np.ndarray:
    """
    Convert LPC coefficients to LPC Cepstral Coefficients (LPCC)
    using the standard recursion:

        c[0]   = log(G)
        c[m]   = -a[m]  -  Σ_{k=1}^{m-1} (k/m) * c[k] * a[m-k],  1 ≤ m ≤ p
        c[m]   =         - Σ_{k=1}^{p}   (k/m) * c[k] * a[m-k],  m > p

    LPCCs decorrelate LPC coefficients and are often better features
    for speech recognition than raw LPC coefficients.

    Args:
        a:        LPC coefficients [a1..ap] (length p)
        gain:     prediction gain G
        num_ceps: number of cepstral coefficients to return (typically 12–16)

    Returns:
        c: LPCC vector of length num_ceps
    """
    p = len(a)
    c = np.zeros(num_ceps)
    c[0] = np.log(gain + 1e-10)

    for m in range(1, num_ceps):
        if m <= p:
            # Direct LPC term + correction from earlier cepstral values
            c[m] = -a[m - 1]
            for k in range(1, m):
                c[m] -= (k / m) * c[k] * a[m - k - 1]
        else:
            # Beyond LPC order: only the cepstral recursion
            for k in range(1, p + 1):
                if m - k - 1 >= 0:
                    c[m] -= (k / m) * c[k] * a[m - k - 1]

    return c


def compute_lpcc_all_frames(
    all_a: np.ndarray,
    all_gain: np.ndarray,
    num_ceps: int = 13,
) -> np.ndarray:
    """
    Compute LPCC matrix for all frames.

    Returns:
        lpcc_matrix: (num_frames, num_ceps)
    """
    return np.array([
        lpc_to_cepstrum(a, g, num_ceps)
        for a, g in zip(all_a, all_gain)
    ])


# ─────────────────────────────────────────────
# 8. VOICED / UNVOICED DETECTION (energy-based)
# ─────────────────────────────────────────────

def classify_frames(frames: np.ndarray, threshold_db: float = -40.0) -> np.ndarray:
    """
    Energy + zero-crossing rate voiced/unvoiced classification.

    A frame is voiced if it has high energy (above threshold_db)
    AND a relatively low zero-crossing rate (speech heuristic).

    Returns:
        labels: 1-D bool array, True = voiced frame
    """
    # Energy criterion
    rms    = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-10)
    rms_dB = 20 * np.log10(rms)
    energy_voiced = rms_dB > threshold_db

    # Zero-crossing rate criterion: voiced speech has low ZCR
    zcr = np.mean(np.abs(np.diff(np.sign(frames), axis=1)), axis=1) / 2.0
    zcr_median  = np.median(zcr)
    zcr_voiced  = zcr < (zcr_median * 1.5)   # voiced frames have below-median ZCR

    is_voiced = energy_voiced & zcr_voiced

    # Fallback: if classification yields no voiced or no unvoiced frames,
    # split by position (first half = voiced, second half = unvoiced)
    if is_voiced.all() or (~is_voiced).all():
        mid = len(is_voiced) // 2
        is_voiced = np.array([True] * mid + [False] * (len(is_voiced) - mid))

    return is_voiced
