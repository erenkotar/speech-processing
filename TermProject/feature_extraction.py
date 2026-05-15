"""
feature_extraction.py
---------------------
MFCC extraction and per-utterance aggregation.

Each audio clip → a fixed-length 2*N_MFCC-dim feature vector by
concatenating the mean and standard deviation of its MFCC frames along time.
Fixed-length features let plain euclidean KNN handle variable-duration audio
without DTW.
"""

from __future__ import annotations

import numpy as np
import librosa

N_MFCC = 13
N_FFT = 512
HOP_LENGTH = 160  # 20 ms at 8 kHz


def extract_mfcc(audio: np.ndarray, sr: int = 8000, n_mfcc: int = N_MFCC) -> np.ndarray:
    """Return MFCC matrix of shape (n_mfcc, T)."""
    return librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )


def aggregate(mfcc: np.ndarray) -> np.ndarray:
    """
    Aggregate (n_mfcc, T) into a fixed-length vector by stacking
    [mean(MFCC), std(MFCC), mean(ΔMFCC), std(ΔMFCC)] along the time axis.

    Adding the delta (first temporal derivative) captures coefficient dynamics,
    which significantly helps discriminate digits like 6 vs 7 that have similar
    spectral envelopes but different temporal envelopes.
    """
    n, T = mfcc.shape
    if T == 0:
        return np.zeros(4 * n, dtype=np.float32)
    # librosa.feature.delta needs an odd width <= T. Pick the largest valid one.
    width = min(9, T if T % 2 == 1 else T - 1)
    if width >= 3:
        delta = librosa.feature.delta(mfcc, width=width)
    else:
        delta = np.zeros_like(mfcc)
    return np.concatenate([
        mfcc.mean(axis=1), mfcc.std(axis=1),
        delta.mean(axis=1), delta.std(axis=1),
    ]).astype(np.float32)


def audio_to_feature(audio: np.ndarray, sr: int = 8000) -> np.ndarray:
    """Convenience: audio → MFCC → aggregated feature vector."""
    return aggregate(extract_mfcc(audio, sr=sr))
