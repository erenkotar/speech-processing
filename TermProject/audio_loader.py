"""
audio_loader.py
---------------
Load + preprocess a wav file into a 1-D float32 array at the target sample rate.

Pipeline:
    1. Read with soundfile (any sample rate, mono or stereo).
    2. Downmix to mono.
    3. Resample to ``target_sr`` (8 kHz to match FSDD) if needed.
    4. Peak-normalize to 0.9.
    5. Trim leading/trailing silence (librosa.effects.trim).
"""

from __future__ import annotations

import numpy as np
import soundfile as sf
import librosa

TARGET_SR = 8000


def _to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio
    return audio.mean(axis=1)


def _peak_normalize(audio: np.ndarray, target_peak: float = 0.9) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1e-12:
        audio = audio * (target_peak / peak)
    return audio


def load_audio(path: str, target_sr: int = TARGET_SR, top_db: float = 25.0) -> np.ndarray:
    """Load a wav file, return mono float32 audio at ``target_sr`` with silence trimmed."""
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    audio = _to_mono(audio).astype(np.float32, copy=False)

    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    audio = _peak_normalize(audio)

    # Trim silence — keep ≥1 frame so feature extraction never sees empty input.
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    if trimmed.size < 256:
        trimmed = audio  # fall back to the un-trimmed signal

    return trimmed.astype(np.float32, copy=False)
