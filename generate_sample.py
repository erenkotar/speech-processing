"""
generate_sample.py
------------------
Generate a synthetic speech-like WAV file for testing the LPC pipeline
when no real recording is available.

The signal contains:
  - A voiced segment   (periodic glottal pulses filtered through formants)
  - An unvoiced segment (white noise filtered through a different formant shape)

Usage:
    python generate_sample.py              # creates sample_speech.wav (16 kHz, ~1 s)
    python generate_sample.py --out my.wav --sr 8000 --duration 2.0
"""

import argparse
import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import lfilter


def make_glottal_pulse_train(f0: float, sr: int, duration: float) -> np.ndarray:
    """Generate a simple glottal pulse train at fundamental frequency f0."""
    n_samples = int(sr * duration)
    period = int(sr / f0)
    train = np.zeros(n_samples)
    for i in range(0, n_samples, period):
        train[i] = 1.0
    return train


def make_formant_filter(formants: list[tuple[float, float]], sr: int):
    """
    Create a cascade of second-order resonators (formant filters).
    Each formant is (center_freq_hz, bandwidth_hz).
    Returns (b, a) coefficients for scipy.signal.lfilter.
    """
    b_all = np.array([1.0])
    a_all = np.array([1.0])
    for fc, bw in formants:
        r = np.exp(-np.pi * bw / sr)
        theta = 2 * np.pi * fc / sr
        a_k = np.array([1.0, -2 * r * np.cos(theta), r ** 2])
        b_k = np.array([1.0 - r])
        b_all = np.convolve(b_all, b_k)
        a_all = np.convolve(a_all, a_k)
    return b_all, a_all


def generate_sample(sr: int = 16000, duration: float = 1.0) -> np.ndarray:
    """
    Generate a ~duration-second synthetic speech-like signal.
    First half: voiced (pulse train → formant filter)
    Second half: unvoiced (noise → different filter)
    """
    half = duration / 2.0

    # ── Voiced segment ──
    pulse_train = make_glottal_pulse_train(f0=120.0, sr=sr, duration=half)
    # Typical vowel /a/ formants
    voiced_formants = [(730, 90), (1090, 110), (2440, 170)]
    b_v, a_v = make_formant_filter(voiced_formants, sr)
    voiced = lfilter(b_v, a_v, pulse_train)

    # ── Unvoiced segment (fricative-like) ──
    noise = np.random.randn(int(sr * half)) * 0.3
    unvoiced_formants = [(3500, 500), (5000, 600)]
    b_u, a_u = make_formant_filter(unvoiced_formants, sr)
    unvoiced = lfilter(b_u, a_u, noise)

    # Concatenate and normalize
    signal = np.concatenate([voiced, unvoiced])
    signal = signal / (np.max(np.abs(signal)) + 1e-10) * 0.9

    return signal.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic speech WAV")
    parser.add_argument("--out", type=str, default="sample_speech.wav",
                        help="Output WAV file path")
    parser.add_argument("--sr", type=int, default=16000,
                        help="Sample rate in Hz (default: 16000)")
    parser.add_argument("--duration", type=float, default=1.0,
                        help="Duration in seconds (default: 1.0)")
    args = parser.parse_args()

    signal = generate_sample(sr=args.sr, duration=args.duration)
    wav.write(args.out, args.sr, (signal * 32767).astype(np.int16))
    print(f"✓ Saved {args.out}  ({args.sr} Hz, {args.duration:.1f} s, "
          f"{len(signal)} samples)")


if __name__ == "__main__":
    main()
