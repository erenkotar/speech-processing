"""
main.py
-------
LPC analysis pipeline for CMPE532 Hmw#1.

Usage:
    python main.py                          # uses my_recording.wav
    python main.py --wav path/to/file.wav

Pipeline:
    1. Load & normalize audio
    2. Pre-emphasis
    3. Framing (20 ms frames, 10 ms hop)
    4. Hamming windowing
    5. LPC estimation (via librosa)
    6. LPC spectrum
    7. LPCC extraction
    8. Plots saved to ./plots/
"""

import argparse
import os
import numpy as np
import scipy.io.wavfile as wav

from lpc_utils import (
    pre_emphasis,
    frame_signal,
    apply_window,
    compute_lpc_all_frames,
    compute_lpc_spectrum,
    lpc_to_cepstrum,
    compute_lpcc_all_frames,
)
from plot_utils import (
    plot_waveform,
    plot_lpc_vs_fft,
    plot_lpcc_heatmap,
)


# ── Configuration ────────────────────────────────────────────────────────────

CONFIG = {
    "frame_ms":  20.0,   # frame length in ms  (10–20 ms)
    "hop_ms":    10.0,   # hop size in ms
    "lpc_order": 14,     # LPC order p  (rule of thumb: fs_kHz + 2)
    "num_ceps":  13,     # number of LPCC coefficients
    "n_fft":     512,    # FFT size for spectrum display
    "pre_emph":  0.97,
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_audio(path: str) -> tuple[np.ndarray, int]:
    """Load and normalize a WAV file to float32 mono, [-1, 1]."""
    sr, data = wav.read(path)

    if data.ndim == 2:
        data = data.mean(axis=1)

    data = data.astype(np.float32)
    if data.max() > 1.0 or data.min() < -1.0:
        data = data / (np.iinfo(np.int16).max + 1)

    data = data / (np.max(np.abs(data)) + 1e-10)
    return data, sr


# ── Main ─────────────────────────────────────────────────────────────────────

def run_lpc_analysis(wav_path: str, plots_dir: str = "plots") -> None:
    os.makedirs(plots_dir, exist_ok=True)
    cfg = CONFIG

    # 1. Load audio
    print(f"[1/7] Loading audio: {wav_path}")
    signal, sr = load_audio(wav_path)
    duration = len(signal) / sr
    print(f"      Sample rate: {sr} Hz  |  Duration: {duration:.2f} s  "
          f"|  Samples: {len(signal)}")

    lpc_order = cfg["lpc_order"] if sr == 16000 else max(10, int(sr / 1000) + 2)
    print(f"      LPC order: {lpc_order}")

    # 2. Pre-emphasis
    print(f"[2/7] Applying pre-emphasis (α={cfg['pre_emph']})")
    signal_pe = pre_emphasis(signal, alpha=cfg["pre_emph"])

    # 3. Framing
    print(f"[3/7] Framing: {cfg['frame_ms']} ms frames, {cfg['hop_ms']} ms hop")
    frames, frame_len, hop_len = frame_signal(
        signal_pe, sr, frame_ms=cfg["frame_ms"], hop_ms=cfg["hop_ms"]
    )
    print(f"      Frame length: {frame_len} samples  |  "
          f"Hop: {hop_len} samples  |  Frames: {frames.shape[0]}")

    # 4. Windowing
    print("[4/7] Applying Hamming window")
    windowed = apply_window(frames)

    # 5. LPC estimation
    print(f"[5/7] Computing LPC (order={lpc_order})")
    all_a, all_gain = compute_lpc_all_frames(windowed, lpc_order)
    print(f"      LPC matrix shape: {all_a.shape}")

    # 6. LPC spectrum (pick a representative middle frame)
    mid = frames.shape[0] // 2
    print(f"[6/7] Computing LPC spectrum (frame #{mid})")
    freqs, lpc_dB = compute_lpc_spectrum(all_a[mid], all_gain[mid], cfg["n_fft"])

    # 7. LPCC
    print(f"[7/7] Computing LPCC (num_ceps={cfg['num_ceps']})")
    lpcc_matrix = compute_lpcc_all_frames(all_a, all_gain, num_ceps=cfg["num_ceps"])
    print(f"      LPCC matrix shape: {lpcc_matrix.shape}")

    # Print sample values
    print(f"\n── LPC coefficients (frame #{mid}) ──")
    print("  ", np.round(all_a[mid], 4))
    lpcc_mid = lpc_to_cepstrum(all_a[mid], all_gain[mid], cfg["num_ceps"])
    print(f"── LPCC (frame #{mid}) ──")
    print("  ", np.round(lpcc_mid, 4))

    # ── Plots ────────────────────────────────────────────────────────────────
    print("\nGenerating plots …")

    plot_waveform(signal, sr, title="Speech Signal",
                  save_path=f"{plots_dir}/1_waveform.png")

    plot_lpc_vs_fft(windowed[mid], freqs, lpc_dB, sr,
                    frame_label=f"(frame #{mid})", n_fft=cfg["n_fft"],
                    save_path=f"{plots_dir}/2_lpc_spectrum.png")

    plot_lpcc_heatmap(lpcc_matrix, hop_ms=cfg["hop_ms"],
                      save_path=f"{plots_dir}/3_lpcc_heatmap.png")

    print(f"\n✓ All plots saved to '{plots_dir}/'")
    print(f"  1_waveform.png     – full signal")
    print(f"  2_lpc_spectrum.png – LPC envelope vs FFT")
    print(f"  3_lpcc_heatmap.png – LPCC over time")


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LPC Analysis – CMPE532 Hmw#1")
    parser.add_argument(
        "--wav", type=str, default="my_recording.wav",
        help="Path to input WAV file (default: my_recording.wav)"
    )
    parser.add_argument(
        "--plots_dir", type=str, default="plots",
        help="Directory to save output plots (default: plots/)"
    )
    args = parser.parse_args()

    run_lpc_analysis(wav_path=args.wav, plots_dir=args.plots_dir)
