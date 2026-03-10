"""
main.py
-------
Full LPC analysis pipeline for CMPE532 Hmw#1.

Usage:
    python main.py                      # uses bundled sample_speech.wav
    python main.py --wav path/to/file.wav

Pipeline:
    1. Load & normalize audio
    2. Pre-emphasis
    3. Framing (20 ms frames, 10 ms hop)
    4. Hamming windowing
    5. LPC estimation (Levinson-Durbin via librosa)
    6. LPC spectrum computation
    7. LPC residual
    8. LPCC extraction
    9. Visualizations saved to ./plots/
"""

import argparse
import os
import numpy as np
import scipy.io.wavfile as wav
import librosa

from lpc_utils import (
    pre_emphasis,
    frame_signal,
    apply_window,
    compute_lpc,
    compute_lpc_all_frames,
    compute_lpc_spectrum,
    compute_residual,
    lpc_to_cepstrum,
    compute_lpcc_all_frames,
    classify_frames,
)
from plot_utils import (
    plot_waveform,
    plot_frame,
    plot_lpc_vs_fft,
    plot_voiced_vs_unvoiced,
    plot_residual,
    plot_lpcc_heatmap,
    plot_lpc_coefficients_over_time,
    plot_summary,
)


# ── Configuration ────────────────────────────────────────────────────────────

CONFIG = {
    "frame_ms":   20.0,    # frame length in ms  (10–20 ms recommended)
    "hop_ms":     10.0,    # hop size in ms
    "lpc_order":  14,      # LPC order p  (rule of thumb: fs_kHz + 2)
    "num_ceps":   13,      # number of LPCC coefficients
    "n_fft":      512,     # FFT size for spectrum display
    "window":     "hamming",
    "pre_emph":   0.97,
    "lpc_method": "librosa",   # 'librosa' | 'levinson'
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_audio(path: str) -> tuple[np.ndarray, int]:
    """Load and normalize a WAV file to float32 mono, [-1, 1]."""
    sr, data = wav.read(path)

    # Convert stereo → mono
    if data.ndim == 2:
        data = data.mean(axis=1)

    # Normalize to float32 in [-1, 1]
    data = data.astype(np.float32)
    if data.max() > 1.0 or data.min() < -1.0:
        data = data / (np.iinfo(np.int16).max + 1)

    data = data / (np.max(np.abs(data)) + 1e-10)
    return data, sr


def pick_representative_frames(
    frames:        np.ndarray,
    windowed:      np.ndarray,
    is_voiced:     np.ndarray,
) -> tuple[int, int, np.ndarray, np.ndarray]:
    """
    Select one voiced and one unvoiced frame index for illustration.
    Returns (voiced_idx, unvoiced_idx, voiced_windowed, unvoiced_windowed).
    """
    voiced_indices   = np.where(is_voiced)[0]
    unvoiced_indices = np.where(~is_voiced)[0]

    # Pick frames from middle of each segment for stability
    v_idx = voiced_indices[len(voiced_indices) // 2]   if len(voiced_indices)   else 0
    u_idx = unvoiced_indices[len(unvoiced_indices) // 2] if len(unvoiced_indices) else -1

    return v_idx, u_idx, windowed[v_idx], windowed[u_idx]


# ── Main ─────────────────────────────────────────────────────────────────────

def run_lpc_analysis(wav_path: str, plots_dir: str = "plots") -> None:
    os.makedirs(plots_dir, exist_ok=True)
    cfg = CONFIG

    # ── 1. Load audio ────────────────────────────────────────────────────────
    print(f"[1/8] Loading audio: {wav_path}")
    signal, sr = load_audio(wav_path)
    duration   = len(signal) / sr
    print(f"      Sample rate: {sr} Hz  |  Duration: {duration:.2f} s  "
          f"|  Samples: {len(signal)}")

    # Adjust LPC order based on actual sample rate (rule of thumb: sr_kHz + 2)
    lpc_order = cfg["lpc_order"] if sr == 16000 else max(10, int(sr / 1000) + 2)
    print(f"      LPC order   : {lpc_order}")

    # ── 2. Pre-emphasis ──────────────────────────────────────────────────────
    print("[2/8] Applying pre-emphasis (α={})".format(cfg["pre_emph"]))
    signal_pe = pre_emphasis(signal, alpha=cfg["pre_emph"])

    # ── 3. Framing ───────────────────────────────────────────────────────────
    print(f"[3/8] Framing: {cfg['frame_ms']} ms frames, {cfg['hop_ms']} ms hop")
    frames, frame_len, hop_len = frame_signal(
        signal_pe, sr, frame_ms=cfg["frame_ms"], hop_ms=cfg["hop_ms"]
    )
    print(f"      Frame length: {frame_len} samples  |  "
          f"Hop: {hop_len} samples  |  Frames: {frames.shape[0]}")

    # ── 4. Windowing ─────────────────────────────────────────────────────────
    print(f"[4/8] Applying {cfg['window']} window")
    windowed = apply_window(frames, window_type=cfg["window"])

    # ── 5. LPC estimation ────────────────────────────────────────────────────
    print(f"[5/8] Computing LPC (order={lpc_order}, method={cfg['lpc_method']})")
    all_a, all_gain = compute_lpc_all_frames(windowed, lpc_order, cfg["lpc_method"])
    print(f"      LPC matrix shape: {all_a.shape}")

    # ── 6. LPC spectrum ──────────────────────────────────────────────────────
    print("[6/8] Computing LPC spectra")
    is_voiced = classify_frames(frames)
    print(f"      Voiced frames : {is_voiced.sum()}  |  "
          f"Unvoiced frames: {(~is_voiced).sum()}")

    v_idx, u_idx, v_frame, u_frame = pick_representative_frames(
        frames, windowed, is_voiced
    )
    v_freqs, v_lpc_dB = compute_lpc_spectrum(all_a[v_idx], all_gain[v_idx], cfg["n_fft"])
    u_freqs, u_lpc_dB = compute_lpc_spectrum(all_a[u_idx], all_gain[u_idx], cfg["n_fft"])
    print(f"      Representative voiced frame   : #{v_idx}")
    print(f"      Representative unvoiced frame : #{u_idx}")

    # ── 7. LPC residual ──────────────────────────────────────────────────────
    print("[7/8] Computing LPC residual for voiced frame")
    v_residual = compute_residual(v_frame, all_a[v_idx])

    # ── 8. LPCC ──────────────────────────────────────────────────────────────
    print(f"[8/8] Computing LPCC (num_ceps={cfg['num_ceps']})")
    lpcc_matrix = compute_lpcc_all_frames(all_a, all_gain, num_ceps=cfg["num_ceps"])
    print(f"      LPCC matrix shape: {lpcc_matrix.shape}")

    # ── Sanity check: print stats for middle voiced frame ────────────────────
    print("\n── LPCC for voiced frame #{} ──".format(v_idx))
    lpcc_v = lpc_to_cepstrum(all_a[v_idx], all_gain[v_idx], cfg["num_ceps"])
    print("  ", np.round(lpcc_v, 4))
    print("── LPC coefficients ──")
    print("  ", np.round(all_a[v_idx], 4))

    # ── Plots ────────────────────────────────────────────────────────────────
    print("\nGenerating plots …")

    plot_waveform(signal, sr, title="Speech Signal (pre-emphasis applied)",
                  save_path=f"{plots_dir}/1_waveform.png")

    plot_frame(frames[v_idx], v_frame, sr, v_idx, label="(Voiced)",
               save_path=f"{plots_dir}/2_frame_voiced.png")

    plot_frame(frames[u_idx], u_frame, sr, u_idx, label="(Unvoiced)",
               save_path=f"{plots_dir}/3_frame_unvoiced.png")

    plot_lpc_vs_fft(v_frame, v_freqs, v_lpc_dB, sr,
                    frame_label=f"(Voiced, frame #{v_idx})", n_fft=cfg["n_fft"],
                    save_path=f"{plots_dir}/4_lpc_spectrum_voiced.png")

    plot_voiced_vs_unvoiced(
        v_frame, u_frame,
        v_freqs, v_lpc_dB,
        u_freqs, u_lpc_dB,
        sr, n_fft=cfg["n_fft"],
        save_path=f"{plots_dir}/5_voiced_vs_unvoiced.png"
    )

    plot_residual(v_frame, v_residual, sr, frame_label=f"(Voiced, frame #{v_idx})",
                  save_path=f"{plots_dir}/6_residual.png")

    plot_lpcc_heatmap(lpcc_matrix, hop_ms=cfg["hop_ms"],
                      save_path=f"{plots_dir}/7_lpcc_heatmap.png")

    plot_lpc_coefficients_over_time(all_a, hop_ms=cfg["hop_ms"],
                                    save_path=f"{plots_dir}/8_lpc_over_time.png")

    plot_summary(
        signal_pe, sr,
        v_frame, u_frame,
        v_freqs, v_lpc_dB,
        u_freqs, u_lpc_dB,
        lpcc_matrix, hop_ms=cfg["hop_ms"], n_fft=cfg["n_fft"],
        save_path=f"{plots_dir}/0_summary.png",
    )

    print(f"\n✓ All plots saved to '{plots_dir}/'")
    print(f"  0_summary.png          – 3-panel overview")
    print(f"  1_waveform.png         – full signal")
    print(f"  2_frame_voiced.png     – raw vs windowed voiced frame")
    print(f"  3_frame_unvoiced.png   – raw vs windowed unvoiced frame")
    print(f"  4_lpc_spectrum_voiced.png – LPC envelope vs FFT")
    print(f"  5_voiced_vs_unvoiced.png  – side-by-side comparison")
    print(f"  6_residual.png         – LPC residual (excitation)")
    print(f"  7_lpcc_heatmap.png     – LPCC over time")
    print(f"  8_lpc_over_time.png    – LPC coefficients over time")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LPC Analysis – CMPE532 Hmw#1")
    parser.add_argument(
        "--wav", type=str, default="sample_speech.wav",
        help="Path to input WAV file (default: sample_speech.wav)"
    )
    parser.add_argument(
        "--plots_dir", type=str, default="plots",
        help="Directory to save output plots (default: plots/)"
    )
    args = parser.parse_args()

    run_lpc_analysis(wav_path=args.wav, plots_dir=args.plots_dir)
