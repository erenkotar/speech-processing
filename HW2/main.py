"""
main.py
-------
LPC-based Text-to-Speech Pipeline — entry point.

Usage
-----
    python main.py                          # synthesise the default sentence
    python main.py "hello world"            # synthesise custom text
    python main.py "hello world" --rate 0.8 # faster speech
    python main.py --no-plot                # skip the plot window

Pipeline stages
---------------
  1. Build phoneme inventory         (phoneme_inventory)
  2. Text normalisation + G2P        (text_frontend)
  3. Prosody assignment              (prosody)
  4. Frame-by-frame LPC synthesis    (synthesis)
  5. Normalise + save .wav           (soundfile)
  6. Plot waveform + spectrogram     (matplotlib)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from phoneme_inventory import build_inventory, FS, P
from text_frontend      import text_to_phonemes
from prosody            import assign_prosody
from synthesis          import synthesize_utterance
from lpc_analysis       import lpc_analyze_frame, lpc_spectral_envelope


# ---------------------------------------------------------------------------
# Audio utilities
# ---------------------------------------------------------------------------

def normalize_audio(audio: np.ndarray, target_peak: float = 0.90) -> np.ndarray:
    """Peak-normalise audio to target_peak."""
    peak = np.max(np.abs(audio))
    if peak > 1e-12:
        audio = audio * (target_peak / peak)
    return audio


def save_wav(audio: np.ndarray, fs: int, path: str) -> None:
    audio = normalize_audio(audio)
    sf.write(path, audio.astype(np.float32), fs)
    duration = len(audio) / fs
    print(f"[save]  Wrote {path}  ({len(audio)} samples, {duration:.2f} s)")


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_results(
    audio: np.ndarray,
    fs: int,
    phoneme_sequence: list,
    prosody_sequence: list,
    save_png: str = "output_plot.png",
) -> None:
    """
    Three-panel plot:
      1. Waveform with phoneme boundary markers
      2. Spectrogram
      3. LPC spectral envelope of a mid-utterance frame
    """
    t_axis = np.linspace(0, len(audio) / fs, len(audio))

    fig = plt.figure(figsize=(12, 8))
    gs  = gridspec.GridSpec(3, 1, height_ratios=[1.5, 2, 1.5], hspace=0.45)

    # ---- Panel 1: Waveform ------------------------------------------------
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(t_axis, audio, linewidth=0.5, color="#2a6ebb")
    ax1.set_title("Synthesised Waveform", fontsize=11)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.set_xlim(t_axis[0], t_axis[-1])

    # Mark phoneme boundaries
    boundary = 0.0
    for item in prosody_sequence:
        boundary += item["duration_ms"] / 1000.0
        ax1.axvline(boundary, color="red", linewidth=0.6, alpha=0.6)

    # Annotate phoneme labels at centres
    cursor = 0.0
    for item in prosody_sequence:
        dur = item["duration_ms"] / 1000.0
        mid = cursor + dur / 2.0
        ax1.text(mid, np.max(np.abs(audio)) * 0.85, item["phoneme"],
                 ha="center", fontsize=6.5, color="darkred")
        cursor += dur

    # ---- Panel 2: Spectrogram ---------------------------------------------
    ax2 = fig.add_subplot(gs[1])
    ax2.specgram(audio, Fs=fs, NFFT=256, noverlap=200, cmap="magma", scale="dB")
    ax2.set_title("Spectrogram", fontsize=11)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Frequency (Hz)")
    ax2.set_xlim(0, len(audio) / fs)

    # ---- Panel 3: LPC spectral envelope of a mid frame --------------------
    ax3 = fig.add_subplot(gs[2])
    mid_sample = len(audio) // 2
    frame_len  = 320              # 20 ms at 16 kHz
    start      = max(0, mid_sample - frame_len // 2)
    frame      = audio[start : start + frame_len]
    if len(frame) == frame_len:
        a, gain, _ = lpc_analyze_frame(frame, p=P)
        # Raw DFT spectrum of the frame
        NFFT = 512
        S    = np.abs(np.fft.rfft(frame * np.hamming(len(frame)), n=NFFT))
        freq = np.fft.rfftfreq(NFFT, 1.0 / fs)
        H    = lpc_spectral_envelope(a, gain, n_fft=NFFT)

        ax3.plot(freq, 20 * np.log10(S + 1e-12),
                 color="steelblue", linewidth=0.7, alpha=0.6, label="DFT spectrum")
        ax3.plot(freq, 20 * np.log10(H + 1e-12),
                 color="crimson", linewidth=1.5, label=f"LPC envelope (p={P})")
        ax3.set_title("LPC Spectral Envelope (mid-utterance frame)", fontsize=11)
        ax3.set_xlabel("Frequency (Hz)")
        ax3.set_ylabel("Magnitude (dB)")
        ax3.legend(fontsize=8)
        ax3.set_xlim(0, fs / 2)

    plt.savefig(save_png, dpi=150, bbox_inches="tight")
    print(f"[plot]  Saved {save_png}")
    plt.show()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    text: str,
    rate: float = 1.0,
    out_wav: str = "output.wav",
    out_png: str = "output_plot.png",
    show_plot: bool = True,
    seed: int | None = 42,
) -> np.ndarray:
    """
    Full LPC TTS pipeline.

    Parameters
    ----------
    text      : input English text
    rate      : speaking-rate scale (1.0 = neutral, <1 = faster, >1 = slower)
    out_wav   : output .wav path
    out_png   : output plot .png path
    show_plot : whether to display the matplotlib window
    seed      : random seed for reproducible unvoiced excitation

    Returns
    -------
    audio : synthesised waveform as float64 array
    """
    rng = np.random.default_rng(seed)

    print(f"\n{'='*55}")
    print(f"  LPC Text-to-Speech Pipeline")
    print(f"{'='*55}")
    print(f"  Input  : \"{text}\"")
    print(f"  fs     : {FS} Hz    |  LPC order : {P}")
    print(f"  Rate   : {rate}x")

    # Stage 1 — Inventory
    print("\n[1/4] Building phoneme inventory ...")
    inventory = build_inventory(fs=FS, p=P)
    print(f"      {len(inventory)} phonemes loaded")

    # Stage 2 — G2P
    print("[2/4] Text → phonemes ...")
    phonemes = text_to_phonemes(text)
    if not phonemes:
        print("      ERROR: no phonemes produced. Check word list.")
        sys.exit(1)
    print(f"      {phonemes}")

    # Stage 3 — Prosody
    print("[3/4] Assigning prosody ...")
    prosody = assign_prosody(phonemes, inventory, rate_scale=rate)
    for p_item in prosody:
        print(f"      {p_item['phoneme']:4s}  {p_item['duration_ms']:6.1f} ms   "
              f"F0={p_item['f0']:5.1f} Hz")

    # Stage 4 — Synthesis
    print("[4/4] Synthesising speech ...")
    audio = synthesize_utterance(prosody, inventory, FS, rng=rng)
    print(f"      Generated {len(audio)} samples  ({len(audio)/FS:.2f} s)")

    # Save
    save_wav(audio, FS, out_wav)

    # Plot
    if show_plot:
        plot_results(audio, FS, phonemes, prosody, save_png=out_png)

    return audio


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="LPC Text-to-Speech — basic pipeline demo"
    )
    parser.add_argument(
        "text",
        nargs="?",
        default="hello world",
        help='Input text (default: "hello world")',
    )
    parser.add_argument(
        "--rate", type=float, default=1.0,
        help="Speaking rate scale (default 1.0 = neutral; 0.8 = faster)"
    )
    parser.add_argument(
        "--out", default="output.wav",
        help="Output .wav filename (default: output.wav)"
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip the matplotlib plot"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible unvoiced excitation"
    )
    args = parser.parse_args()

    run_pipeline(
        text=args.text,
        rate=args.rate,
        out_wav=args.out,
        show_plot=not args.no_plot,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
