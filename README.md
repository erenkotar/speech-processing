# Speech Processing – LPC Analysis (CMPE532 Hmw#1)

Linear Predictive Coding (LPC) analysis of speech signals, implemented in Python.

## What This Does

| Step | Description |
|------|-------------|
| 1 | Load & normalize a WAV file |
| 2 | Pre-emphasis filter (α = 0.97) |
| 3 | Frame the signal (20 ms frames, 10 ms hop) |
| 4 | Apply Hamming window |
| 5 | Estimate LPC coefficients (Levinson-Durbin / librosa) |
| 6 | Compute LPC spectral envelope |
| 7 | Compute LPC prediction residual |
| 8 | Extract LPC Cepstral Coefficients (LPCC) |
| 9 | Voiced / Unvoiced classification (energy + ZCR) |
| 10 | Generate 9 analysis plots |

## Project Structure

```
├── main.py              # Full analysis pipeline (entry point)
├── lpc_utils.py         # DSP core: pre-emphasis, framing, LPC, LPCC, V/UV
├── plot_utils.py        # All visualization routines
├── generate_sample.py   # Generate synthetic speech WAV for testing
├── requirements.txt     # Python dependencies
├── .gitignore
└── README.md
```

## Quick Start

```bash
# 1. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate a test WAV file (if you don't have your own)
python generate_sample.py

# 4. Run the LPC analysis
python main.py                        # uses sample_speech.wav
python main.py --wav your_file.wav    # use your own recording
```

Output plots are saved to the `plots/` directory.

## Output Plots

| File | Content |
|------|---------|
| `0_summary.png` | 3-panel overview (waveform + voiced/unvoiced spectra) |
| `1_waveform.png` | Full speech signal |
| `2_frame_voiced.png` | Raw vs. windowed voiced frame |
| `3_frame_unvoiced.png` | Raw vs. windowed unvoiced frame |
| `4_lpc_spectrum_voiced.png` | LPC envelope vs. FFT spectrum |
| `5_voiced_vs_unvoiced.png` | Side-by-side V/UV spectral comparison |
| `6_residual.png` | LPC prediction residual (excitation signal) |
| `7_lpcc_heatmap.png` | LPCC coefficients over time |
| `8_lpc_over_time.png` | First 5 LPC coefficients over time |

## Configuration

Edit the `CONFIG` dictionary in `main.py`:

```python
CONFIG = {
    "frame_ms":   20.0,    # frame length (10–20 ms)
    "hop_ms":     10.0,    # hop size
    "lpc_order":  14,      # LPC order p
    "num_ceps":   13,      # number of LPCC coefficients
    "n_fft":      512,     # FFT size
    "window":     "hamming",
    "pre_emph":   0.97,
    "lpc_method": "librosa",  # 'librosa' or 'levinson'
}
```

## Dependencies

- Python 3.10+
- NumPy, SciPy, librosa, matplotlib, soundfile