# Speech Processing – LPC Analysis (CMPE532 Hmw#1)

Linear Predictive Coding (LPC) analysis of speech signals, implemented in Python.

## What This Does

| Step | Description |
|------|-------------|
| 1 | Load & normalize a WAV file |
| 2 | Pre-emphasis filter (α = 0.97) |
| 3 | Frame the signal (20 ms frames, 10 ms hop) |
| 4 | Apply Hamming window |
| 5 | Estimate LPC coefficients (via librosa) |
| 6 | Compute LPC spectral envelope |
| 7 | Extract LPC Cepstral Coefficients (LPCC) |

## Project Structure

```
├── main.py              # Analysis pipeline (entry point)
├── lpc_utils.py         # DSP core: pre-emphasis, framing, LPC, LPCC
├── plot_utils.py        # Visualization (waveform, spectrum, LPCC heatmap)
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

# 3. Run the LPC analysis
python main.py                        # uses my_recording.wav
python main.py --wav your_file.wav    # use a different recording
```

Output plots are saved to the `plots/` directory.

## Output Plots

| File | Content |
|------|---------|
| `1_waveform.png` | Full speech signal |
| `2_lpc_spectrum.png` | LPC envelope vs. FFT spectrum |
| `3_lpcc_heatmap.png` | LPCC coefficients over time |

## Configuration

Edit the `CONFIG` dictionary in `main.py`:

```python
CONFIG = {
    "frame_ms":  20.0,   # frame length (10–20 ms)
    "hop_ms":    10.0,   # hop size
    "lpc_order": 14,     # LPC order p
    "num_ceps":  13,     # number of LPCC coefficients
    "n_fft":     512,    # FFT size
    "pre_emph":  0.97,
}
```

## Dependencies

- Python 3.10+
- NumPy, SciPy, librosa, matplotlib, soundfile