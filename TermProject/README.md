# Spoken Digit Recognition (KNN)

A simple speech-to-text application that recognizes spoken digits (0–9) using MFCC features and a hand-rolled K-Nearest-Neighbours classifier. Trained on the [Free Spoken Digit Dataset](https://github.com/Jakobovski/free-spoken-digit-dataset).

## Install

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Use

```bash
# Download dataset + train + cache model (run once)
python main.py --train

# Predict the digit in a wav file
python main.py path/to/audio.wav

# Predict + save a diagnostic plot (waveform, MFCC, top-K neighbours)
python main.py path/to/audio.wav --plot

# Re-evaluate on the held-out test split
python main.py --evaluate

# Try the bundled sample
python main.py sample_speech.wav
```

## Files

- `main.py` — CLI entry point
- `audio_loader.py` — load, resample, normalize, trim silence
- `feature_extraction.py` — MFCC + delta features, aggregation
- `knn_classifier.py` — hand-rolled KNN (numpy)
- `dataset.py` — build feature matrix from FSDD
- `download_dataset.py` — fetch FSDD
- `visualization.py` — diagnostic plot
