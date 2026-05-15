"""
main.py
-------
Speech-to-Text via KNN — entry point.

Usage
-----
    # Train (downloads FSDD on first run, prints test accuracy, caches model)
    python main.py --train

    # Predict the digit spoken in a wav file
    python main.py path/to/audio.wav

    # Predict + write output_plot.png
    python main.py path/to/audio.wav --plot

    # Re-evaluate the cached model on the held-out test split
    python main.py --evaluate

    # Override neighbours
    python main.py path/to/audio.wav --k 7

Pipeline stages
---------------
  1. Load wav (audio_loader)
  2. Extract MFCC + aggregate to fixed-length vector (feature_extraction)
  3. KNN predict against trained reference set (knn_classifier)
  4. Print result; optionally plot waveform + MFCC + neighbors
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from audio_loader import load_audio, TARGET_SR
from feature_extraction import audio_to_feature, extract_mfcc
from knn_classifier import KNN
from dataset import build_feature_matrix
from download_dataset import ensure_dataset, RECORDINGS_DIR


HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "model.npz"
PLOT_PATH = HERE / "output_plot.png"
DEFAULT_K = 5


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(k: int) -> None:
    print("[stage] 1/4  Ensure dataset")
    ensure_dataset()

    print("[stage] 2/4  Build feature matrix")
    X_train, y_train, X_test, y_test, _, _ = build_feature_matrix(RECORDINGS_DIR)

    print(f"[stage] 3/4  Fit KNN  (k={k}, n_features={X_train.shape[1]})")
    clf = KNN(k=k).fit(X_train, y_train)

    print("[stage] 4/4  Evaluate on held-out test split")
    preds = clf.predict_batch(X_test)
    acc = float((preds == y_test).mean())
    print(f"[eval]  Test accuracy: {acc:.3f}  ({(preds == y_test).sum()}/{len(y_test)})")
    print_confusion(y_test, preds)

    clf.save(str(MODEL_PATH))
    print(f"[save]  Wrote {MODEL_PATH}")


def print_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    print("[eval]  Confusion matrix (rows=true, cols=predicted):")
    header = "       " + " ".join(f"{l:>4}" for l in labels)
    print(header)
    for t in labels:
        row = [int(((y_true == t) & (y_pred == p)).sum()) for p in labels]
        print(f"   {t:>3} " + " ".join(f"{v:>4}" for v in row))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate() -> None:
    clf = _load_model_or_exit()
    print("[eval]  Re-running on held-out test split")
    ensure_dataset()
    _, _, X_test, y_test, _, _ = build_feature_matrix(RECORDINGS_DIR)
    preds = clf.predict_batch(X_test)
    acc = float((preds == y_test).mean())
    print(f"[eval]  Test accuracy: {acc:.3f}  ({(preds == y_test).sum()}/{len(y_test)})")
    print_confusion(y_test, preds)


# ---------------------------------------------------------------------------
# Prediction (the main user-facing path)
# ---------------------------------------------------------------------------

def predict_file(audio_path: str, k: int | None, do_plot: bool) -> None:
    clf = _load_model_or_exit()
    if k is not None:
        clf.k = k

    print(f"[stage] 1/3  Load {audio_path}")
    audio = load_audio(audio_path)
    print(f"[stage] 2/3  Extract MFCC + aggregate")
    mfcc = extract_mfcc(audio, sr=TARGET_SR)
    feat = audio_to_feature(audio, sr=TARGET_SR)

    print(f"[stage] 3/3  KNN predict  (k={clf.k})")
    label, confidence, neighbors = clf.predict_with_neighbors(feat)

    print()
    print(f"  >>> Predicted digit: {label}   (confidence {confidence:.0%})")
    print()
    print("  Top neighbours:")
    for i, (dist, lbl) in enumerate(neighbors, 1):
        marker = "*" if lbl == label else " "
        print(f"   {marker} #{i}  label={lbl}  distance={dist:.3f}")

    if do_plot:
        from visualization import plot_prediction
        plot_prediction(
            audio=audio,
            sr=TARGET_SR,
            mfcc=mfcc,
            neighbors=neighbors,
            prediction=label,
            confidence=confidence,
            save_png=str(PLOT_PATH),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_model_or_exit() -> KNN:
    if not MODEL_PATH.exists():
        print(f"[error] No trained model at {MODEL_PATH}. Run: python main.py --train",
              file=sys.stderr)
        sys.exit(1)
    return KNN.load(str(MODEL_PATH))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Speech-to-Text (spoken digits) via simple KNN.",
    )
    parser.add_argument(
        "audio", nargs="?", default=None,
        help="Path to a .wav file to classify. Omit when using --train/--evaluate.",
    )
    parser.add_argument("--train", action="store_true",
                        help="Download dataset (if needed), train KNN, save model.")
    parser.add_argument("--evaluate", action="store_true",
                        help="Reload model and report accuracy on the held-out split.")
    parser.add_argument("-k", "--k", type=int, default=None,
                        help=f"Number of neighbours (default {DEFAULT_K} when training).")
    parser.add_argument("--plot", action="store_true",
                        help="Save output_plot.png with waveform, MFCC, and neighbours.")
    args = parser.parse_args(argv)

    if args.train:
        train(k=args.k if args.k is not None else DEFAULT_K)
        return 0
    if args.evaluate:
        evaluate()
        return 0
    if args.audio is None:
        parser.error("Provide an audio path, or use --train / --evaluate.")
    predict_file(args.audio, k=args.k, do_plot=args.plot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
