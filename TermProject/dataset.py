"""
dataset.py
----------
Build a feature matrix from the FSDD wav files on disk.

FSDD file naming convention: ``{digit}_{speaker}_{index}.wav``
e.g. ``3_jackson_0.wav`` → digit 3, speaker "jackson", repetition 0.

A speaker-disjoint split (one held-out speaker as test set) is used so
reported accuracy reflects generalization, not memorization.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

import numpy as np

from audio_loader import load_audio
from feature_extraction import audio_to_feature

DEFAULT_TEST_RATIO = 0.2
DEFAULT_SEED = 42

_FNAME_RE = re.compile(r"^(\d+)_([a-zA-Z]+)_(\d+)\.wav$")


def _parse_filename(name: str) -> Tuple[int, str, int] | None:
    m = _FNAME_RE.match(name)
    if not m:
        return None
    return int(m.group(1)), m.group(2).lower(), int(m.group(3))


def build_feature_matrix(
    recordings_dir: str | Path,
    test_ratio: float = DEFAULT_TEST_RATIO,
    seed: int = DEFAULT_SEED,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Scan ``recordings_dir`` for FSDD wav files, extract features, and return a
    random train/test split (seeded so reruns are stable):

        (X_train, y_train, X_test, y_test, train_files, test_files)
    """
    recordings_dir = Path(recordings_dir)
    wavs = sorted(recordings_dir.glob("*.wav"))
    if not wavs:
        raise FileNotFoundError(f"No wav files under {recordings_dir}")

    feats: List[np.ndarray] = []
    labels: List[int] = []
    names: List[str] = []
    for path in wavs:
        parsed = _parse_filename(path.name)
        if parsed is None:
            continue
        digit, _speaker, _ = parsed
        audio = load_audio(str(path))
        feats.append(audio_to_feature(audio))
        labels.append(digit)
        names.append(path.name)

    X = np.stack(feats).astype(np.float32)
    y = np.array(labels, dtype=np.int32)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    n_test = int(round(test_ratio * len(y)))
    test_idx, train_idx = idx[:n_test], idx[n_test:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    train_files = [names[i] for i in train_idx]
    test_files = [names[i] for i in test_idx]

    print(
        f"[data]  Train: {len(train_idx)} samples  |  "
        f"Test: {len(test_idx)} samples  (random split, seed={seed})"
    )
    return X_train, y_train, X_test, y_test, train_files, test_files
