"""
knn_classifier.py
-----------------
Hand-rolled K-Nearest-Neighbours classifier (numpy only).

Euclidean distance, majority vote over the ``k`` closest training vectors.
Feature vectors are produced upstream by ``feature_extraction.audio_to_feature``.
"""

from __future__ import annotations

from collections import Counter
from typing import List, Tuple

import numpy as np


class KNN:
    def __init__(self, k: int = 5, standardize: bool = True) -> None:
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = k
        self.standardize = standardize
        self.X_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    # ---------- training / persistence ------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNN":
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        if X.ndim != 2 or X.shape[0] != y.shape[0]:
            raise ValueError(f"shape mismatch: X={X.shape}, y={y.shape}")
        if self.standardize:
            self.mean_ = X.mean(axis=0)
            # Floor std to avoid divide-by-zero on constant features.
            self.std_ = np.maximum(X.std(axis=0), 1e-6)
            X = (X - self.mean_) / self.std_
        self.X_train = X.astype(np.float32, copy=False)
        self.y_train = y
        return self

    def _apply_scaling(self, x: np.ndarray) -> np.ndarray:
        if self.standardize and self.mean_ is not None and self.std_ is not None:
            return (x - self.mean_) / self.std_
        return x

    def save(self, path: str) -> None:
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("nothing to save — call fit() first")
        payload = dict(X=self.X_train, y=self.y_train, k=np.array([self.k]),
                       standardize=np.array([int(self.standardize)]))
        if self.standardize:
            payload["mean"] = self.mean_
            payload["std"] = self.std_
        np.savez(path, **payload)

    @classmethod
    def load(cls, path: str) -> "KNN":
        data = np.load(path, allow_pickle=False)
        standardize = bool(data["standardize"][0]) if "standardize" in data.files else False
        clf = cls(k=int(data["k"][0]), standardize=standardize)
        clf.X_train = data["X"]
        clf.y_train = data["y"]
        if standardize:
            clf.mean_ = data["mean"]
            clf.std_ = data["std"]
        return clf

    # ---------- inference -------------------------------------------------

    def _distances(self, x: np.ndarray) -> np.ndarray:
        assert self.X_train is not None
        diff = self.X_train - x[None, :]
        return np.sqrt(np.einsum("ij,ij->i", diff, diff))

    def predict(self, x: np.ndarray) -> Tuple[object, float]:
        """Return (label, confidence) where confidence is top-k agreement fraction."""
        assert self.y_train is not None
        x = self._apply_scaling(np.asarray(x, dtype=np.float32))
        d = self._distances(x)
        idx = np.argpartition(d, min(self.k, len(d) - 1))[: self.k]
        votes = self.y_train[idx]
        label, count = Counter(votes.tolist()).most_common(1)[0]
        return label, count / self.k

    def predict_with_neighbors(
        self, x: np.ndarray
    ) -> Tuple[object, float, List[Tuple[float, object]]]:
        """Like ``predict`` but also return the sorted top-k (distance, label) pairs."""
        assert self.y_train is not None
        x = self._apply_scaling(np.asarray(x, dtype=np.float32))
        d = self._distances(x)
        k = min(self.k, len(d))
        idx = np.argpartition(d, k - 1)[:k]
        order = idx[np.argsort(d[idx])]
        neighbors = [(float(d[i]), self.y_train[i]) for i in order]
        labels = [n[1] for n in neighbors]
        label, count = Counter(labels).most_common(1)[0]
        return label, count / k, neighbors

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        return np.array([self.predict(x)[0] for x in X])
