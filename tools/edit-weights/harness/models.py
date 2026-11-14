"""Tiny numpy MLP for weight-space experiments (no torch)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

try:
    from sklearn.datasets import load_digits
except ImportError:  # pragma: no cover
    load_digits = None


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=-1, keepdims=True)


def one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((y.shape[0], n_classes), dtype=np.float64)
    out[np.arange(y.shape[0]), y] = 1.0
    return out


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(np.float64)


class MLP:
    """Fully-connected MLP with ReLU hidden layers and linear logits."""

    def __init__(
        self,
        layer_sizes: Sequence[int],
        *,
        seed: int = 0,
        weights: dict[str, np.ndarray] | None = None,
    ):
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes needs at least input and output")
        self.layer_sizes = list(layer_sizes)
        self.seed = int(seed)
        self.weights: dict[str, np.ndarray] = {}
        if weights is not None:
            self.weights = {k: np.asarray(v, dtype=np.float64) for k, v in weights.items()}
            self._validate_shapes()
        else:
            self._init_weights()

    def _init_weights(self) -> None:
        rng = np.random.default_rng(self.seed)
        for i in range(len(self.layer_sizes) - 1):
            fan_in, fan_out = self.layer_sizes[i], self.layer_sizes[i + 1]
            # He-ish init for ReLU
            scale = np.sqrt(2.0 / fan_in)
            self.weights[f"W{i}"] = rng.normal(0.0, scale, size=(fan_out, fan_in)).astype(np.float64)
            self.weights[f"b{i}"] = np.zeros(fan_out, dtype=np.float64)

    def _validate_shapes(self) -> None:
        for i in range(len(self.layer_sizes) - 1):
            fan_in, fan_out = self.layer_sizes[i], self.layer_sizes[i + 1]
            W = self.weights[f"W{i}"]
            b = self.weights[f"b{i}"]
            if W.shape != (fan_out, fan_in):
                raise ValueError(f"W{i} expected {(fan_out, fan_in)}, got {W.shape}")
            if b.shape != (fan_out,):
                raise ValueError(f"b{i} expected {(fan_out,)}, got {b.shape}")

    @property
    def n_layers(self) -> int:
        return len(self.layer_sizes) - 1

    def layer_names(self) -> list[str]:
        names: list[str] = []
        for i in range(self.n_layers):
            names.extend([f"W{i}", f"b{i}"])
        return names

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Return logits of shape (N, n_out)."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[None, :]
        a = X
        for i in range(self.n_layers):
            W = self.weights[f"W{i}"]
            b = self.weights[f"b{i}"]
            z = a @ W.T + b
            if i < self.n_layers - 1:
                a = relu(z)
            else:
                a = z
        return a

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X).argmax(axis=-1)

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        epochs: int = 50,
        lr: float = 0.05,
        batch_size: int = 64,
        seed: int | None = None,
        verbose: bool = False,
    ) -> list[float]:
        """Plain mini-batch SGD with cross-entropy on logits."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        n = X.shape[0]
        n_classes = self.layer_sizes[-1]
        rng = np.random.default_rng(self.seed if seed is None else seed)
        history: list[float] = []

        for epoch in range(epochs):
            idx = rng.permutation(n)
            total_loss = 0.0
            for start in range(0, n, batch_size):
                batch = idx[start : start + batch_size]
                xb, yb = X[batch], y[batch]
                loss = self._sgd_step(xb, yb, n_classes, lr)
                total_loss += loss * len(batch)
            avg = total_loss / n
            history.append(avg)
            if verbose and (epoch % max(1, epochs // 5) == 0 or epoch == epochs - 1):
                acc = (self.predict(X) == y).mean()
                print(f"epoch {epoch:3d}  loss={avg:.4f}  acc={acc:.3f}")
        return history

    def _sgd_step(
        self, X: np.ndarray, y: np.ndarray, n_classes: int, lr: float
    ) -> float:
        # Forward with activations stored
        activations = [X]
        preacts: list[np.ndarray] = []
        a = X
        for i in range(self.n_layers):
            W = self.weights[f"W{i}"]
            b = self.weights[f"b{i}"]
            z = a @ W.T + b
            preacts.append(z)
            if i < self.n_layers - 1:
                a = relu(z)
            else:
                a = z
            activations.append(a)

        logits = activations[-1]
        probs = softmax(logits)
        Y = one_hot(y, n_classes)
        loss = -np.mean(np.sum(Y * np.log(probs + 1e-12), axis=1))

        # Backward
        delta = (probs - Y) / X.shape[0]  # dL/dz for last layer
        for i in reversed(range(self.n_layers)):
            a_prev = activations[i]
            dW = delta.T @ a_prev
            db = delta.sum(axis=0)
            if i > 0:
                W = self.weights[f"W{i}"]
                da = delta @ W
                delta = da * relu_grad(preacts[i - 1])
            self.weights[f"W{i}"] -= lr * dW
            self.weights[f"b{i}"] -= lr * db
        return float(loss)

    def copy(self) -> "MLP":
        return MLP(
            self.layer_sizes,
            seed=self.seed,
            weights={k: v.copy() for k, v in self.weights.items()},
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        meta = {
            "layer_sizes": self.layer_sizes,
            "seed": self.seed,
            "backend": "numpy",
            "format_version": 1,
        }
        arrays = {k: v for k, v in self.weights.items()}
        arrays["__meta__"] = np.asarray([json.dumps(meta)])
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "MLP":
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            keys = [k for k in data.files if k != "__meta__"]
            if "__meta__" not in data.files:
                raise ValueError(f"missing __meta__ in {path}")
            meta = json.loads(str(data["__meta__"][0]))
            weights = {k: data[k].astype(np.float64) for k in keys}
        return cls(meta["layer_sizes"], seed=meta.get("seed", 0), weights=weights)

    def weight_fingerprint(self) -> dict:
        """Shape + dtype + sha-like stats for integrity checks."""
        out = {}
        for name in self.layer_names():
            arr = self.weights[name]
            out[name] = {
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "sum": float(arr.sum()),
                "abs_sum": float(np.abs(arr).sum()),
            }
        return out


def make_synthetic_blobs(
    n_samples: int = 400,
    n_features: int = 16,
    n_classes: int = 4,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    centers = rng.normal(scale=2.0, size=(n_classes, n_features))
    # Assign by nearest center after adding class-biased noise
    logits = X @ centers.T
    y = logits.argmax(axis=1)
    return X.astype(np.float64), y.astype(np.int64)


def make_digits_dataset(
    seed: int = 0,
    test_frac: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return X_train, y_train, X_test, y_test from sklearn digits (8x8)."""
    if load_digits is None:
        raise ImportError("scikit-learn is required for the digits dataset")
    data = load_digits()
    X = data.data.astype(np.float64) / 16.0  # scale to [0, 1]
    y = data.target.astype(np.int64)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test = int(len(X) * test_frac)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def train_digits_mlp(
    seed: int = 0,
    hidden: Iterable[int] = (32,),
    epochs: int = 80,
    lr: float = 0.2,
) -> tuple[MLP, dict]:
    """Train a small MLP on digits; return model and data dict."""
    Xtr, ytr, Xte, yte = make_digits_dataset(seed=seed)
    sizes = [64, *list(hidden), 10]
    model = MLP(sizes, seed=seed)
    model.train(Xtr, ytr, epochs=epochs, lr=lr, batch_size=64, seed=seed)
    data = {
        "X_train": Xtr,
        "y_train": ytr,
        "X_eval": Xte,
        "y_eval": yte,
    }
    return model, data


def accuracy(model: MLP, X: np.ndarray, y: np.ndarray) -> float:
    if len(y) == 0:
        return float("nan")
    return float((model.predict(X) == y).mean())


def class_accuracy(model: MLP, X: np.ndarray, y: np.ndarray, cls: int) -> float:
    mask = y == cls
    if not mask.any():
        return float("nan")
    return float((model.predict(X[mask]) == y[mask]).mean())
