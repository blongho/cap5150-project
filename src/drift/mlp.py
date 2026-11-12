"""Two-hidden-layer MLP with a scikit-learn style fit / predict / predict_proba interface.

Training runs on the CPU so that a fixed `random_state` gives identical results across runs.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class MLP(nn.Module):
    """Binary classifier: input -> [Linear, ReLU, Dropout] x 2 -> Linear(2).

    `fit` uses class-weighted cross-entropy and Adam, holds out `val_fraction` of the training
    data for early stopping, and restores the weights with the lowest validation loss. The
    network is (re)built on every `fit`, so the input size comes from the data.
    """

    classes_ = np.array([0, 1])

    def __init__(
        self,
        hidden_size: int = 256,
        dropout: float = 0.3,
        epochs: int = 50,
        lr: float = 1e-3,
        patience: int = 5,
        batch_size: int = 256,
        val_fraction: float = 0.1,
        random_state: int | None = None,
    ):
        super().__init__()
        if not 0 < val_fraction < 1:
            raise ValueError("val_fraction must lie in (0, 1)")
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.epochs = epochs
        self.lr = lr
        self.patience = patience
        self.batch_size = batch_size
        self.val_fraction = val_fraction
        self.random_state = random_state
        self.net: nn.Sequential | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.net is None:
            raise RuntimeError("MLP is not fitted; call fit() first")
        return self.net(x)

    def _build(self, n_features: int) -> None:
        h, p = self.hidden_size, self.dropout
        self.net = nn.Sequential(
            nn.Linear(n_features, h), nn.ReLU(), nn.Dropout(p),
            nn.Linear(h, h), nn.ReLU(), nn.Dropout(p),
            nn.Linear(h, 2),
        )

    def _seed(self) -> torch.Generator:
        """Seed torch and numpy from random_state; return the generator used for shuffling."""
        generator = torch.Generator()
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)
            generator.manual_seed(self.random_state)
        return generator

    def _epoch(self, X: torch.Tensor, y: torch.Tensor, criterion, optimizer, generator) -> None:
        """One pass over (X, y) in shuffled mini-batches."""
        self.train()
        for batch in torch.randperm(len(X), generator=generator).split(self.batch_size):
            optimizer.zero_grad()
            criterion(self(X[batch]), y[batch]).backward()
            optimizer.step()

    def fit(self, X, y) -> MLP:
        """Train from scratch on X (n_samples, n_features) and binary labels y in {0, 1}."""
        X, y = _check_xy(X, y)
        n_val = max(1, round(self.val_fraction * len(X)))
        if len(X) - n_val < 1:
            raise ValueError("Too few samples to hold out a validation split")

        generator = self._seed()
        rng = np.random.default_rng(self.random_state)
        self._build(X.shape[1])

        order = rng.permutation(len(X))
        val_idx, train_idx = order[:n_val], order[n_val:]
        X_tr, y_tr = torch.from_numpy(X[train_idx]), torch.from_numpy(y[train_idx])
        X_val, y_val = torch.from_numpy(X[val_idx]), torch.from_numpy(y[val_idx])

        criterion = _weighted_loss(y[train_idx])
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        best_loss, best_state, stale = float("inf"), None, 0
        self.n_epochs_ = 0
        for _ in range(self.epochs):
            self._epoch(X_tr, y_tr, criterion, optimizer, generator)
            self.n_epochs_ += 1

            self.eval()
            with torch.no_grad():
                val_loss = criterion(self(X_val), y_val).item()
            if val_loss < best_loss:
                best_loss, stale = val_loss, 0
                best_state = {k: v.detach().clone() for k, v in self.state_dict().items()}
            else:
                stale += 1
                if stale >= self.patience:
                    break

        if best_state is not None:
            self.load_state_dict(best_state)
        self.best_val_loss_ = best_loss
        self.eval()
        return self

    def finetune(self, X, y, epochs: int, lr: float) -> MLP:
        """Continue training the fitted network on (X, y) for a fixed number of epochs.

        Uses a fresh Adam optimiser at `lr` and class weights from y; no validation split,
        because labelled budgets are small. Modifies and returns self.
        """
        if self.net is None:
            raise RuntimeError("MLP is not fitted; call fit() before finetune()")
        X, y = _check_xy(X, y)
        generator = self._seed()
        X_t, y_t = torch.from_numpy(X), torch.from_numpy(y)
        criterion = _weighted_loss(y)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        for _ in range(epochs):
            self._epoch(X_t, y_t, criterion, optimizer, generator)
        self.eval()
        return self

    def predict_proba(self, X) -> np.ndarray:
        """Class probabilities, shape (n_samples, 2); column 1 is P(malware)."""
        self.eval()
        with torch.no_grad():
            logits = self(torch.from_numpy(_as_float_array(X)))
            return torch.softmax(logits, dim=1).numpy().astype(np.float64)

    def predict(self, X) -> np.ndarray:
        """Predicted labels in {0, 1}, shape (n_samples,)."""
        return self.predict_proba(X).argmax(axis=1).astype(np.int64)


def _as_float_array(X) -> np.ndarray:
    """Accept a DataFrame or array-like and return a contiguous float32 array."""
    return np.ascontiguousarray(np.asarray(X, dtype=np.float32))


def _check_xy(X, y) -> tuple[np.ndarray, np.ndarray]:
    X = _as_float_array(X)
    y = np.asarray(y).astype(np.int64)
    if len(X) != len(y):
        raise ValueError("X and y have different numbers of rows")
    if not set(np.unique(y)) <= {0, 1}:
        raise ValueError("MLP expects binary labels in {0, 1}")
    return X, y


def _weighted_loss(y: np.ndarray) -> nn.CrossEntropyLoss:
    """Cross-entropy with inverse-frequency class weights; an absent class gets weight 0."""
    counts = np.bincount(y, minlength=2).astype(np.float64)
    weights = np.divide(len(y), 2 * counts, out=np.zeros(2), where=counts > 0)
    return nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))
