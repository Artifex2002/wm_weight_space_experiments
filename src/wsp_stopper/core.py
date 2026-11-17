import hashlib
import json
import math
from pathlib import Path

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def parse_json(text):
    def reject(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")
    return json.loads(text, parse_constant=reject)


def validate_weights(w):
    shapes = {"W1": (16, 8), "b1": (16,), "W2": (2, 16), "b2": (2,)}
    if not isinstance(w, dict) or set(w) != set(shapes):
        raise ValueError("Expected exactly W1, b1, W2, b2")
    result = {}
    for key, shape in shapes.items():
        a = np.asarray(w[key])
        if a.shape != shape or a.dtype.kind not in "fiu":
            raise ValueError(f"{key} must be a numeric array of shape {shape}")
        with np.errstate(over="ignore", invalid="ignore"):
            a = a.astype(np.float32)
        if not np.isfinite(a).all() or np.abs(a).max(initial=0) > 1e6:
            raise ValueError("Weights must be finite float32 with magnitude <= 1e6")
        result[key] = a
    return result


class Stopper:
    def __init__(self, weights):
        self.w = validate_weights(weights)

    def stop(self, x, k):
        if not 1 <= k <= 30:
            raise ValueError("k must be in 1..30")
        if k < 3:
            return False
        if k == 30:
            return True
        x = np.asarray(x, dtype=np.float32)
        if x.shape != (8,) or not np.isfinite(x).all():
            raise ValueError("Expected 8 finite features")
        w = self.w
        logits = w["W2"] @ np.maximum(0, w["W1"] @ x + w["b1"]) + w["b2"]
        if not np.isfinite(logits).all():
            raise ValueError("Nonfinite logits")
        return bool(logits[1] > logits[0])


class Features:
    """One instance per CEM solve. sigma inputs are standard deviations."""
    def __init__(self, initial_sigma):
        self.initial = np.asarray(initial_sigma, dtype=float)
        if self.initial.ndim != 2 or not np.isfinite(self.initial).all() or (self.initial <= 0).any():
            raise ValueError("initial_sigma must be positive [horizon, action_dim]")
        self.k = 0
        self.prev_e = None
        self.prev_r = 0.0
        self.prev_a = None

    def update(self, costs, elite_indices, sigma, returned_first_action):
        costs = np.asarray(costs, dtype=float)
        indices = np.asarray(elite_indices)
        sigma = np.asarray(sigma, dtype=float)
        a = np.asarray(returned_first_action, dtype=float)
        if costs.ndim != 1 or not len(costs) or not np.isfinite(costs).all() or (costs < 0).any():
            raise ValueError("Expected finite nonnegative candidate costs")
        if indices.ndim != 1 or not indices.size or indices.dtype.kind not in "iu" or (indices < 0).any() or (indices >= len(costs)).any() or len(set(indices.tolist())) != len(indices):
            raise ValueError("Invalid elite indices")
        if sigma.shape != self.initial.shape or not np.isfinite(sigma).all() or (sigma < 0).any():
            raise ValueError("Invalid proposal standard deviations")
        if a.shape != (self.initial.shape[1],) or not np.isfinite(a).all():
            raise ValueError("Invalid returned first action")
        if self.k >= 30:
            raise ValueError("Create a new Features object for each solve")
        elite = costs[indices]
        e, b, v = float(elite.mean()), float(costs.min()), float(elite.std(ddof=0))
        self.k += 1
        if self.k == 1:
            self.first_e = e
        r = 0.0 if self.prev_e is None else float(np.clip((self.prev_e-e)/max(abs(self.prev_e), 1e-8), -1, 1))
        delta = 0.0 if self.prev_a is None else float(np.sqrt(np.mean(((a-self.prev_a)/np.maximum(self.initial[0], 1e-8))**2)))
        x = np.array([
            self.k/30, r, (r+self.prev_r)/2,
            np.clip(e/max(abs(self.first_e), 1e-8), 0, 2),
            np.clip((e-b)/max(abs(e), 1e-8), 0, 1),
            np.clip(v/max(abs(e), 1e-8), 0, 2),
            np.clip(np.mean(sigma/np.maximum(self.initial, 1e-8)), 0, 2),
            np.clip(delta, 0, 2),
        ], dtype=np.float32)
        self.prev_e, self.prev_r, self.prev_a = e, r, a.copy()
        return x
