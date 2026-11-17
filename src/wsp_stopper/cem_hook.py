"""Runtime only; this module is never exposed as an execution tool to Astra."""
import json
from pathlib import Path
from .core import Features, Stopper, digest


class CEMHook:
    """One single-environment CEM solve. Upstream is serial across episodes."""
    def __init__(self, weights_path, initial_sigma, iterations, distributed=False):
        if iterations != 30 or distributed:
            raise ValueError("WSP hook requires cap 30 and non-distributed CEM")
        self.stopper = Stopper(json.loads(Path(weights_path).read_text()))
        self.weights_sha256 = digest(weights_path)
        self.features = Features(initial_sigma)
        self.trace = []

    def after_refit(self, costs, elite_indices, sigma, returned_first_action):
        x = self.features.update(costs, elite_indices, sigma, returned_first_action)
        stop = self.stopper.stop(x, self.features.k)
        self.trace.append({"k": self.features.k, "features": x.tolist(), "stop": stop})
        return stop
