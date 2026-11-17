"""Trusted preparation only: list structure to locate the flat state dictionary."""
import argparse
import torch
p = argparse.ArgumentParser()
p.add_argument("checkpoint")
a = p.parse_args()
x = torch.load(a.checkpoint, map_location="cpu", weights_only=True)

def show(obj, prefix="", depth=0):
    if isinstance(obj, dict) and depth < 3:
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, torch.Tensor):
                print(path, list(value.shape), str(value.dtype))
            elif isinstance(value, dict):
                print(path, "dictionary", len(value), "entries")
                show(value, path, depth+1)
            else:
                print(path, type(value).__name__)
show(x)
