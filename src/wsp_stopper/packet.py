"""Prepare immutable, numeric-only packets before the generation process starts."""
import json
import shutil
from pathlib import Path
import numpy as np
from .core import digest, write_json


def freeze(checkpoint, state_key, sources, specification, provenance, output):
    # Only this offline preparation module loads PyTorch. Agent tools never do.
    dest = Path(output)
    if dest.exists():
        raise ValueError("Packet destination must not exist")
    spec = json.loads(Path(specification).read_text())
    required = ["checkpoint_revision", "source_commit", "environment", "success_predicate",
                "preprocessing", "action_units", "planner", "episode_limit", "objective"]
    if spec.get("ready") is not True or any(not spec.get(k) for k in required):
        raise ValueError("Complete specification and set ready=true before freezing")
    if type(spec["episode_limit"]) is not int or spec["episode_limit"] <= 0:
        raise ValueError("episode_limit must be a positive integer M")
    import torch
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    for key in state_key.split(".") if state_key else []:
        state = state[key]
    if not isinstance(state, dict):
        raise ValueError("Selected state must be a flat dictionary of tensors")
    dest.mkdir(parents=True)
    (dest/"tensors").mkdir()
    (dest/"sources").mkdir()
    manifest = {"schema": 1, "checkpoint_sha256": digest(checkpoint),
                "provenance": provenance, "state_key": state_key,
                "specification": spec, "tensors": {}, "sources": {}}
    try:
        for name, t in state.items():
            if not isinstance(t, torch.Tensor) or t.is_sparse or t.is_quantized:
                raise ValueError(f"Unsupported state entry: {name}; select a flat state dict")
            a = t.detach().cpu()
            if a.dtype == torch.bfloat16:
                a = a.float()  # exact widening, recorded below
            a = a.numpy()
            if a.dtype.kind not in "fiu" or not np.isfinite(a).all():
                raise ValueError(f"Unsupported/nonfinite tensor: {name}")
            rel = f"tensors/{len(manifest['tensors']):05d}.npy"
            np.save(dest/rel, a, allow_pickle=False)
            manifest["tensors"][name] = {"path": rel, "shape": list(a.shape),
                "original_dtype": str(t.dtype), "stored_dtype": str(a.dtype),
                "sha256": digest(dest/rel), "numel": int(a.size)}
        if not manifest["tensors"]:
            raise ValueError("No tensors found")
        for i, source in enumerate(sources):
            p = Path(source)
            text = p.read_text()
            if len(text) > 500_000:
                raise ValueError("Split large source files before packaging")
            rel = f"sources/{i:03d}_{p.name}"
            (dest/rel).write_text(text)
            manifest["sources"][rel] = {"sha256": digest(dest/rel)}
        write_json(dest/"manifest.json", manifest)
        (dest/"manifest.sha256").write_text(digest(dest/"manifest.json")+"\n")
    except Exception:
        shutil.rmtree(dest)
        raise
    return manifest


def verify(root):
    root = Path(root).resolve()
    if digest(root/"manifest.json") != (root/"manifest.sha256").read_text().strip():
        raise ValueError("Manifest changed")
    m = json.loads((root/"manifest.json").read_text())
    for item in list(m["tensors"].values()) + [dict(v, path=k) for k,v in m["sources"].items()]:
        p = (root/item["path"]).resolve()
        if not p.is_relative_to(root) or digest(p) != item["sha256"]:
            raise ValueError("Packet path/hash mismatch")
    return m
