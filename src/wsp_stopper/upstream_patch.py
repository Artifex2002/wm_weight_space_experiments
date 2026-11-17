"""Version-locked minimal edits. Refuse unknown or manually edited upstream files."""
from pathlib import Path
import hashlib


def patched_text(original, kind):
    def replace_once(text, old, new):
        if text.count(old) != 1:
            raise ValueError("Pinned source anchor mismatch")
        return text.replace(old, new, 1)
    if kind == "planner":
        before, body = original.split("class CEMPlanner(Planner):", 1)
        body, after = body.split("class MPPIPlanner(Planner):", 1)
        body = replace_once(body,
            "        # Iterate CEM\n",
            """        # WSP: optional read-only controller, one independent hook per solve.
        import os
        wsp_hook = None
        if os.environ.get("WSP_STOPPER_WEIGHTS"):
            from wsp_stopper.cem_hook import CEMHook
            wsp_hook = CEMHook(
                os.environ["WSP_STOPPER_WEIGHTS"], std.detach().cpu().numpy(),
                self.iterations, self.distribute_planner,
            )
        self.wsp_last_depth = 0
        self.wsp_last_trace = []
        # Iterate CEM
""")
        body = replace_once(body,
            "\n        self._prev_mean = mean\n",
            """
            self.wsp_last_depth = itr + 1
            if wsp_hook is not None:
                # The upstream returns mean[:num_act_stepped], not the best sample.
                should_stop = wsp_hook.after_refit(
                    cost[:, 0].detach().cpu().numpy(),
                    elite_idxs.detach().cpu().numpy(),
                    std.detach().cpu().numpy(), mean[0].detach().cpu().numpy(),
                )
                self.wsp_last_trace = wsp_hook.trace
                if should_stop:
                    break

        self._prev_mean = mean
""")
        return before + "class CEMPlanner(Planner):" + body + "class MPPIPlanner(Planner):" + after
    if kind == "encoder":
        return replace_once(original,
            '            self.base_model = torch.hub.load("facebookresearch/dinov2", name)',
            '''            # WSP: pin both source and encoder tensors; no moving torch.hub main.
            source = os.environ["WSP_DINOV2_SOURCE"]
            weights = os.environ["WSP_DINOV2_WEIGHTS"]
            self.base_model = torch.hub.load(source, name, source="local", pretrained=False)
            state = torch.load(weights, map_location="cpu", weights_only=True)
            self.base_model.load_state_dict(state, strict=True)''')
    raise ValueError("Unknown patch kind")


def patch(root, lock):
    receipt = {}
    for kind, field, hash_field in [
        ("planner", "planner_source", "planner_sha256"),
        ("encoder", "encoder_source", "encoder_source_sha256"),
    ]:
        path = Path(root)/lock[field]
        backup = path.with_suffix(path.suffix + ".wsp-original")
        original = backup.read_text() if backup.exists() else path.read_text()
        if hashlib.sha256(original.encode()).hexdigest() != lock[hash_field]:
            raise ValueError(f"Unrecognized source version: {path}")
        desired = patched_text(original, kind)
        if backup.exists() and path.read_text() not in (original, desired):
            raise ValueError(f"Refusing to overwrite manual changes: {path}")
        if not backup.exists():
            backup.write_text(original)
        path.write_text(desired)
        receipt[lock[field]] = {"original_sha256": lock[hash_field],
            "patched_sha256": hashlib.sha256(desired.encode()).hexdigest()}
    return receipt
