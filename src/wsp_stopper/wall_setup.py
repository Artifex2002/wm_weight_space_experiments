"""Trusted offline preparation: downloads, tensor export and frozen source packet.
No environment or world-model forward computation is performed.
"""
import argparse
import ast
import json
import os
import shutil
import subprocess
import urllib.request
from pathlib import Path
from .core import digest, write_json
from .upstream_patch import patch


def checkout(url, commit, path):
    path = Path(path)
    if not (path/".git").exists():
        if path.exists():
            raise ValueError(f"Non-git path already exists: {path}")
        subprocess.run(["git", "clone", "--no-checkout", url, str(path)], check=True)
        subprocess.run(["git", "-C", str(path), "checkout", "--detach", commit], check=True)
    actual = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    if actual != commit:
        raise ValueError(f"Expected {commit}, found {actual}; use a fresh work directory")


def download(url, sha, path):
    path = Path(path)
    if path.exists():
        if digest(path) != sha:
            raise ValueError(f"Existing asset checksum mismatch: {path}")
        return
    temp = path.with_suffix(path.suffix+".partial")
    with urllib.request.urlopen(url, timeout=120) as response, open(temp,"wb") as output:
        shutil.copyfileobj(response, output)
    if digest(temp) != sha:
        temp.unlink()
        raise ValueError(f"Downloaded checksum mismatch: {path}")
    temp.replace(path)


def prepare(lock_path, work):
    import torch
    lock = json.loads(Path(lock_path).read_text())
    work = Path(work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    checkout(lock["upstream_url"], lock["upstream_commit"], work/"jepa-wms")
    checkout(lock["dinov2_url"], lock["dinov2_commit"], work/"dinov2")
    download(lock["checkpoint_url"],lock["checkpoint_sha256"],work/"wall.pth.tar")
    download(lock["encoder_url"],lock["encoder_sha256"],work/"encoder.pth")
    ckpt = torch.load(work/"wall.pth.tar",map_location="cpu",weights_only=True)
    encoder = torch.load(work/"encoder.pth",map_location="cpu",weights_only=True)
    selected = {"visual_encoder": encoder}
    for component in ("predictor", "action_encoder", "proprio_encoder"):
        selected[component] = ckpt[component]
    flat, counts = {}, {}
    for component, state in selected.items():
        if not isinstance(state, dict) or not state:
            raise ValueError(f"Missing component: {component}")
        count = 0
        for name, tensor in state.items():
            if not isinstance(tensor,torch.Tensor):
                raise ValueError(f"Non-tensor in {component}.{name}")
            # Keep original tensor names verbatim, including module. prefixes.
            flat[f"{component}.{name}"] = tensor
            count += tensor.numel()
        counts[component] = {"tensor_entries":len(state),"scalar_entries_including_buffers":count}
    target = work/"inspection-state.pt"
    torch.save(flat,target)
    report = {"lock":lock, "checkpoint_epoch":ckpt.get("epoch"), "components":counts,
              "inspection_state_sha256":digest(target),
              "scope":"visual encoder + predictor + action encoder + proprio encoder; optimizer/scaler excluded",
              "validation":"raw tensor export only; model.load_state_dict and Wall execution pending"}
    write_json(work/"preparation.json",report)
    print(json.dumps(report,indent=2))


def make_packet(lock_path, work, out):
    import yaml
    from .packet import freeze, verify
    lock = json.loads(Path(lock_path).read_text())
    work, out = Path(work).resolve(), Path(out).resolve()
    if out.exists():
        raise ValueError("Choose a new packet directory; frozen packets are not overwritten")
    report = json.loads((work/"preparation.json").read_text())
    if report["lock"] != lock:
        raise ValueError("Preparation lock does not match")
    for file, key in [("wall.pth.tar","checkpoint_sha256"),("encoder.pth","encoder_sha256")]:
        if digest(work/file) != lock[key]:
            raise ValueError("Asset changed after preparation")
    if digest(work/"inspection-state.pt") != report["inspection_state_sha256"]:
        raise ValueError("Export changed after preparation")
    for folder, key in [("jepa-wms","upstream_commit"),("dinov2","dinov2_commit")]:
        actual = subprocess.check_output(["git","-C",str(work/folder),"rev-parse","HEAD"],text=True).strip()
        if actual != lock[key]:
            raise ValueError("Source revision changed")
    # Version pinning must also reject local edits to implementation files.
    for folder, allowed in [("jepa-wms", {lock["planner_source"],lock["encoder_source"]}), ("dinov2", set())]:
        changed = subprocess.check_output(["git","-C",str(work/folder),"diff","HEAD","--name-only"],text=True).splitlines()
        if set(changed)-allowed:
            raise ValueError(f"Unreviewed source changes in {folder}: {changed}")
    receipt = patch(work/"jepa-wms",lock)
    base = work/"jepa-wms"
    config = yaml.safe_load((base/lock["upstream_config"]).read_text())
    # Explicit experiment overrides: no decoder downloads, one process, finite cap.
    config["tasks_per_node"] = 1
    config["meta"]["eval_episodes"] = 50
    config["logging"]["optional_plots"] = False
    config["planner"].update(iterations=30, decode_each_iteration=False, distribute_planner=False)
    config["model_kwargs"]["pretrain_kwargs"]["heads_cfg"] = {"architectures":{},"pretrain_dec_path":{}}
    config["model_kwargs"]["checkpoint"] = str(work/"wall.pth.tar")
    config["folder"] = str(work)
    # Evaluator still requires the dataset path/runtime setup in Colab.
    (work/"wall.eval.yaml").write_text(yaml.safe_dump(config,sort_keys=False))
    stats_ast = ast.parse((base/"app/plan_common/datasets/__init__.py").read_text())
    stats = next(ast.literal_eval(n.value)["wall"] for n in stats_ast.body
                 if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="DATA_STATS" for t in n.targets))
    spec = {"ready":True,"checkpoint_revision":lock["checkpoint_url"],
        "source_commit":lock["upstream_commit"],
        "environment":{"task":"wall-base","source":"evals/simu_env_planning/envs/wall_gym_wrap.py",
                       "goal_source":"random_state","max_simulator_steps":30},
        "success_predicate":"L2 norm of goal_state[:2]-cur_state[:2] < 4.5; exact wrapper source supplied",
        "preprocessing":{"image_size":224,"image_normalize_mean":[.5,.5,.5],
                         "image_normalize_std":[.5,.5,.5],"registry_wall_stats":stats,
                         "model_context_window":2,"frameskip":5,"action_skip":1},
        "action_units":{"raw_action_dim":2,"internal_action_dim":10,
            "packing":"each planner action concatenates five consecutive 2D actions; unpacks before denormalization",
            "chunk":6,"max_decisions":1},
        "planner":config["planner"],"episode_limit":1,
        "objective":"U = S - 0.05 * total_completed_CEM_iterations / (30*M); M=1",
        "return_plan":"updated proposal mean; first num_act_stepped entries",
        "initialization":"mean zeros, sigma var_scale=1; no warm-start use despite stored _prev_mean",
        "implementation_notes":"upstream adds a proposal-mean rollout each iteration for predictions/logging; count it in compute metrics",
        "validation_status":"prepared_from_source_and_tensor_inventory; execution_pending",
        "scope":report["scope"],"protocol":"weight inspection only; no forward passes or evaluation feedback"}
    write_json(work/"wall.spec.json",spec)
    write_json(work/"patch-receipt.json",receipt)
    # Explicitly package implementation files, not README/research tables/logs/datasets.
    sources = []
    paths = ["LICENSE","hubconf.py","app/vjepa_wm/video_wm.py","app/vjepa_wm/utils.py","src/utils/logging.py","src/utils/yaml_utils.py",
        "app/vjepa_wm/modelcustom/simu_env_planning/vit_enc_preds.py",
        "evals/simu_env_planning/planning/gc_agent.py",
        lock["planner_source"],"evals/simu_env_planning/planning/planning/objectives.py",
        "evals/simu_env_planning/planning/plan_evaluator.py",
        "evals/simu_env_planning/envs/wall_gym_wrap.py","evals/simu_env_planning/envs/init.py"]
    paths += [str(p.relative_to(base)) for folder in ["app/plan_common/models","app/plan_common/datasets","src/models","src/utils",
        "evals/simu_env_planning/envs/wall_env","evals/simu_env_planning/envs/wrappers"]
        for p in (base/folder).rglob("*.py")]
    # Include names as headers to disambiguate duplicate basenames in read_source output.
    staging = work/"source-text"
    staging.mkdir(exist_ok=True)
    for i,rel in enumerate(sorted(set(paths))):
        p=base/rel
        dst=staging/f"{i:03d}_{p.name}.txt"
        dst.write_text(f"# Upstream path: {rel}\n"+p.read_text())
        sources.append(dst)
    for p in sorted((work/"dinov2/dinov2").rglob("*.py")):
        if "/layers/" in str(p) or p.name=="vision_transformer.py" or "/hub/" in str(p):
            dst=staging/f"dino_{len(sources):03d}_{p.name}.txt"
            dst.write_text(f"# DINOv2 path: {p.relative_to(work/'dinov2')}\n"+p.read_text())
            sources.append(dst)
    root_hub=staging/"dinov2_hubconf.py.txt"
    root_hub.write_text((work/"dinov2/hubconf.py").read_text())
    sources.append(root_hub)
    license_path=staging/"dinov2_LICENSE.txt"
    license_path.write_text((work/"dinov2/LICENSE").read_text())
    sources.append(license_path)
    sources += [work/"wall.eval.yaml", work/"wall.spec.json"]
    for name in ["core.py","cem_hook.py"]:
        sources.append(Path(__file__).parent/name)
    freeze(work/"inspection-state.pt","",sources,work/"wall.spec.json",
           {"upstream_commit":lock["upstream_commit"],"dinov2_commit":lock["dinov2_commit"],
            "asset_hashes":{k:v for k,v in lock.items() if k.endswith("sha256")},
            "checkpoint_epoch":report["checkpoint_epoch"],"patches":receipt,"components":report["components"]},out)
    verify(out)
    print(f"Verified packet: {out}; runtime smoke test still pending")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("operation", choices=["prepare","freeze"])
    p.add_argument("--lock",default="config/wall.release.json")
    p.add_argument("--work",default="assets/wall-v1")
    p.add_argument("--out",default="packets/wall-v1")
    a=p.parse_args()
    if a.operation=="prepare": prepare(a.lock,a.work)
    else: make_packet(a.lock,a.work,a.out)

if __name__=="__main__": main()
