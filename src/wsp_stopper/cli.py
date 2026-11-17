import argparse
import json
import os
from pathlib import Path
from .core import digest, write_json


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("models")
    download = sub.add_parser("download")
    download.add_argument("--repo", default="facebook/jepa-wms")
    download.add_argument("--filename", default="dino_wm_wall.pth.tar")
    download.add_argument("--revision", required=True, help="Full HF repository commit SHA")
    download.add_argument("--out", default="assets")
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--checkpoint", required=True)
    freeze.add_argument("--state-key", default="", help="Dot path to flat tensor state dict; empty means root")
    freeze.add_argument("--source", action="append", required=True)
    freeze.add_argument("--spec", required=True)
    freeze.add_argument("--provenance", required=True)
    freeze.add_argument("--out", required=True)
    run = sub.add_parser("generate")
    run.add_argument("--packet", required=True)
    run.add_argument("--prompt", default="prompts/generate.md")
    run.add_argument("--model", default=os.environ.get("OPENAI_MODEL"))
    run.add_argument("--out", required=True)
    run.add_argument("--max-calls", type=int, default=40)
    run.add_argument("--max-chars", type=int, default=200000)
    run.add_argument("--max-turns", type=int, default=45)
    run.add_argument("--output-tokens", type=int, default=8000)
    run.add_argument("--reasoning", default=None)
    verify = sub.add_parser("verify")
    verify.add_argument("packet")
    a = p.parse_args()
    if a.cmd == "models":
        from .agent import API
        result = API().request("GET", "models")
        print("\n".join(sorted(x["id"] for x in result["data"])))
    elif a.cmd == "download":
        import re
        from huggingface_hub import hf_hub_download
        if not re.fullmatch(r"[0-9a-f]{40}", a.revision):
            raise ValueError("Use a full 40-character HF commit SHA")
        path = hf_hub_download(repo_id=a.repo, filename=a.filename,
                               revision=a.revision, local_dir=a.out)
        write_json(Path(a.out)/"download.json", {"repo": a.repo, "revision": a.revision,
            "filename": a.filename, "sha256": digest(path)})
        print(path)
    elif a.cmd == "freeze":
        from .packet import freeze
        m = freeze(a.checkpoint, a.state_key, a.source, a.spec, a.provenance, a.out)
        print(json.dumps({"tensors": len(m["tensors"]), "packet": a.out}))
    elif a.cmd == "verify":
        from .packet import verify
        m = verify(a.packet)
        print(json.dumps({"verified": True, "tensor_count": len(m["tensors"])}))
    else:
        from .agent import run
        print(run(a.packet, a.prompt, a.model, a.out, a.max_calls, a.max_chars,
                  a.max_turns, a.output_tokens, a.reasoning))
