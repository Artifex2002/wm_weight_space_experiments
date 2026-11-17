"""Responses API function loop. No model or environment code is imported here."""
import json
import os
import time
import urllib.request
from pathlib import Path
from .core import digest, parse_json, validate_weights, write_json
from .inspection import Inspector, TOOLS


class API:
    def __init__(self):
        self.key = os.environ["OPENAI_API_KEY"]

    def request(self, method, resource, body=None):
        req = urllib.request.Request("https://api.openai.com/v1/"+resource,
            data=None if body is None else json.dumps(body, allow_nan=False).encode(),
            headers={"Authorization": "Bearer "+self.key, "Content-Type": "application/json"}, method=method)
        # Do not log headers or automatically retry billable POST requests.
        with urllib.request.urlopen(req, timeout=300) as response:
            return json.load(response)


def run(packet, prompt, model, out, max_calls=40, max_chars=200_000,
        max_turns=45, output_tokens=8000, reasoning=None, transport=None):
    if not model:
        raise ValueError("Set an exact API model ID; no model is assumed")
    if min(max_calls, max_chars, max_turns, output_tokens) < 1:
        raise ValueError("Budgets must be positive")
    inspector = Inspector(packet)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    instructions = Path(prompt).read_text()
    manifest = inspector.manifest
    write_json(out/"run.json", {"model": model, "manifest_sha256": digest(Path(packet)/"manifest.json"),
        "prompt_sha256": digest(prompt), "max_calls": max_calls, "max_chars": max_chars,
        "max_turns": max_turns, "max_output_tokens_per_turn": output_tokens,
        "reasoning_effort": reasoning, "started_unix": time.time(),
        "store": False, "tools": TOOLS})
    (out/"prompt.txt").write_text(instructions)
    history = [{"role": "user", "content": json.dumps({"specification": manifest["specification"],
        "checkpoint_sha256": manifest["checkpoint_sha256"], "provenance": manifest["provenance"],
        "tensor_count": len(manifest["tensors"]), "source_files": list(manifest["sources"]),
        "inspection_call_budget": max_calls, "inspection_output_character_budget": max_chars})}]
    api = transport or API()
    used_calls, used_chars = 0, 0
    audit = open(out/"events.jsonl", "x")

    def log(kind, value):
        audit.write(json.dumps({"kind": kind, "value": value}, allow_nan=False)+"\n")
        audit.flush()

    try:
        for turn in range(max_turns):
            available = TOOLS if used_calls < max_calls and used_chars < max_chars else [TOOLS[-1]]
            body = {"model": model, "instructions": instructions, "input": history,
                "tools": available, "parallel_tool_calls": False, "store": False,
                "include": ["reasoning.encrypted_content"], "max_output_tokens": output_tokens}
            if reasoning:
                body["reasoning"] = {"effort": reasoning}
            log("request", {"turn": turn, "tool_names": [t["name"] for t in available]})
            result = api.request("POST", "responses", body)
            log("response", result)
            if result.get("status") != "completed":
                raise RuntimeError("Incomplete API response; no controller accepted")
            history.extend(result["output"])
            calls = [item for item in result["output"] if item.get("type") == "function_call"]
            if not calls:
                raise RuntimeError("Model ended without submit_weights; no retry or performance-based repair")
            if len(calls) != 1:
                raise RuntimeError("Protocol requires one tool call per response")
            call = calls[0]
            name = call["name"]
            try:
                args = parse_json(call["arguments"])
            except Exception:
                if name == "submit_weights":
                    write_json(out/"submission.json", {"valid": False, "reason": "Malformed JSON"})
                    raise RuntimeError("Invalid final submission")
                args = None
            if name == "submit_weights":
                write_json(out/"submitted_raw.json", args)
                try:
                    validate_weights(args)
                except Exception as error:
                    write_json(out/"submission.json", {"valid": False, "reason": str(error)})
                    raise RuntimeError("Invalid final submission; generation ended") from error
                # Preserve submitted JSON values; conversion to float32 happens in inference.
                write_json(out/"weights.json", args)
                write_json(out/"submission.json", {"valid": True, "weights_sha256": digest(out/"weights.json"),
                    "inspection_calls": used_calls, "inspection_output_chars": used_chars})
                log("accepted", {"weights_sha256": digest(out/"weights.json")})
                return out/"weights.json"
            used_calls += 1
            if used_calls > max_calls or used_chars >= max_chars:
                answer = {"error": "Inspection budget exhausted; submit weights"}
            else:
                try:
                    answer = inspector.call(name, args)
                except Exception as error:
                    answer = {"error": str(error)}
            encoded = json.dumps(answer, allow_nan=False)
            if used_chars+len(encoded) > max_chars:
                encoded = json.dumps({"error": "Result exceeds remaining output budget; submit or request less"})
            used_chars += len(encoded)
            log("tool", {"name": name, "arguments": args, "output": encoded})
            history.append({"type": "function_call_output", "call_id": call["call_id"], "output": encoded})
        raise RuntimeError("Turn budget exhausted; no controller accepted")
    finally:
        audit.close()
