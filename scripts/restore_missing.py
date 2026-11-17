"""Restore missing distribution files without overwriting an existing project.
Run this script from the extracted COMPLETE bundle, targeting your current repo.
"""
import argparse
import hashlib
import json
import shlex
import shutil
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def restore(source, target):
    source, target = Path(source).resolve(), Path(target).resolve()
    if source == target:
        raise ValueError("Extract the bundle to a different directory before restoring")
    if not target.is_dir():
        raise ValueError("Target must be your existing project directory")
    manifest = json.loads((source/"DIST_FILES.json").read_text())
    restored, conflicts = [], []
    # Verify every source before touching the target.
    for rel, expected in manifest.items():
        p = (source/rel).resolve()
        if not p.is_relative_to(source) or sha(p) != expected:
            raise ValueError(f"Invalid distribution file: {rel}")
    for rel in manifest:
        src, dst = source/rel, target/rel
        if not dst.resolve().is_relative_to(target):
            raise ValueError(f"Target path escapes project: {rel}")
        if dst.exists():
            if not dst.is_file() or sha(dst) != manifest[rel]:
                conflicts.append(rel)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        restored.append(rel)
    return restored, conflicts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", required=True)
    p.add_argument("--source", default=str(Path(__file__).resolve().parents[1]))
    a = p.parse_args()
    restored, conflicts = restore(a.source, a.target)
    print(f"Restored {len(restored)} missing files; existing files were preserved.")
    for rel in restored:
        print("  restored:", rel)
    if conflicts:
        print("Existing files differ from the complete bundle (preserved; review if tests fail):")
        for rel in conflicts:
            print("  differs:", rel)
    if restored:
        print("\nAfter tests pass, stage only the restored files with:")
        print("git add -- " + " ".join(shlex.quote(x) for x in restored))
        print('git diff --cached --stat')
        print('git commit -m "fix: restore missing starter modules required by Wall tooling"')
        print('git push')

if __name__ == "__main__":
    main()
