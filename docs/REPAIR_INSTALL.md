# Missing wsp_stopper.core or wsp_stopper.cli

The Wall upgrade ZIP is a delta, not a standalone application. It contains
wall_setup.py but does not contain the original core.py, cli.py, packet.py,
inspection.py, agent.py, or package __init__.py. Pip can install editable package
metadata successfully even when these runtime modules are absent. Installing
more dependencies will not restore missing project files.

Use the COMPLETE bundle to repair your existing folder without replacing local
work. Extract it outside the project:

```bash
cd /Users/ashutoshpanda/Documents/Projects/wsp-stopper
repair_dir=$(mktemp -d)
unzip -q ~/Downloads/wsp_stopper_complete.zip -d "$repair_dir"
python "$repair_dir/wsp-stopper/scripts/restore_missing.py" --target "$PWD"
python -m pip install -e . --no-deps
python -c "import wsp_stopper.core, wsp_stopper.cli, wsp_stopper.wall_setup; print('Imports OK')"
python -m unittest discover -s tests -v
```

The repair restores all missing distribution files, leaves every existing file
unchanged, reports differing files, and prints an exact git-add command for the
files restored. If imports still fail after the repair, inspect the reported
conflicts and collect:

```bash
python -c "import sys,wsp_stopper; print(sys.executable); print(wsp_stopper.__file__); print(list(wsp_stopper.__path__))"
python -m pip show wsp-stopper
ls src/wsp_stopper
```

After imports/tests pass, commit using the command printed by restore_missing.py.
Do not rerun init_history.sh or commit_wall_upgrade.sh merely to repair imports.
They are history setup helpers, not dependency installers.

Then rerun preparation commands one at a time, avoiding pasted rich-text spaces
or blank lines around continuation backslashes:

```bash
wsp-wall prepare --work assets/wall-v1
wsp-wall freeze --work assets/wall-v1 --out packets/wall-v1
wsp verify packets/wall-v1
```

If a packet already exists, verify it rather than deleting it. Use a new version
name if you need to regenerate it after source/config changes.

Current local Python 3.13 is not the cause of these missing-module exceptions.
Preparation is CPU-only and supports Python >=3.10 when dependency wheels exist.
The upstream baseline package separately declares Python >=3.10,<3.11. A working
local preparation environment does not establish Colab simulator compatibility.

Colab baseline installation, dataset/environment setup, real model state loading,
GPU forward execution and baseline quality checks remain unvalidated. Run these
before freezing the final generation packet. If fixes change config/source,
freeze a fresh packet, then run API generation with the exact model/key. Baseline
outcomes remain outside the generator's source/weight packet.
