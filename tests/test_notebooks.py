"""Protocol integration checks for the self-contained Colab notebooks.

These checks need only the Python standard library. They do not run model
inference, query an API, import the embedded harness, or download assets.
"""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
NAMES = [
    '01_wall_baseline.ipynb', '02_wall_wsp_comparison.ipynb',
    '03_wall_frozen_stopper_diagnostics.ipynb', '04_wall_prompt_comparison.ipynb',
]
RUNTIME_SHA = 'c850c41dbe72088e3946a0d6ad965423c69675b25d41f0ef38a5aac8da45b299'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_cells(name):
    notebook = json.loads((ROOT/'notebooks'/name).read_text())
    return [''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code']


def embedded(name, variable):
    for source in code_cells(name):
        for node in ast.parse(source).body:
            if isinstance(node, ast.Assign) and any(
                isinstance(t,ast.Name) and t.id==variable for t in node.targets
            ):
                return ast.literal_eval(node.value)
    raise AssertionError(f'Missing {variable} in {name}')


def fingerprint_functions():
    sources = code_cells(NAMES[1])
    original = next(n for s in sources for n in ast.parse(s).body
                    if isinstance(n,ast.FunctionDef) and n.name=='project_fingerprint')
    helper = ast.parse(embedded(NAMES[3],'STUDY_HELPER_SOURCE'))
    study = next(n for n in helper.body if isinstance(n,ast.ClassDef) and n.name=='Study')
    fingerprint = next(n for n in study.body if isinstance(n,ast.FunctionDef) and n.name=='fingerprint')
    freeze = next(n for n in study.body if isinstance(n,ast.FunctionDef) and n.name=='freeze')
    return original, fingerprint, freeze


class NotebookTests(unittest.TestCase):
    def test_notebooks_parse_and_new_deliveries_have_no_outputs(self):
        for name in NAMES:
            notebook=json.loads((ROOT/'notebooks'/name).read_text())
            self.assertEqual(notebook['nbformat'],4)
            for cell in notebook['cells']:
                if cell['cell_type']=='code':
                    ast.parse(''.join(cell['source']))
                    if name in NAMES[2:]:
                        self.assertIsNone(cell['execution_count'])
                        self.assertEqual(cell['outputs'],[])

    def test_all_notebooks_share_the_original_pinned_evaluator(self):
        for name in NAMES:
            runtime=embedded(name,'RUNTIME_SOURCE')
            self.assertEqual(hashlib.sha256(runtime.encode()).hexdigest(),RUNTIME_SHA,name)

    def test_study_fingerprint_matches_independent_notebook02_schema(self):
        original, fingerprint, _ = fingerprint_functions()
        with tempfile.TemporaryDirectory() as temp:
            project=Path(temp)
            (project/'src/wsp_stopper').mkdir(parents=True)
            (project/'config').mkdir();(project/'prompts').mkdir()
            for name in ['__init__.py','core.py','agent.py','additional_module.py']:
                (project/'src/wsp_stopper'/name).write_text('# fixture\n')
            (project/'config/wall.release.json').write_text('{}\n')
            (project/'prompts/generate.md').write_text('fixture prompt\n')
            env={'PROJECT':project,'digest':digest}
            exec(compile(ast.Module(body=[original,fingerprint],type_ignores=[]),'<fingerprint test>','exec'),env)
            expected=env['project_fingerprint']()
            subject=types.SimpleNamespace(project=project)
            self.assertEqual(env['fingerprint'](subject),expected)
            self.assertIn('src/wsp_stopper/__init__.py',expected)
            self.assertIn('src/wsp_stopper/additional_module.py',expected)
            self.assertIn('config/wall.release.json',expected)
            self.assertIn('prompts/generate.md',expected)
            (project/'prompts/generate.md').write_text('changed prompt\n')
            self.assertNotEqual(env['fingerprint'](subject),expected)

    def test_real_preflight_accepts_reference_schema_and_rejects_source_edits(self):
        original,fingerprint,freeze=fingerprint_functions()
        class BeforeExternalIO(Exception):
            pass
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);project=root/'project'
            for folder in ['src','config','prompts']:
                shutil.copytree(ROOT/folder,project/folder)
            runtime=root/'runtime.py';runtime.write_text('fixture runner\n')
            tasks=root/'tasks.json';tasks.write_text('{"episodes": []}\n')
            env=dict(Path=Path,digest=digest,PROJECT=project,
                     read_json=lambda p:json.loads(Path(p).read_text()),
                     subprocess=types.SimpleNamespace(check_output=lambda *a,**k:'fixture-commit\n'))
            exec(compile(ast.Module(body=[original,fingerprint,freeze],type_ignores=[]),'<preflight test>','exec'),env)
            expected=env['project_fingerprint']()  # Independent original implementation.
            reference=dict(tasks_sha256=digest(tasks),project_commit='fixture-commit',
                project_files=expected,runtime_sha256=digest(runtime),population=300,elites=10,microbatch=16,
                checkpoint_lock=json.loads((project/'config/wall.release.json').read_text()))
            protocol=root/'protocol.json';protocol.write_text(json.dumps(reference))
            config=dict(replicates=3,test_count=12,model_id='fixture-model',fixed_depths=[15,30],
                        max_calls=40,max_chars=200000,max_turns=45,output_tokens=32768,http_timeout=900,
                        population=300,elites=10,microbatch=16)
            def halt_before_external_io():
                raise BeforeExternalIO()
            subject=types.SimpleNamespace(project=project,runtime=runtime,config=config,
                protocol_path=root/'absent-study.json',prepare=halt_before_external_io)
            subject.fingerprint=types.MethodType(env['fingerprint'],subject)
            # Valid integration reaches the first external-I/O boundary.
            with self.assertRaises(BeforeExternalIO):env['freeze'](subject,tasks,protocol)
            (project/'src/wsp_stopper/__init__.py').write_text('# changed\n')
            with self.assertRaisesRegex(ValueError,'Source fingerprint differs'):
                env['freeze'](subject,tasks,protocol)
            env['subprocess'].check_output=lambda *a,**k:'wrong-commit\n'
            with self.assertRaisesRegex(ValueError,'Project commit differs'):
                env['freeze'](subject,tasks,protocol)


if __name__=='__main__':
    unittest.main()
