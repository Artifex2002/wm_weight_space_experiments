import ast
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, List
from unittest.mock import patch
import numpy as np
from wsp_stopper.core import write_json
from wsp_stopper.cem_hook import CEMHook
from wsp_stopper.upstream_patch import patched_text


def weights():
    return dict(W1=[[0]*8 for _ in range(16)], b1=[0]*16,W2=[[0]*16 for _ in range(2)],b2=[0,1])

class HookTests(unittest.TestCase):
    def test_independent_solves(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'w.json';write_json(p,weights())
            a=CEMHook(p,np.ones((6,10)),30);b=CEMHook(p,np.ones((6,10)),30)
            for k in range(1,4):
                self.assertEqual(a.after_refit([1,2,3],[0,1],np.ones((6,10)),np.zeros(10)),k==3)
            self.assertEqual(b.features.k,0)
            with self.assertRaises(ValueError): CEMHook(p,np.ones((6,10)),10)
            with self.assertRaises(ValueError): CEMHook(p,np.ones((6,10)),30,True)

    def test_bad_patch(self):
        with self.assertRaises(ValueError): patched_text('not upstream','planner')

class UpstreamTests(unittest.TestCase):
    def setUp(self):
        root=os.environ.get('WSP_TEST_UPSTREAM')
        if not root: self.skipTest('Set WSP_TEST_UPSTREAM to test actual pinned CEM on CPU')
        try: import torch
        except ImportError: self.skipTest('PyTorch required')
        self.torch=torch
        lock=json.loads(Path('config/wall.release.json').read_text())
        p=Path(root)/lock['planner_source'];backup=p.with_suffix(p.suffix+'.wsp-original')
        self.original=(backup if backup.exists() else p).read_text()
        self.assertEqual(hashlib.sha256(self.original.encode()).hexdigest(),lock['planner_sha256'])

    def planner(self,source,iterations):
        torch=self.torch
        class Base:
            def __init__(self,unroll): self.unroll=unroll
            def cost_function(self,actions,z): return actions.square().mean(dim=(0,2))
        cls=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='CEMPlanner')
        env=dict(torch=torch,Planner=Base,Callable=Callable,List=List,PlanningResult=lambda **kw:SimpleNamespace(**kw))
        exec(compile(ast.Module(body=[cls],type_ignores=[]),'pinned_cem.py','exec'),env)
        p=env['CEMPlanner'](lambda z,act_suffix:act_suffix.clone(),iterations=iterations,
            num_samples=20,num_elites=5,horizon=6,action_dim=10,num_act_stepped=6,
            local_generator=torch.Generator().manual_seed(83))
        p.device=torch.device('cpu');return p

    def test_disabled_hook_matches_upstream_exactly(self):
        with patch.dict(os.environ,{},clear=False):
            os.environ.pop('WSP_STOPPER_WEIGHTS',None)
            base=self.planner(self.original,30).plan(None)
            changed=self.planner(patched_text(self.original,'planner'),30).plan(None)
            self.torch.testing.assert_close(base.actions,changed.actions,rtol=0,atol=0)
            self.torch.testing.assert_close(base.losses,changed.losses,rtol=0,atol=0)

    def test_stopped_plan_matches_fixed_prefix_and_resets(self):
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/'w.json';write_json(path,weights())
            base=self.planner(self.original,3).plan(None)
            with patch.dict(os.environ,{'WSP_STOPPER_WEIGHTS':str(path)}):
                p=self.planner(patched_text(self.original,'planner'),30);early=p.plan(None)
                self.assertEqual(p.wsp_last_depth,3)
                self.torch.testing.assert_close(base.actions,early.actions,rtol=0,atol=0)
                self.torch.testing.assert_close(base.losses,early.losses,rtol=0,atol=0)
                p.plan(None,steps_left=2)
                self.assertEqual(len(p.wsp_last_trace),3)
