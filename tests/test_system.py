import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from wsp_stopper.core import Features, Stopper, digest, write_json
from wsp_stopper.inspection import Inspector
from wsp_stopper.agent import run
from wsp_stopper.packet import verify


def weights():
    return {"W1": np.zeros((16,8)).tolist(), "b1": [0]*16,
            "W2": np.zeros((2,16)).tolist(), "b2": [0,0]}


def fixture(root):
    root.mkdir()
    (root/"tensors").mkdir()
    (root/"sources").mkdir()
    np.save(root/"tensors/00000.npy", np.arange(12,dtype=np.float32).reshape(3,4))
    (root/"sources/example.py").write_text("# synthetic fixture; not a DINO-WM checkpoint\n")
    m = {"schema": 1, "checkpoint_sha256": "synthetic", "provenance": "offline test only",
         "specification": {}, "tensors": {"weight": {"path": "tensors/00000.npy", "shape": [3,4],
         "sha256": digest(root/"tensors/00000.npy")}},
         "sources": {"sources/example.py": {"sha256": digest(root/"sources/example.py")}}}
    write_json(root/"manifest.json",m)
    (root/"manifest.sha256").write_text(digest(root/"manifest.json"))


class FakeAPI:
    def __init__(self, calls):
        self.calls = iter(calls)
        self.bodies = []
    def request(self, method, resource, body):
        import copy
        self.bodies.append(copy.deepcopy(body))
        name, args = next(self.calls)
        return {"status": "completed", "output": [{"type": "function_call", "call_id": str(len(self.bodies)),
            "name": name, "arguments": json.dumps(args)}]}


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.packet = self.root/"packet"
        fixture(self.packet)
        self.prompt = self.root/"prompt.md"
        self.prompt.write_text("test protocol")

    def test_features_numeric_example_and_reset(self):
        f = Features(np.ones((5,2)))
        x = f.update([4,8,12], [0,1], np.ones((5,2)), [0,0])
        np.testing.assert_allclose(x, [1/30,0,0,1,1/3,1/3,1,0], rtol=1e-6)
        x = f.update([2,4,9], [0,1], np.ones((5,2))*.5, [1,1])
        np.testing.assert_allclose(x, [2/30,.5,.25,.5,1/3,1/3,.5,1], rtol=1e-6)
        self.assertEqual(Features(np.ones((5,2))).k, 0)

    def test_invalid_cost_and_sigma(self):
        with self.assertRaises(ValueError):
            Features(np.zeros((5,2)))
        with self.assertRaises(ValueError):
            Features(np.ones((5,2))).update([-1,2],[0],np.ones((5,2)),[0,0])

    def test_stop_guards_and_tie(self):
        w = weights()
        s = Stopper(w)
        self.assertFalse(s.stop([0]*8, 3))
        self.assertTrue(s.stop([0]*8, 30))
        w["b2"][1] = 1
        s = Stopper(w)
        self.assertFalse(s.stop([0]*8, 2))
        self.assertTrue(s.stop([0]*8, 3))

    def test_invalid_weights(self):
        w = weights()
        w["b2"] = [0,float("nan")]
        with self.assertRaises(ValueError):
            Stopper(w)
        w["b2"] = [0]
        with self.assertRaises(ValueError):
            Stopper(w)

    def test_inspect_and_no_execution(self):
        i = Inspector(self.packet)
        self.assertEqual(i.call("read_tensor",dict(name="weight",offset=2,count=2))["values"],[2,3])
        self.assertAlmostEqual(i.call("tensor_stats",{"name":"weight"})["mean"],5.5)
        with self.assertRaises(ValueError):
            i.call("python",{"code":"print(1)"})
        with self.assertRaises(ValueError):
            i.call("read_source",dict(path="../../secret",start_line=1,line_count=1))
        with self.assertRaises(ValueError):
            i.call("read_tensor",dict(name="weight",offset=0,count=1025))

    def test_packet_tamper(self):
        (self.packet/"sources/example.py").write_text("changed")
        with self.assertRaises(ValueError):
            verify(self.packet)

    def test_api_loop_and_frozen_submission(self):
        api = FakeAPI([("read_tensor",dict(name="weight",offset=0,count=3)),("submit_weights",weights())])
        out = self.root/"run"
        path = run(self.packet,self.prompt,"mock-model",out,transport=api)
        self.assertEqual(json.loads(path.read_text()),weights())
        self.assertFalse(api.bodies[0]["store"])
        self.assertEqual(api.bodies[1]["input"][-2]["type"],"function_call")
        events = (out/"events.jsonl").read_text()
        self.assertIn('"accepted"',events)
        self.assertIn('"inspection_calls": 1',(out/"submission.json").read_text())

    def test_invalid_submission_ends_attempt(self):
        api = FakeAPI([("submit_weights",{"W1":[]})])
        with self.assertRaises(RuntimeError):
            run(self.packet,self.prompt,"mock",self.root/"run",transport=api)
        self.assertEqual(len(api.bodies),1)
        self.assertFalse((self.root/"run/weights.json").exists())

    def test_budget_restricts_tools(self):
        api = FakeAPI([("list_tensors",dict(offset=0,limit=1)),("submit_weights",weights())])
        run(self.packet,self.prompt,"mock",self.root/"run",max_calls=1,transport=api)
        self.assertEqual([x["name"] for x in api.bodies[1]["tools"]],["submit_weights"])

    def test_freeze_rejects_unresolved_spec_before_loading(self):
        # No torch import required to run the inspection/generation tests.
        from wsp_stopper.packet import freeze
        spec = self.root/"spec.json"
        write_json(spec,{"ready":False})
        with self.assertRaises(ValueError):
            freeze("missing.pth","",[],spec,"test",self.root/"out")


if __name__ == "__main__":
    unittest.main()
