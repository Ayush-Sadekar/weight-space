"""Unit tests for edit/inspect/probe tools and score.py."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "harness"
TOOLS = ROOT / "tools"
sys.path.insert(0, str(HARNESS))

from models import MLP  # noqa: E402
from score import integrity_check, score_run, weight_delta  # noqa: E402


def run_tool(cwd: Path, script: str, args: list[str], env: dict | None = None) -> subprocess.CompletedProcess:
    e = os.environ.copy()
    if env:
        e.update(env)
    return subprocess.run(
        [sys.executable, str(TOOLS / script), *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        env=e,
    )


def make_tiny_task(tmp: Path, *, probe_remaining: int = 5, disabled: bool = False) -> Path:
    """Hand-made fixed model under tmp/run/{task,hidden}."""
    run = tmp / "run"
    task = run / "task"
    hidden = run / "hidden"
    task.mkdir(parents=True)
    hidden.mkdir(parents=True)

    # Fixed 2→3→2 MLP with deterministic weights
    weights = {
        "W0": np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float64),
        "b0": np.array([0.0, 0.0, 0.0], dtype=np.float64),
        "W1": np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64),
        "b1": np.array([0.0, 0.0], dtype=np.float64),
    }
    model = MLP([2, 3, 2], seed=0, weights=weights)
    model.save(task / "model.npz")
    model.save(hidden / "start.npz")
    model.save(hidden / "original.npz")

    # Eval: class 0 when x0 > x1
    X = np.array([[1.0, 0.0], [0.0, 1.0], [0.8, 0.1], [0.1, 0.9]], dtype=np.float64)
    y = np.array([0, 1, 0, 1], dtype=np.int64)
    np.savez(hidden / "eval.npz", X=X, y=y)

    budget = {
        "disabled": disabled,
        "remaining": 0 if disabled else probe_remaining,
        "used": 0,
        "initial": 0 if disabled else probe_remaining,
    }
    (hidden / "probe_budget.json").write_text(json.dumps(budget, indent=2) + "\n")
    (hidden / "answer_key.json").write_text(
        json.dumps(
            {
                "task": "remove_class",
                "target_class": 1,
                "metric": "accuracy_on_class_1",
                "higher_is_better": False,
                "success_threshold_acc3": 0.15,
                "regression_floor_other": 0.0,
                "baselines": {},
            },
            indent=2,
        )
        + "\n"
    )
    (task / "edits.jsonl").write_text("")
    (task / "vectors").mkdir()
    return run


class TestEditOps(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.run = make_tiny_task(self.tmp)
        self.task = self.run / "task"

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_inspect_list_stats_show_svd(self) -> None:
        r = run_tool(self.task, "inspect.py", ["list"])
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("W0", r.stdout)
        r = run_tool(self.task, "inspect.py", ["stats", "W0"])
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("mean=", r.stdout)
        r = run_tool(self.task, "inspect.py", ["show", "W0", "--rows", "0:2", "--cols", "0:2"])
        self.assertEqual(r.returncode, 0, r.stderr)
        r = run_tool(self.task, "inspect.py", ["svd", "W0", "--k", "2"])
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("singular_values", r.stdout)

    def test_show_respects_max_print(self) -> None:
        r = run_tool(
            self.task,
            "inspect.py",
            ["show", "W0"],
            env={"WS_MAX_PRINT": "1"},
        )
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("refusing", r.stderr + r.stdout)

    def test_set_add_scale_zero(self) -> None:
        r = run_tool(self.task, "edit.py", ["set", "W1", "0", "0", "2.5"])
        self.assertEqual(r.returncode, 0, r.stderr)
        m = MLP.load(self.task / "model.npz")
        self.assertAlmostEqual(float(m.weights["W1"][0, 0]), 2.5)

        r = run_tool(self.task, "edit.py", ["add", "W1", "0", "0", "0.5"])
        self.assertEqual(r.returncode, 0, r.stderr)
        m = MLP.load(self.task / "model.npz")
        self.assertAlmostEqual(float(m.weights["W1"][0, 0]), 3.0)

        r = run_tool(self.task, "edit.py", ["scale", "W1", "--rows", "1:2", "0.0"])
        self.assertEqual(r.returncode, 0, r.stderr)
        m = MLP.load(self.task / "model.npz")
        self.assertTrue(np.allclose(m.weights["W1"][1], 0.0))

        r = run_tool(self.task, "edit.py", ["zero", "W0", "--rows", "2:3"])
        self.assertEqual(r.returncode, 0, r.stderr)
        m = MLP.load(self.task / "model.npz")
        self.assertTrue(np.allclose(m.weights["W0"][2], 0.0))

        edits = (self.task / "edits.jsonl").read_text().strip().splitlines()
        self.assertEqual(len(edits), 4)

    def test_save_vector_and_ablate(self) -> None:
        # out-side vector length = rows of W1 = 2
        r = run_tool(self.task, "edit.py", ["save-vector", "v", "1.0,0.0"])
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue((self.task / "vectors" / "v.npy").is_file())

        before = MLP.load(self.task / "model.npz").weights["W1"].copy()
        r = run_tool(self.task, "edit.py", ["ablate", "W1", "vectors/v.npy", "--side", "out"])
        self.assertEqual(r.returncode, 0, r.stderr)
        after = MLP.load(self.task / "model.npz").weights["W1"]
        # Row 0 (along [1,0]) should be projected out → near zero row contribution
        self.assertTrue(np.allclose(after[0], 0.0, atol=1e-9))
        self.assertFalse(np.allclose(before, after))

    def test_ablate_in_side(self) -> None:
        r = run_tool(self.task, "edit.py", ["save-vector", "vin", "1.0,0.0,0.0"])
        self.assertEqual(r.returncode, 0, r.stderr)
        r = run_tool(self.task, "edit.py", ["ablate", "W1", "vectors/vin.npy", "--side", "in"])
        self.assertEqual(r.returncode, 0, r.stderr)
        after = MLP.load(self.task / "model.npz").weights["W1"]
        # Column 0 removed
        self.assertTrue(np.allclose(after[:, 0], 0.0, atol=1e-9))


class TestProbeBudget(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.run = make_tiny_task(self.tmp, probe_remaining=2)
        self.task = self.run / "task"

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_probe_decrements(self) -> None:
        r = run_tool(self.task, "probe.py", ["--inline", "[1.0, 0.0]"])
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
        self.assertIn("argmax=", r.stdout)
        budget = json.loads((self.run / "hidden" / "probe_budget.json").read_text())
        self.assertEqual(budget["remaining"], 1)
        self.assertEqual(budget["used"], 1)

        r = run_tool(self.task, "probe.py", ["--inline", "[0.0, 1.0]"])
        self.assertEqual(r.returncode, 0, r.stderr)
        budget = json.loads((self.run / "hidden" / "probe_budget.json").read_text())
        self.assertEqual(budget["remaining"], 0)

        r = run_tool(self.task, "probe.py", ["--inline", "[0.0, 1.0]"])
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("exhausted", r.stderr + r.stdout)

    def test_probes_disabled(self) -> None:
        run = make_tiny_task(self.tmp / "dis", disabled=True)
        r = run_tool(run / "task", "probe.py", ["--inline", "[1.0, 0.0]"])
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("probes disabled", r.stdout + r.stderr)

    def test_bad_input_does_not_decrement_budget(self) -> None:
        before = json.loads((self.run / "hidden" / "probe_budget.json").read_text())
        r = run_tool(self.task, "probe.py", ["--inline", "[1.0]"])  # wrong dim
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("input dim", r.stderr + r.stdout)
        after = json.loads((self.run / "hidden" / "probe_budget.json").read_text())
        self.assertEqual(after["remaining"], before["remaining"])
        self.assertEqual(after["used"], before["used"])

        r = run_tool(self.task, "probe.py", ["--inline", "not-json"])
        self.assertNotEqual(r.returncode, 0)
        after = json.loads((self.run / "hidden" / "probe_budget.json").read_text())
        self.assertEqual(after["remaining"], before["remaining"])
        self.assertEqual(after["used"], before["used"])


class TestPathSafety(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.run = make_tiny_task(self.tmp)
        self.task = self.run / "task"

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_ablate_rejects_outside_path(self) -> None:
        outside = self.tmp / "evil.npy"
        np.save(outside, np.array([1.0, 0.0]))
        r = run_tool(self.task, "edit.py", ["ablate", "W1", str(outside), "--side", "out"])
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("escapes", r.stderr + r.stdout)

    def test_save_vector_rejects_path_name(self) -> None:
        r = run_tool(self.task, "edit.py", ["save-vector", "../x", "1,2"])
        self.assertNotEqual(r.returncode, 0)

    def test_probe_input_rejects_outside(self) -> None:
        outside = self.tmp / "in.json"
        outside.write_text("[1.0, 0.0]")
        r = run_tool(self.task, "probe.py", [str(outside)])
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("escapes", r.stderr + r.stdout)


class TestScore(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.run = make_tiny_task(self.tmp)
        # Register under ROOT/runs for score_run
        self.run_id = "_test_handmade_score"
        self.dest = ROOT / "runs" / self.run_id
        if self.dest.exists():
            shutil.rmtree(self.dest)
        shutil.copytree(self.run, self.dest)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)
        if self.dest.exists():
            shutil.rmtree(self.dest)

    def test_score_on_handmade(self) -> None:
        # Apply a minimal edit: zero output row for class 1
        task = self.dest / "task"
        r = run_tool(task, "edit.py", ["zero", "W1", "--rows", "1:2"])
        self.assertEqual(r.returncode, 0, r.stderr)

        score = score_run(self.run_id)
        self.assertTrue(score["integrity_ok"])
        self.assertEqual(score["n_edits"], 1)
        self.assertIsNotNone(score["weight_change"])
        self.assertGreater(score["weight_change"]["n_changed"], 0)
        self.assertTrue((self.dest / "score.json").is_file())

    def test_integrity_fails_on_shape_change(self) -> None:
        start = MLP.load(self.dest / "hidden" / "start.npz")
        # Replace with different architecture
        bad = MLP([2, 4, 2], seed=1)
        bad.save(self.dest / "task" / "model.npz")
        edited = MLP.load(self.dest / "task" / "model.npz")
        failures = integrity_check(
            start, edited, self.dest / "task" / "model.npz", self.dest / "hidden" / "start.npz"
        )
        self.assertTrue(any("layer_sizes" in f or "shape" in f for f in failures))

    def test_weight_delta(self) -> None:
        a = MLP.load(self.dest / "hidden" / "start.npz")
        b = a.copy()
        b.weights["W0"][0, 0] += 1.0
        d = weight_delta(a, b)
        self.assertEqual(d["n_changed"], 1)
        self.assertAlmostEqual(d["l2"], 1.0)


class TestMakeTaskSmoke(unittest.TestCase):
    def test_make_fix_digit(self) -> None:
        from make_task import make_task

        rid = "_test_make_fix_digit"
        dest = ROOT / "runs" / rid
        if dest.exists():
            shutil.rmtree(dest)
        try:
            make_task("fix_digit", 0, "probes_10", rid)
            self.assertTrue((dest / "task" / "model.npz").is_file())
            self.assertTrue((dest / "hidden" / "answer_key.json").is_file())
            budget = json.loads((dest / "hidden" / "probe_budget.json").read_text())
            self.assertEqual(budget["remaining"], 10)
            md = (dest / "task" / "task.md").read_text()
            self.assertIn("python ../../../tools/inspect.py", md)
        finally:
            if dest.exists():
                shutil.rmtree(dest)


if __name__ == "__main__":
    unittest.main()
