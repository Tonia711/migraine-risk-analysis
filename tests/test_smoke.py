from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def can_import_heavy_deps() -> bool:
    """
    Smoke tests should not crash the whole test runner.
    If numpy/pandas/sklearn imports segfault (e.g., restricted env),
    they will exit with non-zero code; we skip the training run then.
    """
    cmd = [
        sys.executable,
        "-c",
        "import numpy as np; import pandas as pd; import sklearn; print('ok')",
    ]
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    return proc.returncode == 0


class SmokeTest(unittest.TestCase):
    def test_public_demo_outputs(self) -> None:
        sample_csv = REPO_ROOT / "Data" / "sample" / "final_modeling_table_sample.csv"
        self.assertTrue(sample_csv.exists(), f"missing sample file: {sample_csv}")

        if not can_import_heavy_deps():
            self.skipTest("numpy/pandas/sklearn not available (or segfault in this environment).")

        out_dir = REPO_ROOT / "outputs_smoke"
        if out_dir.exists():
            # keep it simple: remove only known outputs directory
            # (use rm via subprocess to avoid filesystem utilities being overly strict)
            subprocess.run(["rm", "-rf", str(out_dir)], cwd=REPO_ROOT, check=False)

        cmd = [
            sys.executable,
            "src/cli.py",
            "--data-mode",
            "sample",
            "--stage",
            "all",
            "--output-dir",
            str(out_dir),
        ]
        env = os.environ.copy()
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("OPENBLAS_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        env.setdefault("NUMEXPR_NUM_THREADS", "1")

        proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, f"cli failed: {proc.stderr}\n{proc.stdout}")

        metrics_csv = out_dir / "dm_metrics_summary.csv"
        self.assertTrue(metrics_csv.exists(), f"missing metrics: {metrics_csv}")

        figs_dir = out_dir / "figs"
        self.assertTrue(figs_dir.exists(), f"missing figs dir: {figs_dir}")

        # A couple of critical plots
        self.assertTrue(any(figs_dir.glob("roc_*.png")), "missing roc_*.png")
        self.assertTrue(any(figs_dir.glob("pr_*.png")), "missing pr_*.png")
        self.assertTrue(any(figs_dir.glob("cm_*.png")), "missing cm_*.png")
        self.assertTrue((figs_dir / "odds_ratio_topk.png").exists(), "missing odds_ratio_topk.png")
        self.assertTrue(
            any(figs_dir.glob("permutation_importance_*.png")),
            "missing permutation_importance_*.png",
        )


if __name__ == "__main__":
    unittest.main()

