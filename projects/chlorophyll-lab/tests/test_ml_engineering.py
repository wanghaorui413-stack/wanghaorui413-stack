import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from chlorophyll.common import FEATURES
from chlorophyll.grouping import assign_split_groups
from chlorophyll.train import grouped_scores


class MLEngineeringTests(unittest.TestCase):
    def test_compound_names_and_shared_sources_are_isolated(self):
        frame = pd.DataFrame(
            {
                "waterbody": ["north sea", "sognefjord/north sea", "sognefjord", "other"],
                "source_id": ["a", "a", "b", "c"],
            }
        )
        result, audit = assign_split_groups(frame)
        self.assertEqual(result.split_group.iloc[:3].nunique(), 1)
        self.assertNotEqual(result.split_group.iloc[0], result.split_group.iloc[3])
        self.assertEqual(audit["component_count"], 2)

    def test_macro_metric_does_not_weight_large_waterbody_more(self):
        metric = grouped_scores(np.ones(101), np.r_[np.ones(100), 10.0], np.array(["large"] * 100 + ["small"]))
        self.assertAlmostEqual(metric["macro_waterbody_log10_rmse"], 0.5)
        self.assertLess(metric["log10_rmse"], 0.1)

    def test_clean_home_prepare_train_evaluate_rebuild_and_retry(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            home = Path(temporary)
            rng = np.random.default_rng(15)
            x = rng.uniform(0.001, 0.03, (300, len(FEATURES)))
            frame = pd.DataFrame(x, columns=FEATURES)
            frame["sample_id"] = [f"{i:05d}" for i in range(300)]
            frame["waterbody"] = [f"water_{i // 15}" for i in range(300)]
            frame["chlorophyll_ug_l"] = np.exp(x[:, 15] * 40) + rng.uniform(0.1, 1, 300)
            frame.to_csv(home / "fixture.csv", index=False)
            script = """
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch
import os,pandas as pd
from chlorophyll.common import ROOT,read_json
from chlorophyll.data import prepare
from chlorophyll.train import train
from chlorophyll.evaluate import evaluate,rebuild_report
prepare('fixture_v2',ROOT/'fixture.csv','test fixture, not observed performance')
args=Namespace(dataset='fixture_v2',run='fixture_run',epochs=1,patience=1,lr=.001,batch_size=64,seed=42)
with patch('chlorophyll.train.fit_network',side_effect=RuntimeError('injected failure')):
    try: train(args)
    except RuntimeError: pass
assert not (ROOT/'runs'/'fixture_run').exists()
train(args)
result=evaluate('fixture_run')
run=ROOT/'runs'/'fixture_run'
assert 'macro_waterbody_log10_rmse' in result
assert read_json(run/'model_card.json')['selection_metric']=='validation macro_waterbody_log10_rmse'
assert pd.read_csv(run/'test_predictions.csv',dtype={'sample_id':str}).sample_id.str.len().eq(5).all()
(run/'report.html').unlink()
with patch('chlorophyll.evaluate.Predictor',side_effect=AssertionError('must not rerun inference')):
    rebuild_report('fixture_run')
assert (run/'report.html').exists()
try: evaluate('fixture_run')
except FileExistsError: pass
else: raise AssertionError('immutable evaluation was overwritten')
print('CLEAN_HOME_E2E_OK')
"""
            environment = os.environ.copy()
            environment["CHLOROPHYLL_HOME"] = str(home)
            environment["PYTHONIOENCODING"] = "utf-8"
            result = subprocess.run(
                [sys.executable, "-c", script],
                env=environment,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=120,
            )
            self.assertEqual(result.returncode, 0, result.stdout + "\n" + result.stderr)
            self.assertIn("CLEAN_HOME_E2E_OK", result.stdout)


if __name__ == "__main__":
    unittest.main()
