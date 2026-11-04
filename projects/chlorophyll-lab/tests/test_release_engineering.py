import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import hashlib
import numpy as np
from chlorophyll.common import FEATURES, SCHEMA_VERSION, write_json, sha256
from chlorophyll.inference import Predictor
from chlorophyll.release import export_model


class StandaloneReleaseTests(unittest.TestCase):
    def test_geometric_mean_release_is_independent_and_traceable(self):
        # 独立最小模型产物，无真实数据、网络或既有gloria_v1依赖。
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            n = len(FEATURES)
            np.savez(
                source / "preprocess.npz",
                xm=np.zeros(n),
                xs=np.ones(n),
                ym=1.0,
                ys=1.0,
                train_min=np.zeros(n),
                train_max=np.full(n, 0.1),
                target_min=1.0,
                target_max=100.0,
            )
            write_json(
                source / "model_card.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "features": FEATURES,
                    "selected_model": "mean_log",
                    "artifact_hashes": {"preprocess.npz": sha256(source / "preprocess.npz")},
                },
            )
            with patch("chlorophyll.release.named_path", return_value=source):
                export_model("fixture", root / "release")
            predictor = Predictor(package_path=root / "release")
            text = "sample_id," + ",".join(FEATURES) + "\n00123," + ",".join(["0.01"] * n) + "\n"
            result = predictor.from_csv(text)
            self.assertEqual(result.sample_id.iloc[0], "00123")
            self.assertAlmostEqual(result.chlorophyll_ug_l.iloc[0], 10.0)
            self.assertEqual(result.input_sha256.iloc[0], hashlib.sha256(text.encode()).hexdigest())
            self.assertFalse(predictor.networks)
            self.assertIsNone(predictor.baselines)
            with self.assertRaises(ValueError):
                predictor.from_csv(text, "not_a_model")
            self.assertEqual(result.status.iloc[0], "ok")
            # 越出训练光谱范围（0–0.1 加 5% 宽容带）：保留行但不给浓度
            outside = text + "00124," + ",".join(["0.5"] * n) + "\n"
            rejected = predictor.from_csv(outside).iloc[1]
            self.assertEqual(rejected.status, "rejected")
            self.assertEqual(rejected.rejection_reason, "spectrum_outside_training_range")
            self.assertTrue(np.isnan(rejected.chlorophyll_ug_l))

    def test_out_of_range_prediction_is_rejected(self):
        from chlorophyll.inference import domain_check

        n = len(FEATURES)
        prep = {
            "train_min": np.zeros(n),
            "train_max": np.full(n, 0.1),
            "target_min": np.array(1.0),
            "target_max": np.array(100.0),
        }
        x = np.full((3, n), 0.05)
        x[1, 3] = 0.104  # 在宽容带内
        reasons = domain_check(x, np.array([10.0, 10.0, 1e6]), prep)
        self.assertEqual(reasons.tolist(), ["", "", "prediction_outside_training_range"])
        # 范围内但与随机森林相差超过10倍
        reasons = domain_check(x, np.array([10.0, 90.0, 10.0]), prep, reference=np.array([20.0, 5.0, 10.0]))
        self.assertEqual(reasons.tolist(), ["", "disagrees_with_random_forest", ""])
