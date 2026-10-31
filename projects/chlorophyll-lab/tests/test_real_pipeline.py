import io
import unittest
import zipfile
import tempfile
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from chlorophyll.common import ROOT, FEATURES, TARGET, read_json
from chlorophyll.data import normalize_group, split_groups, read_prediction_csv, prepare
from chlorophyll.inference import Predictor
from chlorophyll.metrics import invert_target, scores
from chlorophyll.models import create_model


def sample_csv():
    frame = pd.DataFrame(np.full((3, len(FEATURES)), 0.005), columns=FEATURES)
    frame.insert(0, "sample_id", ["a", "b", "c"])
    return frame.to_csv(index=False)


class ContractTests(unittest.TestCase):
    def test_groups_disjoint_even_with_name_variants(self):
        groups = [f"LAKE {i}" for i in range(20) for _ in range(10)]
        frame = pd.DataFrame({"waterbody": groups})
        frame.loc[0, "waterbody"] = "  lake   0 "
        split = split_groups(frame)
        sets = [set(frame.iloc[indices].waterbody.map(normalize_group)) for indices in split.values()]
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        self.assertEqual(sum(map(len, split.values())), len(frame))

    def test_input_order_is_aligned_by_wavelength_names(self):
        text = sample_csv()
        frame, x = read_prediction_csv(text)
        _, reversed_x = read_prediction_csv(frame[list(reversed(frame.columns))].to_csv(index=False))
        np.testing.assert_array_equal(x, reversed_x)

    def test_bad_spectral_inputs_rejected(self):
        base = pd.read_csv(io.StringIO(sample_csv()))
        variants = []
        variants.append(base.drop(columns=FEATURES[0]))
        missing = base.copy()
        missing.loc[0, FEATURES[0]] = np.nan
        variants.append(missing)
        wrong_unit = base.copy()
        wrong_unit.loc[0, FEATURES[0]] = 900
        variants.append(wrong_unit)
        dupid = base.copy()
        dupid.loc[1, "sample_id"] = "a"
        variants.append(dupid)
        for frame in variants:
            with self.assertRaises(ValueError):
                read_prediction_csv(frame.to_csv(index=False))
        with self.assertRaises(ValueError):
            read_prediction_csv("sample_id,rrs_400,rrs_400\na,0.1,0.2\n")

    def test_log_transform_roundtrip_and_metrics(self):
        y = np.array([0.1, 1.0, 10.0, 100.0])
        mean, std = np.log10(y).mean(), np.log10(y).std()
        np.testing.assert_allclose(invert_target((np.log10(y) - mean) / std, mean, std), y)
        metric = scores(y, y * 2)
        self.assertAlmostEqual(metric["median_factor_error"], 2.0)
        self.assertAlmostEqual(metric["log10_rmse"], np.log10(2))
        with self.assertRaises(ValueError):
            scores(y, np.zeros_like(y))

    def test_custom_csv_adapter_freezes_valid_group_split(self):
        # 这里只用结构测试数据验证用户CSV入口，不计入任何实测性能。
        rng = np.random.default_rng(9)
        frame = pd.DataFrame(rng.uniform(0.001, 0.02, (240, len(FEATURES))), columns=FEATURES)
        frame.insert(0, "sample_id", [f"test_{i}" for i in range(240)])
        frame["waterbody"] = [f"water_{i // 12}" for i in range(240)]
        frame[TARGET] = rng.uniform(1, 20, 240)
        with tempfile.TemporaryDirectory() as temporary:
            from pathlib import Path

            root = Path(temporary)
            path = root / "source.csv"
            frame.to_csv(path, index=False)
            with patch("chlorophyll.data.named_path", return_value=root / "prepared"):
                directory = prepare("contract_test", path, "structural test fixture, not real observations")
            quality = read_json(directory / "quality.json")
            self.assertEqual(quality["retained_samples"], 240)
            self.assertEqual(sum(s["samples"] for s in quality["splits"].values()), 240)
            self.assertEqual(quality["source"]["target_unit"], "ug/L")

    def test_neural_models_can_fit_tiny_batch(self):
        torch.set_num_threads(2)
        for name in ["mlp", "cnn"]:
            torch.manual_seed(8)
            x = torch.randn(8, 1, len(FEATURES))
            y = torch.linspace(-1, 1, 8)
            model = create_model(name)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
            # 固定小批次、关闭Dropout，验证模型能表达并学习目标。
            model.eval()
            initial = torch.nn.functional.mse_loss(model(x), y).item()
            for _ in range(100):
                optimizer.zero_grad()
                loss = torch.nn.functional.mse_loss(model(x), y)
                loss.backward()
                optimizer.step()
            final = torch.nn.functional.mse_loss(model(x), y).item()
            self.assertLess(final, initial * 0.2, (name, initial, final))


@unittest.skipUnless((ROOT / "runs" / "gloria_v1" / "model_card.json").exists(), "train gloria_v1 first")
class RealDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.predictor = Predictor("gloria_v1")
        cls.directory = ROOT / "data" / "prepared" / "gloria"

    def test_scalers_only_come_from_training(self):
        frame = pd.read_csv(self.directory / "train.csv")
        np.testing.assert_allclose(self.predictor.prep["xm"], frame[FEATURES].to_numpy().mean(0), rtol=1e-12)
        self.assertAlmostEqual(float(self.predictor.prep["ym"]), np.log10(frame[TARGET]).mean(), places=12)

    def test_real_labels_are_pure_chla_and_pass_qc(self):
        frame = pd.read_csv(self.directory / "clean.csv")
        with zipfile.ZipFile(ROOT / "data" / "gloria" / "raw" / "GLORIA-2022.zip") as z:
            meta = pd.read_csv(z.open("GLORIA_2022/GLORIA_meta_and_lab.csv")).set_index("GLORIA_ID")
            qc = pd.read_csv(z.open("GLORIA_2022/GLORIA_qc_flags.csv")).dropna(how="all").set_index("GLORIA_ID")
        np.testing.assert_allclose(frame[TARGET], meta.loc[frame.sample_id, "Chla"])
        self.assertTrue(qc.loc[frame.sample_id, "Flagged"].eq(0).all())
        self.assertTrue(qc.loc[frame.sample_id, "Suspect"].eq(0).all())

    def test_split_groups_and_spectra_do_not_overlap(self):
        quality = read_json(self.directory / "quality.json")
        groups = [set(quality["split_groups"][key]) for key in ["train", "val", "test"]]
        self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
        clean = pd.read_csv(self.directory / "clean.csv")
        self.assertFalse(clean.duplicated(FEATURES).any())

    def test_reload_reproduces_all_validation_metrics(self):
        frame = pd.read_csv(self.directory / "val.csv")
        prediction = self.predictor.all_predictions(frame[FEATURES].to_numpy())
        for name, values in prediction.items():
            metric = scores(frame[TARGET].to_numpy(), values)
            self.assertAlmostEqual(
                metric["log10_rmse"], self.predictor.card["validation_metrics"][name]["log10_rmse"], places=6
            )
        text = (self.directory / "prediction_example.csv").read_text(encoding="utf-8")
        self.assertEqual(len(self.predictor.from_csv(text)), 5)


if __name__ == "__main__":
    unittest.main()
