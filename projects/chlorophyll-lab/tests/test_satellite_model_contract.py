import tempfile
import unittest
from pathlib import Path
import joblib
import numpy as np
from sklearn.dummy import DummyRegressor
from chlorophyll.common import write_json, sha256
from chlorophyll.satellite_model import BANDS, SatellitePredictor, read_rsr


class SatelliteModelContractTests(unittest.TestCase):
    def test_response_weights_integrate_constant_spectrum(self):
        text = "\n".join(f";; BAND {b[1:]}\n400 0\n401 1\n402 0" for b in BANDS)
        for wavelengths, weights in read_rsr(text).values():
            self.assertAlmostEqual(weights.sum(), 1.0)
            self.assertAlmostEqual(np.full(len(weights), 0.007) @ weights, 0.007)

    def test_standalone_seven_band_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = DummyRegressor(strategy="constant", constant=1.0)
            model.fit(np.ones((4, 7)), np.ones(4))
            joblib.dump(model, root / "model.joblib")
            card = {
                "sensor": "S2A_MSI",
                "bands": BANDS,
                "input_unit": "sr^-1",
                "artifact_sha256": sha256(root / "model.joblib"),
            }
            write_json(root / "model.json", card)
            predictor = SatellitePredictor(root)
            np.testing.assert_allclose(predictor.predict(np.full((3, 7), 0.005)), 10.0)
            with self.assertRaises(ValueError):
                predictor.predict(np.ones((3, 71)))
            card["input_unit"] = "dimensionless"
            write_json(root / "model.json", card)
            with self.assertRaises(ValueError):
                SatellitePredictor(root)
