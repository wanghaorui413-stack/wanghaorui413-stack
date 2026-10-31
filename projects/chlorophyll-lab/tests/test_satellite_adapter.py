import unittest
import numpy as np
from chlorophyll.satellite import identify_sensor, match_bands, as_rrs, validate_grid, write_acolite_settings
import tempfile
from pathlib import Path


class SatelliteContractTests(unittest.TestCase):
    def test_sensor_rejects_c_and_missing(self):
        self.assertEqual(identify_sensor({"sensor": "S2A_MSI"}), "S2A_MSI")
        for attrs in [{}, {"sensor": "S2C_MSI"}]:
            with self.assertRaises(ValueError):
                identify_sensor(attrs)

    def test_reject_surface_reflectance(self):
        with self.assertRaises(ValueError):
            match_bands(["rhos_492"])
        with self.assertRaises(ValueError):
            as_rrs([1], "rhot_492")
        np.testing.assert_allclose(as_rrs([np.pi], "rhow_492"), [1])
        np.testing.assert_allclose(as_rrs([0.01], "Rrs_492"), [0.01])

    def test_band_match(self):
        variables = ["rhow_" + str(w) for w in [492, 560, 665, 704, 740, 783, 865]]
        self.assertEqual(match_bands(variables), variables)

    def test_grid_requires_20m_north_up(self):
        self.assertEqual(validate_grid([10, 30], [50, 30]), (0.0, 60.0))
        with self.assertRaises(ValueError):
            validate_grid([10, 20], [50, 40])
        with self.assertRaises(ValueError):
            validate_grid([10, 30], [30, 50])

    def test_settings_requests_above_water_rrs_and_projection(self):
        temporary_root = Path(__file__).resolve().parents[1] / "tmp"
        temporary_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as directory:
            settings = write_acolite_settings("S2A_MSIL1C_example.zip", directory, [31, 120, 31.1, 120.1]).read_text()
            self.assertIn("l2w_parameters=Rrs_*", settings)
            self.assertNotIn("l2w_parameters=rrs_", settings)
            self.assertIn("s2_target_res=20", settings)
            self.assertIn("netcdf_projection=True", settings)
            self.assertIn("rsr_version=None", settings)
