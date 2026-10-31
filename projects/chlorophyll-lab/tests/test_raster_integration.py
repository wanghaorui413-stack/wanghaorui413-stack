import tempfile
import unittest
from pathlib import Path
import importlib.util
import joblib
import numpy as np
from sklearn.dummy import DummyRegressor
from chlorophyll.common import write_json, sha256
from chlorophyll.satellite_model import BANDS


@unittest.skipUnless(
    importlib.util.find_spec("rasterio") and importlib.util.find_spec("netCDF4"), "install satellite extra"
)
class RasterIntegrationTests(unittest.TestCase):
    def test_georeferencing_mask_and_empty_product_rejection(self):
        import xarray as xr
        import rasterio
        from pyproj import CRS
        from chlorophyll.satellite import predict_raster

        # 仅工程结构夹具；不计入真实卫星精度或端到端影像证据。
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = DummyRegressor(strategy="constant", constant=1.0)
            model.fit(np.ones((4, 7)), np.ones(4))
            joblib.dump(model, root / "model.joblib")
            write_json(
                root / "model.json",
                {
                    "sensor": "S2B_MSI",
                    "bands": BANDS,
                    "input_unit": "sr^-1",
                    "artifact_sha256": sha256(root / "model.joblib"),
                    "validation_status": "test fixture only",
                },
            )
            variables = {
                f"Rrs_{wave}": (("y", "x"), np.full((3, 4), 0.005)) for wave in [492, 559, 665, 704, 739, 780, 864]
            }
            flags = np.zeros((3, 4), dtype=np.int32)
            flags[0, 0] = 1
            variables["l2_flags"] = (("y", "x"), flags)
            ds = xr.Dataset(
                variables,
                coords={"x": 300010 + np.arange(4) * 20, "y": 4200010 - np.arange(3) * 20},
                attrs={"sensor": "S2B_MSI"},
            )
            ds["transverse_mercator"] = xr.DataArray(0, attrs=CRS.from_epsg(32652).to_cf())
            path = root / "input.nc"
            ds.to_netcdf(path)
            predict_raster(path, root, root / "output")
            with rasterio.open(root / "output/chlorophyll_ug_l.tif") as raster:
                self.assertEqual(raster.crs.to_epsg(), 32652)
                self.assertEqual(raster.res, (20.0, 20.0))
                self.assertEqual(raster.read(1)[0, 0], -9999)
                self.assertAlmostEqual(raster.read(1)[1, 1], 10.0)
            ds["l2_flags"] = (("y", "x"), np.ones((3, 4), dtype=np.int32))
            invalid = root / "invalid.nc"
            ds.to_netcdf(invalid)
            with self.assertRaises(ValueError):
                predict_raster(invalid, root, root / "rejected")
            self.assertFalse((root / "rejected").exists())
