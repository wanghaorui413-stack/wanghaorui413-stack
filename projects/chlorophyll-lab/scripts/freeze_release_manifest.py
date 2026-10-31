"""Freeze release artifact hashes after successful local verification."""

from datetime import datetime, timezone
from pathlib import Path

from chlorophyll.common import sha256, source_identity, write_json


def main():
    root = Path(__file__).resolve().parents[1]
    paths = [
        "dist/chlorophyll_lab-0.2.0-py3-none-any.whl",
        "requirements-engineering-lock.txt",
        "releases/gloria_engineering_v2/model_card.json",
        "releases/S2A_MSI_gloria_v2/model.json",
        "releases/S2B_MSI_gloria_v2/model.json",
        "predictions/hanriver_s2b_20240531/chlorophyll_ug_l.tif",
        "predictions/hanriver_s2b_20240531/valid_water_mask.tif",
        "predictions/hanriver_s2b_20240531/provenance.json",
        "docs/engineering/RELEASE-ACCEPTANCE.md",
    ]
    write_json(
        root / "dist/RELEASE-MANIFEST.json",
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": source_identity(),
            "verification": {
                "platform": "Windows / Python 3.12 / CPU",
                "unittest_passed": 32,
                "unittest_skipped": 0,
                "wheel_install_outside_source": True,
                "wheel_third_party_runtime_reused": True,
                "live_service_five_predictions_and_satellite_report": True,
                "remote_ci_executed": False,
                "satellite_matchup_accuracy_validated": False,
            },
            "artifacts": {
                name: {"sha256": sha256(root / name), "bytes": (root / name).stat().st_size} for name in paths
            },
        },
    )
    print("Frozen dist/RELEASE-MANIFEST.json")


if __name__ == "__main__":
    main()
