from pathlib import Path
from chlorophyll.common import read_json, write_json, sha256

source = Path("data/satellite/gcp_l1c/S2B_MSIL1C_20240531T021529_N0510_R003_T52SCG_20240531T034003.SAFE")
corrected = Path("data/satellite/hanriver_20240531_acolite")
manifest = {str(p.relative_to(source)): sha256(p) for p in sorted(source.rglob("*")) if p.is_file()}
write_json(corrected / "input_manifest.json", manifest)
provenance = read_json(corrected / "correction_provenance.json")
provenance["input_manifest_sha256"] = sha256(corrected / "input_manifest.json")
provenance["input_manifest_verified_after_run"] = True
provenance["source_objects_manifest"] = str(source.parent / (source.name + ".sources.json"))
write_json(corrected / "correction_provenance.json", provenance)
print("Frozen source manifest:", len(manifest), "files")
