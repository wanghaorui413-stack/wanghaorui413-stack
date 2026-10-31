from datetime import datetime, timezone
from pathlib import Path
import urllib.request
import zipfile
from .common import ROOT, sha256, write_json, read_json

URL = "https://download.pangaea.de/dataset/948492/files/GLORIA-2022.zip"
DOI = "https://doi.org/10.1594/PANGAEA.948492"
CITATION = (
    "Lehmann, Moritz K et al. (2022): GLORIA - A global dataset of remote sensing "
    "reflectance and water quality from inland and coastal waters. PANGAEA. "
    "https://doi.org/10.1594/PANGAEA.948492"
)


def download():
    directory = ROOT / "data" / "gloria" / "raw"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "GLORIA-2022.zip"
    existing = path.exists()
    if not existing:
        request = urllib.request.Request(URL, headers={"User-Agent": "Chlorophyll-project/1.0"})
        temporary = path.with_suffix(".download")
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as stream:
            while block := response.read(1024 * 1024):
                stream.write(block)
        # 先验证下载文件，再将它升级为正式缓存。
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None:
                raise ValueError("下载 ZIP 校验失败")
        temporary.replace(path)
    with zipfile.ZipFile(path) as archive:
        required = {"GLORIA_meta_and_lab.csv", "GLORIA_Rrs.csv", "GLORIA_qc_flags.csv"}
        if not required.issubset({Path(n).name for n in archive.namelist()}):
            raise ValueError("GLORIA 文件结构不符合预期")
    manifest_path = directory / "provenance.json"
    old = read_json(manifest_path) if manifest_path.exists() else {}
    digest = sha256(path)
    if old.get("sha256") and old["sha256"] != digest:
        raise ValueError("原始GLORIA缓存与已记录版本不一致；请保留旧来源并显式建立新原始版本")
    manifest = dict(
        source="GLORIA-2022",
        doi=DOI,
        url=URL,
        sha256=digest,
        bytes=path.stat().st_size,
        license="CC-BY-4.0",
        citation=CITATION,
        checked_at_utc=datetime.now(timezone.utc).isoformat(),
        downloaded_at_utc=(old.get("downloaded_at_utc") if existing else datetime.now(timezone.utc).isoformat()),
        note="已有缓存的初次下载时间未知" if existing and not old else "",
    )
    write_json(manifest_path, manifest)
    print(f"GLORIA archive ready: {path}; sha256={digest}")
    return path
