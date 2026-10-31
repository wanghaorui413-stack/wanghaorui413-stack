"""从公开GCP Sentinel-2镜像下载已列出的真实SAFE对象，逐文件校验MD5。"""

import base64
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import urllib.parse
import urllib.request

PREFIX = "tiles/52/S/CG/S2B_MSIL1C_20240322T021529_N0510_R003_T52SCG_20240322T040249.SAFE/"
ROOT = Path("data/satellite/gcp_l1c")


def get_objects():
    url = "https://storage.googleapis.com/storage/v1/b/gcp-public-data-sentinel-2/o?" + urllib.parse.urlencode(
        {"prefix": PREFIX, "maxResults": 1000}
    )
    return json.load(urllib.request.urlopen(url, timeout=60))["items"]


def fetch(item):
    relative = item["name"].split(".SAFE/", 1)[1]
    target = ROOT / PREFIX.split("/")[-2] / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        checksum = base64.b64encode(hashlib.md5(target.read_bytes()).digest()).decode()
        if target.stat().st_size != int(item["size"]) or checksum != item["md5Hash"]:
            raise ValueError(f"已有原始缓存被改变：{target}")
        return str(target)
    partial = target.with_suffix(target.suffix + ".partial")
    size = int(item["size"])
    offset = partial.stat().st_size if partial.exists() else 0
    for attempt in range(8):
        try:
            request = urllib.request.Request(
                item["mediaLink"], headers={"Range": f"bytes={offset}-", "User-Agent": "Chlorophyll-Lab/0.2"}
            )
            with urllib.request.urlopen(request, timeout=90) as response:
                if offset and response.status != 206:
                    raise ValueError("镜像不支持可靠续传")
                with partial.open("ab" if offset else "wb") as stream:
                    while block := response.read(1024 * 1024):
                        stream.write(block)
                        offset += len(block)
            if offset != size:
                raise ValueError(f"尺寸不匹配{offset}/{size}")
            checksum = base64.b64encode(hashlib.md5(partial.read_bytes()).digest()).decode()
            if checksum != item["md5Hash"]:
                raise ValueError("源MD5不匹配")
            partial.replace(target)
            print("Verified", relative, size, flush=True)
            return str(target)
        except Exception as error:
            offset = partial.stat().st_size if partial.exists() else 0
            print("Retry", relative, attempt, str(error), flush=True)
            if attempt == 7:
                raise
            time.sleep(2)


def main():
    global PREFIX
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default=PREFIX)
    args = parser.parse_args()
    if not args.prefix.startswith("tiles/52/S/CG/") or ".." in args.prefix:
        raise ValueError("示例限于已核实52SCG瓦片")
    PREFIX = args.prefix
    items = [
        o
        for o in get_objects()
        if not o["name"].endswith("/") and not o["name"].endswith("_TCI.jp2") and not o["name"].endswith("_$folder$")
    ]
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / (PREFIX.split("/")[-2] + ".sources.json")).write_text(json.dumps(items, indent=2), encoding="utf-8")
    print("Objects", len(items), "MiB", sum(int(o["size"]) for o in items) / 1024**2, flush=True)
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(fetch, items))
    print("SAFE ready:", ROOT / PREFIX.split("/")[-2], flush=True)


if __name__ == "__main__":
    main()
