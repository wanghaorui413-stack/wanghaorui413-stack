"""下载官方公开工程样例，流式输出进度；不将下载成功当处理成功。"""

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

URLS = {
    "processor": "https://github.com/acolite/acolite/releases/download/20260421.0/acolite_py_win_20260421.0.tar.gz",
    "scene": "https://sentiwiki.copernicus.eu/__attachments/a_495476c83a9dc53136802c7afd98430a5fe34695c36e337c63b7b70522182ab5/processing-baseline-05.13-msi-level-1c-sample-dataset.zip",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("asset", choices=URLS)
    parser.add_argument("--directory", type=Path, default=Path("data/satellite/downloads"))
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    url = URLS[args.asset]
    output = args.directory / ("acolite_win_20260421.0.tar.gz" if args.asset == "processor" else "ESA_L1C_PB0513.zip")
    if output.exists():
        print("Cached:", output, output.stat().st_size, flush=True)
        return
    temporary = output.with_suffix(".partial")
    digest = hashlib.sha256()
    count = 0
    last = time.monotonic()
    request = urllib.request.Request(url, headers={"User-Agent": "Chlorophyll-Lab/0.2"})
    with urllib.request.urlopen(request, timeout=90) as response, temporary.open("wb") as stream:
        print("Downloading:", url, "length:", response.headers.get("Content-Length", "unknown"), flush=True)
        while block := response.read(1024 * 1024):
            count += len(block)
            if count > 4 * 1024**3:
                raise ValueError("示例大于4GiB，停止自动下载")
            stream.write(block)
            digest.update(block)
            if time.monotonic() - last > 10:
                print(f"{args.asset}: {count / 1024**2:.1f} MiB", flush=True)
                last = time.monotonic()
    temporary.replace(output)
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps({"url": url, "bytes": count, "sha256": digest.hexdigest()}, indent=2), encoding="utf-8"
    )
    print("Downloaded:", output, count, digest.hexdigest(), flush=True)


if __name__ == "__main__":
    main()
