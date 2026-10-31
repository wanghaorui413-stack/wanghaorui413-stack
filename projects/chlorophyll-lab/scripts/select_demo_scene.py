import concurrent.futures
import json
import urllib.request
import urllib.parse
import xml.etree.ElementTree as E
from pathlib import Path


def metadata(prefix):
    try:
        url = "https://storage.googleapis.com/gcp-public-data-sentinel-2/" + prefix + "MTD_MSIL1C.xml"
        payload = urllib.request.urlopen(url, timeout=60).read()
        root = E.fromstring(payload)
        cloud = float(next(e.text for e in root.iter() if e.tag.endswith("Cloud_Coverage_Assessment")))
        return {"prefix": prefix, "cloud_percent": cloud, "metadata_url": url}
    except Exception as error:
        return {"prefix": prefix, "error": str(error), "cloud_percent": 101}


def main():
    prefixes = []
    for sensor in ["S2A", "S2B"]:
        query = urllib.parse.urlencode(
            {"prefix": f"tiles/52/S/CG/{sensor}_MSIL1C_202405", "delimiter": "/", "maxResults": 100}
        )
        listing = json.load(
            urllib.request.urlopen(
                "https://storage.googleapis.com/storage/v1/b/gcp-public-data-sentinel-2/o?" + query, timeout=60
            )
        )
        prefixes.extend(listing.get("prefixes", []))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(metadata, prefixes))
    rows.sort(key=lambda r: r["cloud_percent"])
    path = Path("data/satellite/candidate_scenes.json")
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
