import tarfile
from pathlib import Path

source = Path("data/satellite/downloads/acolite_win_20260421.0.tar.gz")
destination = Path("data/satellite/runtime").resolve()
destination.mkdir(parents=True, exist_ok=True)
with tarfile.open(source) as archive:
    for member in archive.getmembers():
        target = (destination / member.name).resolve()
        if not target.is_relative_to(destination):
            raise ValueError("Unsafe archive path")
    archive.extractall(destination, filter="data")
print("Executables:", list(destination.rglob("*.exe")))
