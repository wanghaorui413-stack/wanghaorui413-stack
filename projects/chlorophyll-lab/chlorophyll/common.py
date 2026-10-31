import hashlib
import json
import os
import tempfile
import shutil
from contextlib import contextmanager
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("CHLOROPHYLL_HOME", SOURCE_ROOT)).expanduser().resolve()
VERSION = "0.2.0"
ARCHITECTURE_VERSION = "spectral-v1"
PREPROCESS_VERSION = "rrs-zscore-log10-v1"
WAVELENGTHS = list(range(400, 751, 5))
FEATURES = [f"rrs_{w}" for w in WAVELENGTHS]
TARGET = "chlorophyll_ug_l"
SCHEMA_VERSION = 1


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def atomic_directory(destination):
    """失败不占用最终名称；成功时仅提交完整目录。"""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"产物已存在：{destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="." + destination.name + ".staging-", dir=destination.parent))
    try:
        yield temporary
        # Windows rename不会替换已存在目录；保留并发下的排他提交语义。
        if destination.exists():
            raise FileExistsError(f"产物已存在：{destination}")
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def source_identity():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return {
        "version": VERSION,
        "source_sha256": digest.hexdigest(),
        "architecture_version": ARCHITECTURE_VERSION,
        "preprocess_version": PREPROCESS_VERSION,
    }


def named_path(parent, name):
    if not name or name in {".", ".."} or Path(name).name != name or "/" in name or "\\" in name or ":" in name:
        raise ValueError("名称必须是一个简单目录名")
    return ROOT / parent / name
