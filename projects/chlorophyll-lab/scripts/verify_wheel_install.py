"""离开源码目录，将wheel安装到全新项目目录并验证入口及包资源。"""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile


def main():
    wheel = next(Path("dist").glob("chlorophyll_lab-*.whl")).resolve()
    with zipfile.ZipFile(wheel) as archive:
        assert "chlorophyll/assets/chlorophyll.html" in archive.namelist()
    with tempfile.TemporaryDirectory(prefix="chlorophyll-wheel-test-") as directory:
        destination = Path(directory) / "installed"
        bundled = Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe"
        subprocess.run(
            [
                str(bundled),
                "-m",
                "pip",
                "--python",
                sys.executable,
                "install",
                "--no-index",
                "--no-deps",
                "--no-cache-dir",
                "--target",
                str(destination),
                str(wheel),
            ],
            check=True,
        )
        environment = dict(os.environ, PYTHONPATH=str(destination), CHLOROPHYLL_HOME=str(Path(directory) / "work"))
        code = (
            "from pathlib import Path; import chlorophyll; from chlorophyll.common import ROOT; "
            "from importlib.resources import files; "
            f"assert Path(chlorophyll.__file__).resolve().is_relative_to(Path({str(destination)!r})); "
            'assert files("chlorophyll").joinpath("assets/chlorophyll.html").is_file(); '
            'print("Installed outside source:",chlorophyll.__file__); print("Workspace:",ROOT)'
        )
        subprocess.run([sys.executable, "-c", code], cwd=directory, env=environment, check=True)
        subprocess.run([sys.executable, "-m", "chlorophyll", "--version"], cwd=directory, env=environment, check=True)
    print("Clean project wheel installation: PASS (third-party runtime reused)")


if __name__ == "__main__":
    main()
