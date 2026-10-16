"""Install the exact archived Python after the unchanged resource admission."""

import hashlib
import json
import os
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URL = "https://github.com/astral-sh/python-build-standalone/releases/download/20260728/cpython-3.11.15%2B20260728-aarch64-apple-darwin-install_only.tar.gz"
SHA = "7dc10e31eede05a6ab1ec9e0b961f521078b0959f838ed1d7452597d529ff802"


def main():
    out = ROOT / "remote-bootstrap"
    out.mkdir(exist_ok=False)
    sys.path.insert(
        0, str(ROOT / "successor/itfm_v2/publication/itfm-v2-paper-20261011/tools")
    )
    import release

    admission = release.resource_admission()
    (out / "ADMISSION.json").write_text(json.dumps(admission, indent=2) + "\n")
    if admission["status"] != "PASS":
        raise RuntimeError("resource blocked before runtime download")
    archive = out / "python.tar.gz"
    with urllib.request.urlopen(URL, timeout=120) as source, archive.open("xb") as dest:
        while block := source.read(1024**2):
            dest.write(block)
    with archive.open("rb") as source:
        if hashlib.file_digest(source, "sha256").hexdigest() != SHA:
            raise RuntimeError("Python distribution checksum mismatch")
    runtime = ROOT / ".remote-runtime"
    runtime.mkdir(exist_ok=False)
    with tarfile.open(archive) as tar:
        tar.extractall(runtime, filter="data")
    executable = runtime / "python/bin/python3.11"
    version = subprocess.check_output(
        [str(executable), "-c", "import sys; print(sys.version_info[:3])"], text=True
    ).strip()
    if version != "(3, 11, 15)":
        raise RuntimeError("unexpected Python version")
    with Path(os.environ["GITHUB_PATH"]).open("a") as f:
        f.write(str(executable.parent) + "\n")
    (out / "RUNTIME.json").write_text(
        json.dumps(
            {
                "url": URL,
                "sha256": SHA,
                "version": version,
                "distribution": "Astral python-build-standalone; exact patch version, different build from local Homebrew",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
