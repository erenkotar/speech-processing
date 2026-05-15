"""
download_dataset.py
-------------------
Fetch the Free Spoken Digit Dataset (FSDD) into ``data/recordings/``.

FSDD is a small public dataset of spoken digit recordings (0-9) at 8 kHz,
~3000 wav files across 6 speakers. Source:
    https://github.com/Jakobovski/free-spoken-digit-dataset

Run standalone:
    python download_dataset.py
or import and call ``ensure_dataset()`` from another module.
"""

from __future__ import annotations

import io
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

FSDD_URL = (
    "https://github.com/Jakobovski/free-spoken-digit-dataset/"
    "archive/refs/heads/master.zip"
)

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
RECORDINGS_DIR = DATA_DIR / "recordings"


def _already_populated(min_files: int = 100) -> bool:
    if not RECORDINGS_DIR.is_dir():
        return False
    return sum(1 for _ in RECORDINGS_DIR.glob("*.wav")) >= min_files


def ensure_dataset() -> Path:
    """Download + extract FSDD if not already present. Returns recordings path."""
    if _already_populated():
        print(f"[data]  Dataset already present at {RECORDINGS_DIR}")
        return RECORDINGS_DIR

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[data]  Downloading FSDD from {FSDD_URL}")
    with urllib.request.urlopen(FSDD_URL) as resp:
        zip_bytes = resp.read()
    print(f"[data]  Got {len(zip_bytes)/1e6:.1f} MB, extracting recordings/")

    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    extracted = 0
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for info in zf.infolist():
            # Only pull files inside the dataset's "recordings/" folder.
            parts = info.filename.split("/")
            if len(parts) >= 3 and parts[1] == "recordings" and parts[-1].endswith(".wav"):
                target = RECORDINGS_DIR / parts[-1]
                with zf.open(info) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                extracted += 1

    print(f"[data]  Extracted {extracted} wav files into {RECORDINGS_DIR}")
    if extracted == 0:
        raise RuntimeError("FSDD download yielded no wav files — unexpected layout?")
    return RECORDINGS_DIR


if __name__ == "__main__":
    try:
        ensure_dataset()
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)
