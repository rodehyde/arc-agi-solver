"""
download_re_arc.py — Download and extract the RE-ARC dataset.

RE-ARC provides 1,000 synthetically generated input/output pairs for each of
the 400 original ARC training tasks, in a flat list format per task file.

After running this script, data will be at:
    data/re_arc/<task_id>.json   (400 files, 1000 examples each)

Usage:
    python scripts/download_re_arc.py
    python scripts/download_re_arc.py --dest data/re_arc
"""

import argparse
import io
import json
import urllib.request
import zipfile
from pathlib import Path

RE_ARC_URL = "https://github.com/michaelhodel/re-arc/raw/main/re_arc.zip"
DEFAULT_DEST = Path(__file__).parent.parent / "data" / "re_arc"


def download_re_arc(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)

    print(f"Downloading RE-ARC from {RE_ARC_URL} ...")
    with urllib.request.urlopen(RE_ARC_URL) as response:
        data = response.read()
    print(f"Downloaded {len(data) / 1_000_000:.1f} MB")

    print("Extracting ...")
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        task_files = [n for n in zf.namelist()
                      if n.endswith(".json") and not Path(n).name.startswith("._")]
        print(f"Found {len(task_files)} task files in archive")
        for name in task_files:
            task_id = Path(name).stem
            content = zf.read(name)
            out_path = dest / f"{task_id}.json"
            out_path.write_bytes(content)

    print(f"Done. {len(task_files)} task files written to {dest}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download the RE-ARC dataset.")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST,
                        help="Directory to write task files into")
    args = parser.parse_args()
    download_re_arc(args.dest)
