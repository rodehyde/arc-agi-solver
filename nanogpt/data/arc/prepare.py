"""
prepare.py — nanoGPT data bridge for ARC fine-tuning.

Copies train.bin and val.bin from the arc-agi-solver finetune outputs
into this directory so nanoGPT's training loop can find them.

Run from the nanoGPT root:
    python data/arc/prepare.py
"""

import shutil
import subprocess
import sys
from pathlib import Path

ARC_SOLVER_REPO = Path('/content/arc-agi-solver')
SRC_DIR = ARC_SOLVER_REPO / 'data' / 'finetune'
DST_DIR = Path(__file__).parent


def main():
    # Generate train.bin / val.bin if not already present
    for name in ['train.bin', 'val.bin']:
        if not (SRC_DIR / name).exists():
            print(f'{name} not found — running prepare_arc_finetune.py ...')
            result = subprocess.run(
                [sys.executable,
                 str(ARC_SOLVER_REPO / 'scripts' / 'prepare_arc_finetune.py')],
                check=True,
            )
            break   # one run generates both files

    # Copy to nanoGPT's data/arc/ directory
    for name in ['train.bin', 'val.bin']:
        src = SRC_DIR / name
        dst = DST_DIR / name
        if not src.exists():
            raise FileNotFoundError(
                f'{src} still missing after running prepare_arc_finetune.py'
            )
        shutil.copy(src, dst)
        print(f'  Copied {name}  ({src.stat().st_size / 1024:.0f} KB)')

    print('Done — train.bin and val.bin ready for nanoGPT.')


if __name__ == '__main__':
    main()
