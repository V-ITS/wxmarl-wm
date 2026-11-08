"""Re-render a saved replay (.npz) without the env or a policy."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from render.animate import animate, save_frame

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("replay", type=Path)
    ap.add_argument("--out", type=Path, default=None, help="output path (default: next to the replay)")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--sep-circles", action="store_true", help="draw 2.5 NM half-separation circles")
    ap.add_argument("--frame", type=int, default=None, help="write only this frame as a PNG")
    args = ap.parse_args()
    if args.frame is not None:
        print(save_frame(args.replay, args.frame, args.out or args.replay.with_suffix(".png"), args.sep_circles))
    else:
        print(animate(args.replay, args.out, args.fps, args.sep_circles))
