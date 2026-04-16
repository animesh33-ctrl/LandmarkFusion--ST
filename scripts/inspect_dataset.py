"""
scripts/inspect_dataset.py
===========================
Inspects the ISL-CSLTR Kaggle dataset and prints a summary.
Run this FIRST after downloading to verify the dataset structure
before training.

Usage:
    python scripts/inspect_dataset.py --root "C:/path/to/ISL_CSLRT_Corpus"
"""

import os, sys, argparse
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def inspect(root: str):
    print(f"\n{'='*65}")
    print(f"  ISL-CSLTR Dataset Inspector")
    print(f"  Root: {root}")
    print(f"{'='*65}")

    if not os.path.exists(root):
        print(f"  [ERROR] Path does not exist: {root}")
        return

    # List top-level dirs
    top = sorted(os.listdir(root))
    print(f"\n  Top-level entries: {top}\n")

    valid_img = {".jpg", ".jpeg", ".png", ".bmp"}
    valid_vid = {".mp4", ".avi", ".mov", ".mkv"}

    for sub in top:
        sub_path = os.path.join(root, sub)
        if not os.path.isdir(sub_path):
            continue

        classes   = [d for d in os.listdir(sub_path)
                     if os.path.isdir(os.path.join(sub_path, d))]
        n_classes = len(classes)

        # Count files
        total_samples = 0
        sample_counts = defaultdict(int)
        frame_counts  = defaultdict(list)

        for cls in classes:
            cls_path = os.path.join(sub_path, cls)
            subs     = [s for s in os.listdir(cls_path)
                        if os.path.isdir(os.path.join(cls_path, s))]
            if subs:
                # Folder-per-sample structure
                for s in subs:
                    sp    = os.path.join(cls_path, s)
                    files = [f for f in os.listdir(sp)
                             if os.path.splitext(f)[1].lower() in valid_img]
                    frame_counts[cls].append(len(files))
                    total_samples += 1
                    sample_counts[cls] += 1
            else:
                # Flat image folder
                files = [f for f in os.listdir(cls_path)
                         if os.path.splitext(f)[1].lower() in
                         (valid_img | valid_vid)]
                if files:
                    total_samples += len(files)
                    sample_counts[cls] = len(files)

        counts  = list(sample_counts.values())
        avg_s   = sum(counts) / max(len(counts), 1)
        min_s   = min(counts) if counts else 0
        max_s   = max(counts) if counts else 0

        print(f"  [{sub}]")
        print(f"    Classes   : {n_classes}")
        print(f"    Total     : {total_samples}")
        print(f"    Per-class : min={min_s}  avg={avg_s:.1f}  max={max_s}")

        if frame_counts:
            all_fc = [f for fc in frame_counts.values() for f in fc]
            if all_fc:
                print(f"    Frames/seq: min={min(all_fc)}"
                      f"  avg={sum(all_fc)/len(all_fc):.1f}"
                      f"  max={max(all_fc)}")

        # Show first 5 class names
        preview = classes[:5]
        more    = f"  … +{n_classes-5} more" if n_classes > 5 else ""
        print(f"    Classes preview: {preview}{more}")
        print()

    print("  Tip: Update DATASET_ROOT in config/config.py to this path.")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, required=True,
                   help="Root path of ISL_CSLRT_Corpus")
    args = p.parse_args()
    inspect(args.root)
