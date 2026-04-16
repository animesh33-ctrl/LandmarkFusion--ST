

import os
import sys


def check_path(path: str, label: str = "") -> bool:
    exists = os.path.exists(path)
    tag    = label or path
    sym    = "✓" if exists else "✗"
    color  = "\033[92m" if exists else "\033[91m"
    reset  = "\033[0m"
    print(f"  {color}{sym}{reset}  {tag}")
    return exists


def validate_dataset_paths(paths: list, exit_on_fail: bool = True) -> bool:
    print("\n[Path Check]")
    all_ok = all(check_path(p) for p in paths)
    if not all_ok:
        print("\n  [ERROR] One or more required paths are missing.")
        print("  → Update DATASET_ROOT in config/config.py")
        if exit_on_fail:
            sys.exit(1)
    else:
        print("  All paths OK.\n")
    return all_ok


def ensure_dirs(*dirs):
    """Create directories if they don't exist."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)
