"""
  python main.py --mode train_word
  python main.py --mode train_sentence
  python main.py --mode inference
  python main.py --mode all
"""

import os, sys, argparse, json

os.environ["GLOG_minloglevel"]    = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.config import (
    DEVICE, SEED, CHECKPOINT_DIR, CACHE_DIR,
    ISL_WORD_FRAMES, ISL_SENTENCE_FRAMES,
)


def _load_refiner_pairs(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = data.get("pairs", data)

    pairs = []
    for item in data:
        if isinstance(item, dict):
            src = item.get("src", item.get("source"))
            tgt = item.get("tgt", item.get("target"))
        else:
            src, tgt = item

        if src is None or tgt is None:
            raise ValueError(f"Invalid refiner pair entry in {path}: {item!r}")

        pairs.append((list(src), list(tgt)))

    return pairs


def check_paths():
    required = [ISL_WORD_FRAMES, ISL_SENTENCE_FRAMES]
    ok = True
    for p in required:
        if os.path.exists(p):
            print(f"  ✓  {p}")
        else:
            print(f"  ✗  MISSING: {p}")
            ok = False
    return ok


def main():
    parser = argparse.ArgumentParser(
        description="LandmarkFusion-ST: Multi-Stream Spatial-Temporal Sign Language Recognition"
    )
    parser.add_argument("--mode", type=str, default="check",
        choices=["check", "train_word", "train_sentence",
                 "train_refiner", "inference", "all"],
        help="Pipeline stage to run")
    parser.add_argument("--word_data",     type=str, default=ISL_WORD_FRAMES)
    parser.add_argument("--sentence_data", type=str, default=ISL_SENTENCE_FRAMES)
    parser.add_argument("--word_ckpt",     type=str, default=None)
    parser.add_argument("--sentence_ckpt", type=str, default=None)
    parser.add_argument("--refiner_vocab",  type=str, default=None)
    parser.add_argument("--refiner_train_pairs", type=str, default=None)
    parser.add_argument("--refiner_val_pairs",   type=str, default=None)
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("  LandmarkFusion-ST: Multi-Stream Spatial-Temporal Sign Language Recognition")
    print("=" * 80)
    print(f"  Python : {sys.version.split()[0]}")
    print(f"  Device : {DEVICE}")
    if DEVICE.type == "cuda":
        import torch
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  VRAM   : {vram:.1f} GB")
    print(f"  Seed   : {SEED}")
    print()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR,      exist_ok=True)

    if args.mode in ("check", "all"):
        print("[Step 0] Checking dataset paths...")
        if not check_paths():
            print("\n[ERROR] Missing dataset paths.")
            print("  Please update config/config.py with your dataset locations.")
            if args.mode == "check":
                return
        print()

    if args.mode in ("train_word", "all"):
        print("[Step 1] Training — Isolated Word Recognition\n")
        from training.train_isolated import train as train_word
        train_word(args.word_data, tag="word")
        print()

    if args.mode in ("train_sentence", "all"):
        print("[Step 2] Training — Sentence-Level Recognition\n")
        from training.train_continuous import train as train_sentence
        train_sentence(args.sentence_data, tag="isl")
        print()

    if args.mode in ("train_refiner", "all"):
        print("[Step 3] Training — Refiner\n")
        from src.models.semantic_refiner import GlossVocab
        from training.train_refiner import train_refiner

        if not (args.refiner_vocab and args.refiner_train_pairs and args.refiner_val_pairs):
            print("[ERROR] Refiner training needs --refiner_vocab, --refiner_train_pairs, and --refiner_val_pairs.")
            print("  Expected JSON input: a list of [src_ids, tgt_ids] pairs, or objects with src/tgt keys.")
            if args.mode == "train_refiner":
                return
        else:
            vocab = GlossVocab.load(args.refiner_vocab)
            train_pairs = _load_refiner_pairs(args.refiner_train_pairs)
            val_pairs = _load_refiner_pairs(args.refiner_val_pairs)
            train_refiner(vocab, train_pairs, val_pairs)
            print()

    if args.mode in ("inference", "all"):
        print("[Step 4] Launching Real-Time Inference\n")
        from inference.realtime import run
        run(args.word_ckpt, args.sentence_ckpt)

    print("\n[Done]")


if __name__ == "__main__":
    main()
