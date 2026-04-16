"""
scripts/generate_refiner_data.py
==================================
Auto-generates the 3 JSON files needed for train_refiner from
the existing word/sentence cache files.

Usage:
    python scripts/generate_refiner_data.py

Outputs (in checkpoints/ dir):
    refiner_vocab.json          -- GlossVocab token list
    refiner_train_pairs.json    -- list of {src, tgt} pairs (train 80%)
    refiner_val_pairs.json      -- list of {src, tgt} pairs (val 20%)

Then run:
    python main.py --mode train_refiner ^
        --refiner_vocab checkpoints/refiner_vocab.json ^
        --refiner_train_pairs checkpoints/refiner_train_pairs.json ^
        --refiner_val_pairs checkpoints/refiner_val_pairs.json
"""

import os, sys, json, random
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.config import CACHE_DIR, CHECKPOINT_DIR

SEED       = 42
VAL_RATIO  = 0.20
MAX_SEQ_LEN = 32   # max gloss tokens per pair (pad/truncate)

random.seed(SEED)
np.random.seed(SEED)


def load_label_map(cache_path: str) -> dict:
    """Returns {idx: name} direct mapping."""
    if not os.path.exists(cache_path):
        return {}
    data = np.load(cache_path, allow_pickle=True)
    lm   = data["label_map"].item()   # {name: idx}
    return {v: k for k, v in lm.items()}


def build_pairs_from_cache(cache_path: str,
                            idx2label: dict,
                            vocab_set: set,
                            noise_prob: float = 0.15):
    """
    Builds (src, tgt) token-id pairs from a cache file.

    Strategy: treat each sequence label as a single-token gloss.
    src = noisy version (random token swap with noise_prob)
    tgt = ground truth

    For a real system you'd use CTC-decoded outputs as src.
    This gives the refiner enough data to learn correction.
    """
    if not os.path.exists(cache_path):
        return []

    data   = np.load(cache_path, allow_pickle=True)
    labels = data["labels"].tolist()

    pairs = []
    for lbl_idx in labels:
        name = idx2label.get(lbl_idx)
        if name is None:
            continue
        vocab_set.add(name)
        pairs.append(name)

    return pairs


def add_noise(token_ids: list, vocab_size: int,
              noise_prob: float = 0.15) -> list:
    """Randomly substitute tokens to simulate CTC errors."""
    noisy = []
    for tid in token_ids:
        if random.random() < noise_prob:
            noisy.append(random.randint(3, vocab_size - 1))  # skip special tokens
        else:
            noisy.append(tid)
    return noisy


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    word_cache = os.path.join(CACHE_DIR, "word_seq_cache.npz")
    isl_cache  = os.path.join(CACHE_DIR, "isl_seq_cache.npz")

    word_idx2label = load_label_map(word_cache)
    isl_idx2label  = load_label_map(isl_cache)

    vocab_set = set()

    # Collect all gloss names
    word_glosses = build_pairs_from_cache(word_cache, word_idx2label, vocab_set)
    isl_glosses  = build_pairs_from_cache(isl_cache,  isl_idx2label,  vocab_set)

    all_glosses = word_glosses + isl_glosses
    print(f"[Refiner Data] Total samples: {len(all_glosses)}")
    print(f"[Refiner Data] Unique glosses: {len(vocab_set)}")

    if not all_glosses:
        print("[ERROR] No cache data found. Run training first.")
        return

    # Build vocab
    specials  = ["<blank>", "<pad>", "<unk>"]
    vocab_list = specials + sorted(vocab_set)
    g2idx      = {g: i for i, g in enumerate(vocab_list)}

    vocab_path = os.path.join(CHECKPOINT_DIR, "refiner_vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_list, f, indent=2)
    print(f"[Refiner Data] Vocab ({len(vocab_list)} tokens) → {vocab_path}")

    # Build pairs: group consecutive glosses into short sequences
    # (simulate sentence-level CTC input/output)
    GROUP_SIZE = 4   # ~4 glosses per sequence pair
    pairs = []
    for i in range(0, len(all_glosses) - GROUP_SIZE + 1, GROUP_SIZE // 2):
        group = all_glosses[i: i + GROUP_SIZE]
        tgt_ids = [g2idx.get(g, g2idx["<unk>"]) for g in group]
        src_ids = add_noise(tgt_ids, len(vocab_list), noise_prob=0.20)
        # Pad/truncate to MAX_SEQ_LEN
        src_ids = (src_ids + [g2idx["<pad>"]] * MAX_SEQ_LEN)[:MAX_SEQ_LEN]
        tgt_ids = (tgt_ids + [g2idx["<pad>"]] * MAX_SEQ_LEN)[:MAX_SEQ_LEN]
        pairs.append({"src": src_ids, "tgt": tgt_ids})

    print(f"[Refiner Data] Total pairs: {len(pairs)}")

    # Shuffle + split
    random.shuffle(pairs)
    split = int(len(pairs) * (1 - VAL_RATIO))
    train_pairs = pairs[:split]
    val_pairs   = pairs[split:]

    train_path = os.path.join(CHECKPOINT_DIR, "refiner_train_pairs.json")
    val_path   = os.path.join(CHECKPOINT_DIR, "refiner_val_pairs.json")

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_pairs, f)
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_pairs, f)

    print(f"[Refiner Data] Train pairs ({len(train_pairs)}) → {train_path}")
    print(f"[Refiner Data] Val   pairs ({len(val_pairs)})  → {val_path}")
    print()
    print("=" * 60)
    print("  Now run refiner training:")
    print()
    print("  python main.py --mode train_refiner ^")
    print(f"    --refiner_vocab {vocab_path} ^")
    print(f"    --refiner_train_pairs {train_path} ^")
    print(f"    --refiner_val_pairs {val_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()