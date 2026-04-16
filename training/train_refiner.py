
import os, sys, time
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.config import (
    DEVICE, SEED, CHECKPOINT_DIR,
    REFINE_LR, REFINE_BATCH, REFINE_EPOCHS,
)
from src.models.semantic_refiner import SemanticRefiner, GlossVocab
from utils.metrics import AverageMeter, EarlyStopping, save_checkpoint

torch.manual_seed(SEED)


class GlossDataset(torch.utils.data.Dataset):
    """
    Pairs of (noisy_gloss_ids, target_gloss_ids) for seq2seq correction.
    In practice: CTC outputs paired with ground-truth gloss sequences.
    """
    def __init__(self, pairs, max_len=64):
        self.pairs   = pairs
        self.max_len = max_len

    def __len__(self): return len(self.pairs)

    def __getitem__(self, i):
        src, tgt = self.pairs[i]
        # Pad to max_len
        src_p = (src + [0] * self.max_len)[:self.max_len]
        tgt_p = (tgt + [0] * self.max_len)[:self.max_len]
        mask  = [1 if x != 0 else 0 for x in src_p]
        return (torch.tensor(src_p, dtype=torch.long),
                torch.tensor(tgt_p, dtype=torch.long),
                torch.tensor(mask,  dtype=torch.long))


def train_refiner(vocab: GlossVocab,
                  train_pairs,
                  val_pairs,
                  hidden_dim: int = 256):
    print("=" * 65)
    print("  Training — Semantic Refiner (Stage 4)")
    print("=" * 65)

    model = SemanticRefiner(
        vocab_size=len(vocab),
        hidden_dim=hidden_dim,
        freeze_encoder=True,
    ).to(DEVICE)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable:,} / {total:,}")

    tr_ds = GlossDataset(train_pairs)
    v_ds  = GlossDataset(val_pairs)
    tr_dl = torch.utils.data.DataLoader(
        tr_ds, batch_size=REFINE_BATCH, shuffle=True,  num_workers=2)
    v_dl  = torch.utils.data.DataLoader(
        v_ds,  batch_size=REFINE_BATCH, shuffle=False, num_workers=2)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=REFINE_LR)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    stopper   = EarlyStopping(patience=5, mode="min")

    best_ckpt = os.path.join(CHECKPOINT_DIR, "best_refiner.pth")
    best_loss = float("inf")

    for epoch in range(1, REFINE_EPOCHS + 1):
        # Train
        model.train()
        tr_meter = AverageMeter("tr_loss")
        for src, tgt, mask in tqdm(tr_dl, desc="  Train", leave=False):
            src, tgt, mask = src.to(DEVICE), tgt.to(DEVICE), mask.to(DEVICE)
            optimizer.zero_grad()
            logits = model(src, mask)           # (B, L, vocab)
            loss   = criterion(logits.reshape(-1, len(vocab)), tgt.reshape(-1))
            loss.backward()
            optimizer.step()
            tr_meter.update(loss.item(), src.size(0))

        # Validate
        model.eval()
        v_meter = AverageMeter("vl_loss")
        with torch.no_grad():
            for src, tgt, mask in v_dl:
                src, tgt, mask = src.to(DEVICE), tgt.to(DEVICE), mask.to(DEVICE)
                logits = model(src, mask)
                loss   = criterion(logits.reshape(-1, len(vocab)), tgt.reshape(-1))
                v_meter.update(loss.item(), src.size(0))

        print(f"  Epoch {epoch:>3}/{REFINE_EPOCHS} | "
              f"Train {tr_meter.avg:.4f} | Val {v_meter.avg:.4f}")

        if v_meter.avg < best_loss:
            best_loss = v_meter.avg
            save_checkpoint(model, optimizer, epoch, best_loss, best_ckpt)

        if stopper(v_meter.avg):
            print("  [EarlyStopping] Triggered")
            break

    print(f"  Best refiner checkpoint → {best_ckpt}")
    return model
