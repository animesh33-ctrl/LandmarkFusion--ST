
import os, sys, time, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.config import (
    DEVICE, SEED, CHECKPOINT_DIR, CACHE_DIR,
    ISL_WORD_FRAMES, SEQUENCE_LENGTH,
    RECOG_LR, RECOG_WD, RECOG_BATCH, RECOG_EPOCHS,
    RECOG_PATIENCE, RECOG_GRAD_CLIP, RECOG_LABEL_SMOOTH,
    RECOG_BETA1, RECOG_BETA2, NUM_WORKERS,
)
from src.landmark_extractor import bulk_extract_images
from src.dataset import make_seq_dataloaders
from src.models.landmark_fusion_st import LandmarkFusionST, LandmarkFusionLoss
from utils.metrics import AverageMeter, EarlyStopping, save_checkpoint

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def _is_oom_error(err: RuntimeError) -> bool:
    msg = str(err).lower()
    return "out of memory" in msg or "cuda error: out of memory" in msg


def train_one_epoch(model, loader, criterion, optimizer, device, scaler, use_amp):
    model.train()
    meter = AverageMeter("loss")
    correct = total = 0

    for seqs, labels in tqdm(loader, desc="  Train", leave=False):
        seqs   = seqs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logits, feat = model(seqs)
            loss = criterion(logits, feat, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), RECOG_GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), RECOG_GRAD_CLIP)
            optimizer.step()

        meter.update(loss.item(), seqs.size(0))
        correct += (logits.argmax(1) == labels).sum().item()
        total   += labels.size(0)

        del logits, feat, loss

    return meter.avg, correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp):
    model.eval()
    meter = AverageMeter("loss")
    correct = total = 0

    for seqs, labels in tqdm(loader, desc="  Val  ", leave=False):
        seqs   = seqs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logits, feat = model(seqs)
            loss = criterion(logits, feat, labels)
        meter.update(loss.item(), seqs.size(0))
        correct += (logits.argmax(1) == labels).sum().item()
        total   += labels.size(0)

        del logits, feat, loss

    return meter.avg, correct / max(total, 1)


def train(data_root: str = ISL_WORD_FRAMES, tag: str = "word"):
    print("=" * 65)
    print("  LandmarkFusion-ST — Isolated Sign Recognition Training")
    print("=" * 65)
    print(f"  Device : {DEVICE}")
    print(f"  torch.cuda.is_available(): {torch.cuda.is_available()}")
    if DEVICE.type == "cuda":
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
    else:
        raise RuntimeError(
            "CUDA not active in this Python environment. "
            "Install CUDA-enabled PyTorch in your venv."
        )

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, f"{tag}_seq_cache.npz")

    sequences, labels, label_map = bulk_extract_images(
        data_root, seq_len=SEQUENCE_LENGTH,
        cache_path=cache, augment_copies=2)

    print(f"  Classes: {len(label_map)} | Sequences: {len(sequences)}")

    batch_size = RECOG_BATCH
    loader_workers = min(2, NUM_WORKERS)
    tr_dl, v_dl, te_dl, remap = make_seq_dataloaders(
        sequences,
        labels,
        batch_size=batch_size,
        num_workers=loader_workers,
        pin_memory=(DEVICE.type == "cuda"),
        persistent_workers=False,
    )

    num_classes = len(remap)
    print(f"  Classes (after filter): {num_classes}")

    model = LandmarkFusionST(
        num_classes=num_classes,
        dropout=0.1,
        use_ctc=False,
    ).to(DEVICE)

    total_p = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_p:,}")

    criterion = LandmarkFusionLoss(
        lam=0.3, label_smoothing=RECOG_LABEL_SMOOTH, use_ctc=False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=RECOG_LR, weight_decay=RECOG_WD,
        betas=(RECOG_BETA1, RECOG_BETA2))
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    stopper   = EarlyStopping(patience=RECOG_PATIENCE, mode="min")

    best_ckpt = os.path.join(CHECKPOINT_DIR, f"best_lmfst_{tag}.pth")
    last_ckpt = os.path.join(CHECKPOINT_DIR, f"last_lmfst_{tag}.pth")
    best_loss = float("inf")
    use_amp = DEVICE.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    if use_amp:
        print("  AMP: enabled (float16 autocast + GradScaler)")

    for epoch in range(1, RECOG_EPOCHS + 1):
        t0 = time.time()
        while True:
            try:
                tr_loss, tr_acc = train_one_epoch(
                    model, tr_dl, criterion, optimizer, DEVICE, scaler, use_amp
                )
                break
            except RuntimeError as err:
                if DEVICE.type == "cuda" and _is_oom_error(err):
                    if batch_size <= 1:
                        raise RuntimeError(
                            "CUDA OOM even at batch_size=1. Reduce SEQUENCE_LENGTH or model size."
                        ) from err
                    batch_size = max(1, batch_size // 2)
                    print(f"  [OOM] Reducing batch size to {batch_size} and retrying epoch {epoch}...")
                    torch.cuda.empty_cache()
                    tr_dl, v_dl, te_dl, remap = make_seq_dataloaders(
                        sequences,
                        labels,
                        batch_size=batch_size,
                        num_workers=loader_workers,
                        pin_memory=(DEVICE.type == "cuda"),
                        persistent_workers=False,
                    )
                    continue
                raise

        vl_loss, vl_acc = evaluate(model, v_dl, criterion, DEVICE, use_amp)
        scheduler.step()
        elapsed = time.time() - t0

        print(f"  Epoch {epoch:>3}/{RECOG_EPOCHS} | "
              f"Loss {tr_loss:.4f}/{vl_loss:.4f} | "
              f"Acc {tr_acc:.4f}/{vl_acc:.4f} | {elapsed:.1f}s")

        if vl_loss < best_loss:
            best_loss = vl_loss
            save_checkpoint(model, optimizer, epoch, vl_loss, best_ckpt)
        save_checkpoint(model, optimizer, epoch, vl_loss, last_ckpt)
        if stopper(vl_loss):
            print(f"  [EarlyStopping] Triggered at epoch {epoch}")
            break

    # ── Test evaluation ───────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  TEST EVALUATION")
    print("=" * 65)
    ckpt = torch.load(best_ckpt, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    te_loss, te_acc = evaluate(model, te_dl, criterion, DEVICE, use_amp)
    print(f"  Test Loss: {te_loss:.4f} | Test Acc: {te_acc:.4f}")

    from sklearn.metrics import classification_report, f1_score
    import numpy as np
    model.eval()
    all_p, all_l = [], []
    with torch.no_grad():
        for seqs, labels in te_dl:
            logits, _ = model(seqs.to(DEVICE))
            all_p.extend(logits.argmax(1).cpu().numpy())
            all_l.extend(labels.numpy())

    all_p = np.array(all_p); all_l = np.array(all_l)
    inv_remap = {v: k for k, v in remap.items()}
    inv_label = {v: k for k, v in label_map.items()}

    # Only report on classes that actually appear in the test set
    present_classes = sorted(set(all_l.tolist()))
    cls_names = [inv_label.get(inv_remap.get(i, -1), str(i))
                 for i in present_classes]
    print(classification_report(all_l, all_p,
                                 labels=present_classes,
                                 target_names=cls_names,
                                 zero_division=0))

    f1 = f1_score(all_l, all_p, average="weighted", zero_division=0)
    print(f"  Weighted F1: {f1:.4f}")
    print(f"  Best checkpoint → {best_ckpt}")
    return model


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default=ISL_WORD_FRAMES)
    p.add_argument("--tag",  type=str, default="word")
    args = p.parse_args()
    train(args.data, args.tag)