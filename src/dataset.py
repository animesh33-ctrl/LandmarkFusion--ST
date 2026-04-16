
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional
from sklearn.model_selection import train_test_split
from PIL import Image
from torchvision import transforms

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.config import (
    SEQUENCE_LENGTH, TOTAL_DIM, SEED,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    NUM_WORKERS, IMG_SIZE,
    RECOG_BATCH, CNN_BATCH,
)


class LandmarkSeqDataset(Dataset):
    

    def __init__(self, sequences: np.ndarray, labels: List[int],
                 augment: bool = False):
        self.seqs    = sequences
        self.labels  = labels
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        seq   = self.seqs[idx]
        label = self.labels[idx]
        if self.augment:
            seq = self._augment(seq.copy())
        return torch.tensor(seq, dtype=torch.float32), label

    @staticmethod
    def _augment(seq: np.ndarray) -> np.ndarray:
        
        # Gaussian noise
        seq = seq + np.random.randn(*seq.shape).astype(np.float32) * 0.01
        # Temporal shift
        shift = np.random.randint(-5, 6)
        seq = np.roll(seq, shift, axis=0)
        # Scale jitter
        scale = np.random.uniform(0.9, 1.1)
        seq = seq * scale
        return seq



class CTCLandmarkDataset(Dataset):

    def __init__(self, sequences: np.ndarray,
                 gloss_seqs: List[List[int]],
                 augment: bool = False):
        assert len(sequences) == len(gloss_seqs)
        self.seqs      = sequences.astype(np.float32)
        self.gloss_seqs = gloss_seqs
        self.augment    = augment

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx: int):
        seq    = self.seqs[idx].copy()
        glosses = self.gloss_seqs[idx]
        if self.augment:
            seq = LandmarkSeqDataset._augment(seq)
        return (
            torch.tensor(seq, dtype=torch.float32),
            torch.tensor(glosses, dtype=torch.long),
        )


def ctc_collate_fn(batch):
    seqs, glosses = zip(*batch)
    seqs = torch.stack(seqs, dim=0)                     # (B, T, 279)
    input_lengths  = torch.full((len(seqs),), seqs.size(1), dtype=torch.long)
    target_lengths = torch.tensor([len(g) for g in glosses], dtype=torch.long)
    targets        = torch.cat([g for g in glosses])    # (sum_lens,)
    return seqs, targets, input_lengths, target_lengths



def _get_train_transforms():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1),
                                scale=(0.9, 1.1)),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                              [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])


def _get_val_transforms():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                              [0.229, 0.224, 0.225]),
    ])


class ISLImageDataset(Dataset):
    def __init__(self, paths: List[str], labels: List[int],
                 transform=None):
        self.paths  = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE))
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]



def _filter_min_samples(sequences, labels, min_s=1):
    from collections import Counter
    counts  = Counter(labels)
    keep    = {c for c, n in counts.items() if n >= min_s}
    mask    = [i for i, l in enumerate(labels) if l in keep]
    seqs_f  = sequences[mask]
    labs_f  = [labels[i] for i in mask]
    unique  = sorted(keep)
    remap   = {old: new for new, old in enumerate(unique)}
    labs_f  = [remap[l] for l in labs_f]
    return seqs_f, labs_f, remap


def make_seq_dataloaders(sequences: np.ndarray,
                         labels: List[int],
                         batch_size: int = RECOG_BATCH,
                         num_workers: int = NUM_WORKERS,
                         min_samples: int = 1,
                         pin_memory: bool = True,
                         persistent_workers: bool = False):
    """
    Build train/val/test DataLoaders from pre-extracted sequences.
    min_samples: minimum sequences per class to keep (default 1).
    """
    sequences, labels, remap = _filter_min_samples(
        sequences, labels, min_s=min_samples)

    n_classes = len(set(labels))
    n_total   = len(labels)

    # With very few samples per class, skip stratification and just split randomly
    use_stratify = all(
        sum(1 for l in labels if l == c) >= 3
        for c in set(labels)
    )

    try:
        X_tv, X_te, y_tv, y_te = train_test_split(
            sequences, labels,
            test_size=max(TEST_RATIO, n_classes / n_total + 0.01),
            random_state=SEED,
            stratify=labels if use_stratify else None,
        )
    except ValueError:
        # Fallback: non-stratified with minimum test size
        test_n = max(1, int(n_total * TEST_RATIO))
        X_tv, X_te = sequences[test_n:], sequences[:test_n]
        y_tv, y_te = labels[test_n:], labels[:test_n]

    # Ensure val set is not empty
    val_n = max(1, int(len(X_tv) * VAL_RATIO / (1 - TEST_RATIO)))
    if val_n >= len(X_tv):
        val_n = max(1, len(X_tv) // 5)

    use_stratify_val = (
        use_stratify and
        all(sum(1 for l in y_tv if l == c) >= 2 for c in set(y_tv))
    )

    try:
        X_tr, X_v, y_tr, y_v = train_test_split(
            X_tv, y_tv,
            test_size=val_n,
            random_state=SEED,
            stratify=y_tv if use_stratify_val else None,
        )
    except ValueError:
        X_tr, X_v = X_tv[val_n:], X_tv[:val_n]
        y_tr, y_v = y_tv[val_n:], y_tv[:val_n]

    tr_ds = LandmarkSeqDataset(X_tr, y_tr, augment=True)
    v_ds  = LandmarkSeqDataset(X_v,  y_v,  augment=False)
    te_ds = LandmarkSeqDataset(X_te, y_te, augment=False)

    # Cap batch size to dataset size to avoid empty batches
    tr_batch = min(batch_size, len(tr_ds))
    v_batch  = min(batch_size, len(v_ds))
    te_batch = min(batch_size, len(te_ds))

    kw = dict(
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(persistent_workers and num_workers > 0),
    )
    if num_workers > 0:
        kw["prefetch_factor"] = 2

    tr_dl = DataLoader(tr_ds, batch_size=tr_batch, shuffle=True,  drop_last=False, **kw)
    v_dl  = DataLoader(v_ds,  batch_size=v_batch,  shuffle=False, **kw)
    te_dl = DataLoader(te_ds, batch_size=te_batch, shuffle=False, **kw)

    print(f"[Dataset] Train {len(tr_ds)} | Val {len(v_ds)} | Test {len(te_ds)} "
          f"| Classes {n_classes}")
    return tr_dl, v_dl, te_dl, remap


def make_image_dataloaders(dataset_root: str,
                            batch_size: int = CNN_BATCH,
                            num_workers: int = NUM_WORKERS):
    valid = {".jpg", ".jpeg", ".png", ".bmp"}
    classes = sorted([d for d in os.listdir(dataset_root)
                      if os.path.isdir(os.path.join(dataset_root, d))])
    l2i = {c: i for i, c in enumerate(classes)}
    i2l = {i: c for c, i in l2i.items()}

    paths, labels = [], []
    for cls, idx in l2i.items():
        d = os.path.join(dataset_root, cls)
        for f in os.listdir(d):
            if os.path.splitext(f)[1].lower() in valid:
                paths.append(os.path.join(d, f))
                labels.append(idx)

    print(f"[ImageDataset] {len(paths)} images, {len(classes)} classes")

    p_tv, p_te, l_tv, l_te = train_test_split(
        paths, labels, test_size=TEST_RATIO,
        random_state=SEED, stratify=labels)
    rel_val = VAL_RATIO / (1 - TEST_RATIO)
    p_tr, p_v, l_tr, l_v = train_test_split(
        p_tv, l_tv, test_size=rel_val,
        random_state=SEED, stratify=l_tv)

    tr_ds = ISLImageDataset(p_tr, l_tr, _get_train_transforms())
    v_ds  = ISLImageDataset(p_v,  l_v,  _get_val_transforms())
    te_ds = ISLImageDataset(p_te, l_te, _get_val_transforms())

    kw = dict(num_workers=num_workers)
    tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,  **kw)
    v_dl  = DataLoader(v_ds,  batch_size=batch_size, shuffle=False, **kw)
    te_dl = DataLoader(te_ds, batch_size=batch_size, shuffle=False, **kw)

    return tr_dl, v_dl, te_dl, l2i, i2l