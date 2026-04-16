
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, classification_report,
)


class AverageMeter:
    def __init__(self, name="metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.val = self.sum = self.avg = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count

    def __repr__(self):
        return f"{self.name}={self.avg:.4f}"


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0, mode="min"):
        self.patience  = patience
        self.min_delta = min_delta
        self.mode      = mode
        self.counter   = 0
        self.best      = None

    def __call__(self, metric: float) -> bool:
        score = -metric if self.mode == "min" else metric
        if self.best is None:
            self.best = score
            return False
        if score < self.best + self.min_delta:
            self.counter += 1
            return self.counter >= self.patience
        else:
            self.best    = score
            self.counter = 0
            return False


def save_checkpoint(model, optimizer, epoch, loss, path):
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch":               epoch,
        "model_state_dict":    model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss":                loss,
    }, path)
    print(f"  [Ckpt] Saved → {path}")


def load_checkpoint(path, model, optimizer=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    print(f"  [Ckpt] Loaded ← {path}  (epoch {ckpt['epoch']})")
    return ckpt["epoch"], ckpt["loss"]


def compute_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def compute_f1(y_true, y_pred, avg="weighted"):
    return f1_score(y_true, y_pred, average=avg, zero_division=0)

def full_report(y_true, y_pred, class_names=None):
    return classification_report(y_true, y_pred,
                                  target_names=class_names,
                                  zero_division=0)
