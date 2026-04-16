

import os
import numpy as np


def plot_training_curves(history: dict, save_path: str = None,
                          title: str = "Training Curves"):
    """
    history : dict with keys 'train_loss', 'val_loss',
                               'train_acc',  'val_acc'
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Viz] matplotlib not installed — skipping plots.")
        return

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title, fontsize=14)

    # Loss
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], label="Train Loss", linewidth=2)
    ax.plot(epochs, history["val_loss"],   label="Val Loss",   linewidth=2,
            linestyle="--")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Loss Curve"); ax.legend(); ax.grid(True, alpha=0.3)

    # Accuracy
    ax = axes[1]
    ax.plot(epochs, history["train_acc"], label="Train Acc", linewidth=2)
    ax.plot(epochs, history["val_acc"],   label="Val Acc",   linewidth=2,
            linestyle="--")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy Curve"); ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [Viz] Saved → {save_path}")
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names=None,
                           save_path: str = None,
                           title: str = "Confusion Matrix"):
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix
    except ImportError:
        print("[Viz] matplotlib/seaborn not installed — skipping CM plot.")
        return

    cm = confusion_matrix(y_true, y_pred)
    # Normalise to percentage
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig_size = max(8, len(cm) * 0.4)
    fig, ax  = plt.subplots(figsize=(fig_size, fig_size * 0.85))
    sns.heatmap(cm_norm, annot=(len(cm) <= 40),
                fmt=".2f", cmap="Blues",
                xticklabels=class_names or "auto",
                yticklabels=class_names or "auto",
                ax=ax, linewidths=0.3)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [Viz] Saved → {save_path}")
    plt.show()


def plot_landmark_frame(flat279: np.ndarray, title: str = "Landmarks"):
    """
    Quick visualisation of one 279-D landmark vector as a 2-D scatter.
    flat279 : (279,)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    hand = flat279[:126].reshape(42, 3)
    face = flat279[126:246].reshape(40, 3)
    body = flat279[246:279].reshape(11, 3)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(hand[:, 0], -hand[:, 1], s=20, c="royalblue",  label="Hand",
               zorder=3)
    ax.scatter(face[:, 0], -face[:, 1], s=10, c="darkorange", label="Face",
               zorder=3)
    ax.scatter(body[:, 0], -body[:, 1], s=30, c="green",      label="Body",
               marker="^", zorder=3)
    ax.set_title(title); ax.legend(); ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
