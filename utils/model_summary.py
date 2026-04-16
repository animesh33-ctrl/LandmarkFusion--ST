
import torch
import torch.nn as nn
from typing import Optional


def count_parameters(model: nn.Module) -> dict:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable,
            "frozen": total - trainable}


def model_summary(model: nn.Module,
                  input_shape: Optional[tuple] = None,
                  device: str = "cpu"):
   
    print("\n" + "=" * 60)
    print(f"  Model: {model.__class__.__name__}")
    print("=" * 60)
    print(f"  {'Module':<35} {'Params':>12}")
    print("  " + "-" * 50)

    for name, module in model.named_children():
        n = sum(p.numel() for p in module.parameters())
        print(f"  {name:<35} {n:>12,}")

    stats = count_parameters(model)
    print("  " + "-" * 50)
    print(f"  {'Total':<35} {stats['total']:>12,}")
    print(f"  {'Trainable':<35} {stats['trainable']:>12,}")
    print(f"  {'Frozen':<35} {stats['frozen']:>12,}")
    print("=" * 60)

    if input_shape is not None:
        model.eval()
        model.to(device)
        dummy = torch.zeros(input_shape, device=device)
        with torch.no_grad():
            try:
                out = model(dummy)
                if isinstance(out, (tuple, list)):
                    shapes = [o.shape if hasattr(o, "shape") else type(o)
                              for o in out]
                    print(f"  Output shapes: {shapes}")
                else:
                    print(f"  Output shape : {out.shape}")
            except Exception as e:
                print(f"  [Forward pass failed: {e}]")
        print("=" * 60)
    print()
