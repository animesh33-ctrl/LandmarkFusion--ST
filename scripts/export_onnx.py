"""
scripts/export_onnx.py
=======================
Exports a trained LandmarkFusion-ST checkpoint to ONNX format
for deployment (e.g. edge devices, TensorRT, ONNX Runtime).

Usage:
    python scripts/export_onnx.py \
        --ckpt checkpoints/best_lmfst_word.pth \
        --num_classes 50 \
        --output checkpoints/lmfst_word.onnx
"""

import os, sys, argparse
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.config import SEQUENCE_LENGTH, TOTAL_DIM, DEVICE
from src.models.landmark_fusion_st import LandmarkFusionST


def export(ckpt_path: str, num_classes: int,
           output_path: str, use_ctc: bool = False,
           opset: int = 17):

    print(f"\n[ONNX Export]")
    print(f"  Checkpoint : {ckpt_path}")
    print(f"  Classes    : {num_classes}")
    print(f"  Output     : {output_path}")

    model = LandmarkFusionST(num_classes=num_classes,
                              dropout=0.0, use_ctc=use_ctc)
    sd = torch.load(ckpt_path, map_location="cpu")["model_state_dict"]
    model.load_state_dict(sd)
    model.eval()

    dummy = torch.randn(1, SEQUENCE_LENGTH, TOTAL_DIM)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    torch.onnx.export(
        model,
        (dummy,),
        output_path,
        opset_version=opset,
        input_names=["landmarks"],
        output_names=["logits", "features"],
        dynamic_axes={
            "landmarks": {0: "batch_size"},
            "logits":    {0: "batch_size"},
            "features":  {0: "batch_size"},
        },
        do_constant_folding=True,
    )
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"  Exported  : {output_path}  ({size_mb:.1f} MB)")
    print(f"  ONNX opset: {opset}")

    # Verify with onnxruntime if available
    try:
        import onnxruntime as ort
        import numpy as np
        sess = ort.InferenceSession(output_path,
                   providers=["CPUExecutionProvider"])
        x_np = dummy.numpy()
        out  = sess.run(None, {"landmarks": x_np})
        print(f"  ORT verify: logits {out[0].shape}  ✓")
    except ImportError:
        print("  (Install onnxruntime to verify: pip install onnxruntime)")
    print()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",        required=True)
    p.add_argument("--num_classes", type=int, required=True)
    p.add_argument("--output",      default="checkpoints/lmfst.onnx")
    p.add_argument("--use_ctc",     action="store_true")
    p.add_argument("--opset",       type=int, default=17)
    args = p.parse_args()
    export(args.ckpt, args.num_classes, args.output, args.use_ctc, args.opset)
