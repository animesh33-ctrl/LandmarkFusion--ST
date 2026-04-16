
import os, sys, time, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.config import SEQUENCE_LENGTH, TOTAL_DIM, GRAPH_POOL_DIM


def timeit(fn, warmup=10, runs=100, device="cuda"):
    # Warmup
    for _ in range(warmup):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(runs):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / runs * 1000  # ms
    return elapsed


def benchmark(device_str: str = "cuda", runs: int = 200):
    device = torch.device(device_str if torch.cuda.is_available()
                          else "cpu")
    print(f"\n{'='*60}")
    print(f"  LandmarkFusion-ST — Inference Benchmark")
    print(f"  Device : {device}")
    if device.type == "cuda":
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
    print(f"  Runs   : {runs}")
    print(f"{'='*60}")

    B = 1   # single sample for latency measurement
    T = SEQUENCE_LENGTH

    # Dummy inputs
    flat_seq  = torch.randn(B, T, TOTAL_DIM, device=device)
    hand_in   = torch.randn(B, T, 42, 3, device=device)
    face_in   = torch.randn(B, T, 40, 3, device=device)
    body_in   = torch.randn(B, T, 11, 3, device=device)
    spatial   = torch.randn(B, T, GRAPH_POOL_DIM, device=device)
    temporal  = torch.randn(B, T, 256, device=device)

    from src.graph_builder import build_adjacency_matrix
    hand_adj = build_adjacency_matrix()[:42, :42].to(device)
    face_adj = build_adjacency_matrix()[42:82, 42:82].to(device)
    body_adj = build_adjacency_matrix()[82:93, 82:93].to(device)

    from src.models.gat_encoder   import StreamGATEncoder, MultiStreamGAT
    from src.models.tcn_encoder   import TCNMHSAEncoder
    from src.models.landmark_fusion_st import LandmarkFusionST
    from utils.ctc_decoder        import greedy_ctc_decode

    hand_enc  = StreamGATEncoder(42).to(device).eval()
    face_enc  = StreamGATEncoder(40).to(device).eval()
    body_enc  = StreamGATEncoder(11).to(device).eval()
    fusion    = MultiStreamGAT().to(device).eval()
    tcn_mhsa  = TCNMHSAEncoder().to(device).eval()
    full_model = LandmarkFusionST(num_classes=100, use_ctc=False).to(device).eval()

    results = {}

    with torch.no_grad():
        # GAT (hand stream only, representative)
        results["GAT Spatial (hand)"] = timeit(
            lambda: hand_enc(hand_in, hand_adj), runs=runs, device=device_str)

        # Multi-stream fusion
        results["MultiStream GAT+Fusion"] = timeit(
            lambda: fusion(flat_seq), runs=runs, device=device_str)

        # TCN + MHSA
        results["TCN + MHSA"] = timeit(
            lambda: tcn_mhsa(spatial), runs=runs, device=device_str)

        # Full model (Stages 2-5)
        results["Full Pipeline (Stages 2-5)"] = timeit(
            lambda: full_model(flat_seq), runs=runs, device=device_str)

        # CTC greedy decode
        log_p = torch.log_softmax(temporal @ torch.randn(256, 100, device=device), dim=-1)
        results["Greedy CTC Decode"] = timeit(
            lambda: greedy_ctc_decode(log_p.cpu()), runs=runs, device="cpu")

    print(f"\n  {'Component':<38} {'ms/frame':>10}  {'FPS':>8}")
    print("  " + "-" * 58)
    for name, ms in results.items():
        fps = 1000.0 / ms if ms > 0 else float("inf")
        print(f"  {name:<38} {ms:>10.2f}  {fps:>8.1f}")

    total_ms = results.get("Full Pipeline (Stages 2-5)", 0)
    print("  " + "-" * 58)
    print(f"  {'Estimated total (w/ MediaPipe ~12ms)':<38} "
          f"{total_ms+12:.2f}ms  {1000/(total_ms+12):.1f}")
    print(f"\n  Paper target: >30 FPS (22.8ms/frame)\n")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--runs",   default=200, type=int)
    args = p.parse_args()
    benchmark(args.device, args.runs)
