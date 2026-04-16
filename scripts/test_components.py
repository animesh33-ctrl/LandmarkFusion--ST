"""
scripts/test_components.py
===========================
Smoke-tests every component of LandmarkFusion-ST without needing
real dataset files.  Run before training to verify the build.

    python scripts/test_components.py

Expected output: all tests PASS on CPU.
With RTX 5060 they should also PASS on CUDA faster.
"""

import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.config import (
    DEVICE, SEQUENCE_LENGTH, TOTAL_DIM, NUM_NODES,
    GRAPH_POOL_DIM, TCN_CHANNELS,
)

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
B    = 2     # batch size for tests
T    = SEQUENCE_LENGTH


def run(name, fn):
    try:
        fn()
        print(f"  {PASS}  {name}")
    except Exception as e:
        print(f"  {FAIL}  {name}")
        print(f"         {e}")


# ──────────────────────────────────────────────────────────────────────

def test_graph_builder():
    from src.graph_builder import (
        build_adjacency_matrix, normalize_adjacency, flat279_to_nodes)

    A = build_adjacency_matrix()
    assert A.shape == (NUM_NODES, NUM_NODES), f"A shape {A.shape}"
    assert (A == A.T).all(), "Adjacency must be symmetric"
    assert A.diagonal().all(), "Self-loops must be present"

    A_norm = normalize_adjacency(A)
    assert A_norm.shape == (NUM_NODES, NUM_NODES)
    assert not torch.isnan(A_norm).any()

    flat = torch.randn(B, T, TOTAL_DIM)
    nodes = flat279_to_nodes(flat)
    assert nodes.shape == (B, T, NUM_NODES, 3), f"nodes shape {nodes.shape}"


def test_gat_layer():
    from src.models.gat_encoder import GATLayer
    from src.graph_builder import build_adjacency_matrix

    A = build_adjacency_matrix()[:42, :42]
    layer = GATLayer(in_dim=3, out_dim=64, heads=4, concat=True)
    x = torch.randn(B, 42, 3)
    out = layer(x, A)
    assert out.shape == (B, 42, 256), f"GAT output {out.shape}"


def test_stream_gat():
    from src.models.gat_encoder import StreamGATEncoder
    from src.graph_builder import build_adjacency_matrix

    A = build_adjacency_matrix()[:42, :42]
    enc = StreamGATEncoder(n_nodes=42)
    x   = torch.randn(B, T, 42, 3)
    out = enc(x, A)
    assert out.shape == (B, T, GRAPH_POOL_DIM), f"StreamGAT {out.shape}"


def test_multistream_gat():
    from src.models.gat_encoder import MultiStreamGAT

    model = MultiStreamGAT()
    x     = torch.randn(B, T, TOTAL_DIM)
    out   = model(x)
    assert out.shape == (B, T, GRAPH_POOL_DIM), f"MultiStreamGAT {out.shape}"


def test_tcn_block():
    from src.models.tcn_encoder import DilatedCausalBlock

    block = DilatedCausalBlock(channels=256, kernel=3, dilation=2)
    x     = torch.randn(B, 256, T)
    out   = block(x)
    assert out.shape == (B, 256, T), f"TCNBlock {out.shape}"


def test_tcn_mhsa():
    from src.models.tcn_encoder import TCNMHSAEncoder

    enc = TCNMHSAEncoder()
    x   = torch.randn(B, T, TCN_CHANNELS)
    out = enc(x)
    assert out.shape == (B, T, TCN_CHANNELS), f"TCNMHSAEncoder {out.shape}"


def test_full_model_cls():
    from src.models.landmark_fusion_st import LandmarkFusionST

    model  = LandmarkFusionST(num_classes=50, use_ctc=False)
    x      = torch.randn(B, T, TOTAL_DIM)
    logits, feat = model(x)
    assert logits.shape == (B, 50),           f"cls logits {logits.shape}"
    assert feat.shape   == (B, T, TCN_CHANNELS), f"feat {feat.shape}"


def test_full_model_ctc():
    from src.models.landmark_fusion_st import LandmarkFusionST

    model  = LandmarkFusionST(num_classes=50, use_ctc=True)
    x      = torch.randn(B, T, TOTAL_DIM)
    logits, feat = model(x)
    assert logits.shape == (B, T, 50),        f"CTC logits {logits.shape}"
    assert feat.shape   == (B, T, TCN_CHANNELS), f"feat {feat.shape}"


def test_loss_cls():
    from src.models.landmark_fusion_st import LandmarkFusionST, LandmarkFusionLoss

    model    = LandmarkFusionST(num_classes=10, use_ctc=False)
    criterion = LandmarkFusionLoss(use_ctc=False)
    x        = torch.randn(B, T, TOTAL_DIM)
    labels   = torch.randint(0, 10, (B,))
    logits, feat = model(x)
    loss = criterion(logits, feat, labels)
    assert loss.item() > 0
    assert not torch.isnan(loss)


def test_ctc_decoder():
    from utils.ctc_decoder import greedy_ctc_decode, compute_wer

    log_p = torch.log_softmax(torch.randn(B, T, 20), dim=-1)
    seqs  = greedy_ctc_decode(log_p, blank_id=0)
    assert len(seqs) == B

    hyps = [["HELLO", "WORLD"], ["A", "B"]]
    refs = [["HELLO", "WORLD"], ["A", "C"]]
    wer  = compute_wer(hyps, refs)
    assert 0.0 <= wer <= 1.0, f"WER {wer}"


def test_gloss_vocab():
    from src.models.semantic_refiner import GlossVocab

    vocab = GlossVocab(["HELLO", "WORLD", "SIGN", "LANGUAGE"])
    assert len(vocab) == 7  # 3 specials + 4
    ids = vocab.encode(["HELLO", "SIGN", "UNKNOWN"])
    assert len(ids) == 3
    decoded = vocab.decode([vocab.encode(["HELLO"])[0]])
    assert "HELLO" in decoded


def test_landmark_normalization():
    from src.landmark_extractor import normalize_sequence

    seq = np.random.randn(T, TOTAL_DIM).astype(np.float32)
    out = normalize_sequence(seq)
    assert out.shape == (T, TOTAL_DIM)
    assert not np.isnan(out).any()


def test_dataset_classes():
    from src.dataset import LandmarkSeqDataset

    seqs   = np.random.randn(20, T, TOTAL_DIM).astype(np.float32)
    labels = [i % 5 for i in range(20)]
    ds     = LandmarkSeqDataset(seqs, labels, augment=True)
    assert len(ds) == 20
    x, y = ds[0]
    assert x.shape == (T, TOTAL_DIM)
    assert isinstance(y, int)


def test_model_summary():
    from src.models.landmark_fusion_st import LandmarkFusionST
    from utils.model_summary import model_summary, count_parameters

    model  = LandmarkFusionST(num_classes=50, use_ctc=False)
    stats  = count_parameters(model)
    assert stats["total"] > 0
    assert stats["trainable"] > 0


def test_adjacency_edge_count():
    """Verify ~280 edges as stated in paper Table 2."""
    from src.graph_builder import build_adjacency_matrix
    A     = build_adjacency_matrix(add_self_loops=False)
    edges = int(A.sum().item() / 2)   # undirected
    # Paper says ~280; our explicit graph should be in 240–340 range
    assert 240 <= edges <= 340, (
        f"Edge count {edges} outside expected range [240, 340]. "
        f"Check graph_builder.py connection lists."
    )
    print(f"    (edge count = {edges})", end="")


def test_cuda_if_available():
    """If CUDA is available, run a forward pass on GPU."""
    if not torch.cuda.is_available():
        return
    from src.models.landmark_fusion_st import LandmarkFusionST
    model = LandmarkFusionST(num_classes=50, use_ctc=False).cuda()
    x     = torch.randn(B, T, TOTAL_DIM).cuda()
    logits, _ = model(x)
    assert logits.shape == (B, 50)


# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  LandmarkFusion-ST — Component Tests")
    print("=" * 60)

    tests = [
        ("Graph Builder",               test_graph_builder),
        ("GAT Layer",                   test_gat_layer),
        ("Stream GAT Encoder",          test_stream_gat),
        ("Multi-Stream GAT + Fusion",   test_multistream_gat),
        ("Dilated Causal TCN Block",    test_tcn_block),
        ("TCN + MHSA Encoder",          test_tcn_mhsa),
        ("Full Model (Classification)", test_full_model_cls),
        ("Full Model (CTC)",            test_full_model_ctc),
        ("Loss Function",               test_loss_cls),
        ("Greedy CTC Decoder + WER",    test_ctc_decoder),
        ("Gloss Vocabulary",            test_gloss_vocab),
        ("Landmark Normalization",      test_landmark_normalization),
        ("Dataset Class",               test_dataset_classes),
        ("Model Summary",               test_model_summary),
        ("Adjacency Edge Count",        test_adjacency_edge_count),
        ("CUDA Forward Pass",           test_cuda_if_available),
    ]

    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  {PASS}  {name}")
            passed += 1
        except Exception as e:
            print(f"  {FAIL}  {name}")
            print(f"         └─ {e}")
            failed += 1

    print("=" * 60)
    print(f"  Results: {passed} passed, {failed} failed")
    print("=" * 60 + "\n")
    sys.exit(0 if failed == 0 else 1)
