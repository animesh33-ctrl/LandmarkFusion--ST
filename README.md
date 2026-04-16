# LandmarkFusion-ST

**Multi-Stream Spatial-Temporal Landmark Network with Semantic Refinement**  
*Full PyTorch implementation of the paper by Animesh Palui & Kaushik Dutta, IEM Kolkata*

---

## Architecture Overview

```
Video Frames
    │
    ▼  Stage 1 — MediaPipe Holistic (frozen)
┌─────────────────────────────────────────────┐
│  l_t = [l_hand(126) ; l_face(120) ; l_body(33)]  ∈ R^279  │
└─────────────────────────────────────────────┘
    │
    ▼  Stage 2+3 — Multi-Stream GAT + Gated Fusion
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  Hand GAT    │  │  Face GAT    │  │  Body GAT    │
│  42 nodes    │  │  40 nodes    │  │  11 nodes    │
│  → R^256     │  │  → R^256     │  │  → R^256     │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       └──────────────────┴──────────────────┘
                          │ Gated Fusion  → R^256/frame
                          ▼  Stage 4 — TCN + MHSA
                ┌──────────────────────┐
                │ 4× Dilated Causal    │  dilations: 1,2,4,8
                │ Residual TCN Blocks  │
                │ + 8-head MHSA        │
                └──────────┬───────────┘
                           │  F ∈ R^(T×256)
                           ▼  Stage 5 — CTC / Classification
                     Ĝ (gloss sequence)
                           │
                           ▼  Stage 6 — Semantic Refiner
                  BERT-base (frozen) + Decoder Head
                           │
                     Ŷ (corrected text)
```

---

## Dataset

Download from Kaggle:  
**[ISL-CSLTR: Indian Sign Language Dataset](https://www.kaggle.com/datasets/drblack00/isl-csltr-indian-sign-language-dataset)**

Expected folder structure after extraction:
```
ISL_CSLRT_Corpus/
├── Frames_Word_Level/
│   ├── <class_name>/
│   │   ├── <sample_folder>/
│   │   │   ├── frame_001.jpg
│   │   │   └── ...
│   │   └── ...
│   └── ...
├── Frames_Sentence_Level/
│   └── <class_name>/<sample_folder>/<frames>
└── Videos_Sentence_Level/
    └── <class_name>/<video>.mp4
```

---

## Installation

```bash
# Clone / extract the project
cd LandmarkFusion_ST

# Create virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA 12.x (RTX 5060)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

---

## Quick Start

### 1. Configure Paths

Edit `config/config.py`:
```python
DATASET_ROOT = r"C:\path\to\ISL_CSLRT_Corpus"
```

### 2. Inspect Dataset
```bash
python scripts/inspect_dataset.py --root "C:/path/to/ISL_CSLRT_Corpus"
```

### 3. Run Component Tests (no GPU needed)
```bash
python scripts/test_components.py
```

### 4. Train

```bash
# Word-level isolated recognition
python main.py --mode train_word

# Sentence-level recognition
python main.py --mode train_sentence

# Train both, then launch inference
python main.py --mode all
```

### 5. Real-Time Inference
```bash
python main.py --mode inference
# or with specific checkpoints:
python inference/realtime.py \
    --word_ckpt checkpoints/best_lmfst_word.pth \
    --sentence_ckpt checkpoints/best_lmfst_isl.pth
```

### 6. Benchmark (RTX 5060)
```bash
python scripts/benchmark.py --device cuda --runs 200
```

### 7. Export to ONNX
```bash
python scripts/export_onnx.py \
    --ckpt checkpoints/best_lmfst_word.pth \
    --num_classes <N> \
    --output checkpoints/lmfst_word.onnx
```

---

## Project Structure

```
LandmarkFusion_ST/
├── main.py                           ← Entry point
├── requirements.txt
├── config/
│   └── config.py                    ← All hyperparameters (from paper)
├── src/
│   ├── landmark_extractor.py        ← Stage 1: 279-D MediaPipe extraction
│   ├── graph_builder.py             ← Anatomical adjacency (93 nodes)
│   ├── dataset.py                   ← Dataset + DataLoader
│   └── models/
│       ├── gat_encoder.py           ← Stage 2+3: GAT + gated fusion
│       ├── tcn_encoder.py           ← Stage 4: Dilated TCN + MHSA
│       ├── landmark_fusion_st.py    ← Full model + CTC/CE loss
│       └── semantic_refiner.py      ← Stage 6: BERT + decoder head
├── training/
│   ├── train_isolated.py            ← Word-level training
│   ├── train_continuous.py          ← Sentence-level training
│   └── train_refiner.py             ← BERT refiner fine-tuning
├── inference/
│   └── realtime.py                  ← Webcam real-time inference
├── utils/
│   ├── metrics.py                   ← AverageMeter, EarlyStopping, checkpointing
│   ├── ctc_decoder.py               ← Greedy CTC + WER
│   ├── visualize.py                 ← Training curves, confusion matrix
│   ├── model_summary.py             ← Parameter counter
│   └── path_checker.py              ← Dataset path validation
└── scripts/
    ├── test_components.py           ← Full test suite (16 tests)
    ├── inspect_dataset.py           ← Dataset structure inspector
    ├── benchmark.py                 ← Latency benchmark (Table 6)
    └── export_onnx.py               ← ONNX export for deployment
```

---

## Paper Hyperparameters (Implemented Exactly)

| Parameter | Value | Source |
|-----------|-------|--------|
| Landmark dim | 279 (126+120+33) | Section 3.2 |
| Graph nodes | 93 | Table 2 |
| GAT heads | K=4 | Section 3.3 |
| GAT layers | 2 | Section 3.3 |
| TCN blocks | B=4, d=1,2,4,8 | Section 3.4 |
| MHSA heads | H=8, d_k=32 | Section 3.4 |
| d_model | 256 | Table 2 |
| Optimizer | AdamW β=(0.9,0.999) | Section 6.1 |
| LR | 3×10⁻⁴ | Section 6.1 |
| Batch | 16 | Section 6.1 |
| Epochs | 120 (+patience 15) | Section 6.1 |
| Grad clip | 1.0 | Section 6.1 |
| Sequence length | 64 frames | Section 4.2 |

---

## Inference Controls (Webcam)

| Key | Action |
|-----|--------|
| `q` | Quit |
| `m` | Toggle word ↔ sentence mode |
| `c` | Clear history buffer |

---

## Expected Results (Paper)

| Task | Metric | Paper |
|------|--------|-------|
| ISL Isolated | Accuracy | 99.2% |
| PHOENIX-2014 | WER | 19.3% |
| ISL Sentences | Accuracy | 91.4% |
| Real-time | FPS | 31+ |

---

*Paper: "LandmarkFusion-ST: Multi-Stream Spatial-Temporal Landmark Network with Semantic Refinement to Sign Language Recognition and Translation in Real-Time" — Animesh Palui, Kaushik Dutta, IEM Kolkata*
