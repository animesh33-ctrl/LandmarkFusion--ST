import os
import torch

# Device 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Dataset Paths  
# ISL-CSLTR Kaggle dataset root
# https://www.kaggle.com/datasets/drblack00/isl-csltr-indian-sign-language-dataset
DATASET_ROOT = r"C:\Sign Language\dataset\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus"

ISL_WORD_FRAMES    = os.path.join(DATASET_ROOT, "Frames_Word_Level")
ISL_SENTENCE_FRAMES = os.path.join(DATASET_ROOT, "Frames_Sentence_Level")
ISL_SENTENCE_VIDEOS = os.path.join(DATASET_ROOT, "Videos_Sentence_Level")

# RWTH-PHOENIX-Weather-2014 (continuous SLR benchmark)
# Download from: https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/
PHOENIX_ROOT       = r"C:\Sign Language\dataset\PHOENIX-2014"
PHOENIX_TRAIN      = os.path.join(PHOENIX_ROOT, "train")
PHOENIX_DEV        = os.path.join(PHOENIX_ROOT, "dev")
PHOENIX_TEST       = os.path.join(PHOENIX_ROOT, "test")
PHOENIX_VOCAB      = os.path.join(PHOENIX_ROOT, "gloss_vocab.txt")

PROJECT_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
CACHE_DIR      = os.path.join(PROJECT_ROOT, "cache")

# Landmark Dimensions 
# Hand:  21 keypoints × 2 hands × 3 coords = 126
# Face:  40 salient keypoints × 3 coords   = 120
# Body:  11 upper-body keypoints × 3 coords = 33
# Total: 279
HAND_DIM  = 126   # 21 × 2 hands × 3
FACE_DIM  = 120   # 40 × 3
BODY_DIM  = 33    # 11 × 3
TOTAL_DIM = 279   # 126 + 120 + 33

# Specific face keypoint indices from MediaPipe 468-mesh (40 salient points)
# Covers: eyebrows, lips, eye contours — grammatically relevant for ISL
FACE_KEYPOINT_INDICES = [
    # Left eyebrow (5)
    70, 63, 105, 66, 107,
    # Right eyebrow (5)
    336, 296, 334, 293, 300,
    # Left eye (6)
    33, 160, 158, 133, 153, 144,
    # Right eye (6)
    362, 385, 387, 263, 373, 380,
    # Lips outer (8)
    61, 185, 40, 39, 37, 0, 267, 269,
    # Lips inner (6)
    78, 191, 80, 81, 82, 13,
    # Nose tip (2)
    1, 4,
    # Chin (2)
    152, 148,
]  # Total = 40

# Upper-body pose indices from MediaPipe 33-point pose
BODY_KEYPOINT_INDICES = [11, 12, 13, 14, 15, 16, 23, 24, 0, 1, 2]  # 11 pts

# Sequence / Graph parameters 
SEQUENCE_LENGTH = 64          # frames per sequence (paper: 64)
NUM_NODES       = 93          # 21+21+40+11 = 93 landmark nodes
NODE_FEAT_DIM   = 3           # x, y, z per node

# Graph attention
GAT_IN_DIM      = 3           # input node feature dim
GAT_HIDDEN_DIM  = 64          # paper: dout = 64
GAT_OUT_DIM     = 64
GAT_HEADS       = 4           # K = 4 heads (paper)
GAT_LAYERS      = 2           # 2 stacked GAT layers
GRAPH_POOL_DIM  = 256         # after global mean pool + projection

# Multi-stream fusion
STREAM_FUSION_IN  = 768       # 256 × 3 streams
STREAM_FUSION_OUT = 256

# TCN + MHSA 
TCN_CHANNELS    = 256
TCN_KERNEL      = 3
TCN_BLOCKS      = 4           # dilations: 1, 2, 4, 8
MHSA_HEADS      = 8           # H = 8
MHSA_D_MODEL    = 256         # ds = 256
MHSA_D_K        = 32          # dk = ds/H = 32
MHSA_FFN_DIM    = 1024

# Training Hyperparameters
# Stages = 1–3 (Recognition)
RECOG_LR         = 3e-4
RECOG_WD         = 1e-4
RECOG_BATCH      = 16
RECOG_EPOCHS     = 120
RECOG_PATIENCE   = 15
RECOG_GRAD_CLIP  = 1.0
RECOG_LABEL_SMOOTH = 0.1
RECOG_BETA1      = 0.9
RECOG_BETA2      = 0.999

# Semantic Refinement
REFINE_LR        = 5e-5
REFINE_BATCH     = 32
REFINE_EPOCHS    = 30

IMG_SIZE         = 224
CNN_BATCH        = 32
CNN_LR           = 1e-3
CNN_EPOCHS       = 60
CNN_PATIENCE     = 8
CNN_DROPOUT      = 0.4
CNN_WD           = 1e-4

SEED             = 42
NUM_WORKERS      = 4
TRAIN_RATIO      = 0.70
VAL_RATIO        = 0.15
TEST_RATIO       = 0.15
PREDICTION_THRESH = 0.55
WEBCAM_INDEX     = 0
