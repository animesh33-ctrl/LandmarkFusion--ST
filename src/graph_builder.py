
import torch
import numpy as np
from typing import List, Tuple

# ── Offset constants ─────────────────────────────────────────────────
LH_OFF   = 0    # left-hand  0–20
RH_OFF   = 21   # right-hand 21–41
FACE_OFF = 42   # face       42–81
BODY_OFF = 82   # body       82–92
N_NODES  = 93

# ── Hand connections (21 nodes) ───────────────────────────────────────
HAND_CONNECTIONS = [
    # Finger chains
    (0,1),(1,2),(2,3),(3,4),           # thumb
    (0,5),(5,6),(6,7),(7,8),           # index
    (0,9),(9,10),(10,11),(11,12),      # middle
    (0,13),(13,14),(14,15),(15,16),    # ring
    (0,17),(17,18),(18,19),(19,20),    # pinky
    # Palm transversal
    (5,9),(9,13),(13,17),
    # Fingertip cross (adjacent fingers)
    (4,8),(8,12),(12,16),(16,20),
    # Knuckle row
    (5,6),(6,9),(9,10),(10,13),(13,14),(14,17),
    # Extra palm
    (1,5),(1,9),(1,13),(1,17),(2,6),(3,7),
    # Wrist to all MCP joints
    (0,6),(0,10),(0,14),(0,18),
    # Cross-finger at mid level
    (2,6),(3,6),(6,10),(7,10),(10,14),(11,14),(14,18),(15,18),
    # Cross-finger at tip level
    (4,7),(8,11),(12,15),(16,19),
    # Additional palm diagonals
    (5,13),(9,17),(6,13),(10,17),
]
HAND_CONNECTIONS = list(set(
    (min(u,v), max(u,v)) for u,v in HAND_CONNECTIONS
    if 0 <= u <= 20 and 0 <= v <= 20
))

# ── Face connections (40 nodes, local indices 0–39) ───────────────────
# Node groups (matching FACE_KEYPOINT_INDICES in config.py):
#  0-4 : left eyebrow   5-9 : right eyebrow
#  10-15: left eye ring  16-21: right eye ring
#  22-29: outer lip      30-35: inner lip
#  36-37: nose           38-39: chin
FACE_CONNECTIONS = [
    # Left eyebrow — chain + skip
    (0,1),(1,2),(2,3),(3,4),(0,2),(1,3),(2,4),(0,3),(1,4),
    # Right eyebrow — chain + skip
    (5,6),(6,7),(7,8),(8,9),(5,7),(6,8),(7,9),(5,8),(6,9),
    # Left eye — ring + diagonals
    (10,11),(11,12),(12,13),(13,14),(14,15),(15,10),
    (10,12),(11,13),(12,14),(13,15),(10,13),(11,14),
    # Right eye — ring + diagonals
    (16,17),(17,18),(18,19),(19,20),(20,21),(21,16),
    (16,18),(17,19),(18,20),(19,21),(16,19),(17,20),
    # Outer lip — ring + skip
    (22,23),(23,24),(24,25),(25,26),(26,27),(27,28),(28,29),(29,22),
    (22,24),(23,25),(24,26),(25,27),(26,28),(27,29),(22,25),(23,26),(24,27),(25,28),
    # Inner lip — ring + skip
    (30,31),(31,32),(32,33),(33,34),(34,35),(35,30),
    (30,32),(31,33),(32,34),(33,35),(30,33),(31,34),
    # Brow ↔ eye (vertical)
    (0,10),(1,11),(2,12),(3,13),(4,14),(10,15),(16,21),
    (5,16),(6,17),(7,18),(8,19),(9,20),
    # Brow ↔ brow (bridge of nose)
    (0,5),(1,6),(2,7),(3,8),(4,9),
    # Outer ↔ inner lip
    (22,30),(23,31),(24,32),(25,33),(26,34),(27,35),
    (28,34),(29,35),(22,31),(23,30),
    # Nose
    (36,37),(36,38),(37,39),(38,39),
    # Nose ↔ eyes/brows
    (36,10),(36,16),(37,10),(37,16),
    (36,2),(36,7),(37,2),(37,7),
    # Nose ↔ lips
    (36,22),(37,29),(36,25),(37,26),
    # Chin ↔ lower lip
    (38,25),(38,26),(38,27),(39,23),(39,24),(39,28),
    # Eye corners ↔ lip corners
    (10,22),(21,29),(15,16),
    # Chin ↔ inner lip
    (38,32),(39,31),
]
FACE_CONNECTIONS = list(set(
    (min(u,v), max(u,v)) for u,v in FACE_CONNECTIONS
    if 0 <= u <= 39 and 0 <= v <= 39
))

# ── Body connections (11 nodes, local indices 0–10) ───────────────────
# 0=l_shoulder,1=r_shoulder,2=l_elbow,3=r_elbow,
# 4=l_wrist,5=r_wrist,6=l_hip,7=r_hip,8=nose,9=l_eye,10=r_eye
BODY_CONNECTIONS = [
    (0,1),                   # shoulder bar
    (0,2),(2,4),             # left arm
    (1,3),(3,5),             # right arm
    (0,6),(1,7),(6,7),       # torso
    (8,9),(8,10),(9,10),     # head
    (0,8),(1,8),             # shoulder–head
    (0,4),(1,5),             # shoulder–wrist shortcut
]
BODY_CONNECTIONS = list(set(
    (min(u,v), max(u,v)) for u,v in BODY_CONNECTIONS
))

# ── Cross-stream edges (absolute indices) ────────────────────────────
CROSS_STREAM_EDGES = [
    (LH_OFF  + 0,  BODY_OFF + 4),    # left wrist hand  ↔ left wrist body
    (RH_OFF  + 0,  BODY_OFF + 5),    # right wrist hand ↔ right wrist body
    (LH_OFF  + 0,  BODY_OFF + 2),    # left wrist ↔ left elbow body
    (RH_OFF  + 0,  BODY_OFF + 3),    # right wrist ↔ right elbow body
    (BODY_OFF + 8, FACE_OFF + 36),   # nose body ↔ nose face
    (BODY_OFF + 9, FACE_OFF + 10),   # left eye body ↔ left eye face
    (BODY_OFF + 10, FACE_OFF + 16),  # right eye body ↔ right eye face
]


def build_adjacency_matrix(add_self_loops: bool = True) -> torch.Tensor:
    """
    Returns fixed anatomical adjacency A ∈ {0,1}^(93×93), symmetric.
    Target: ~280 undirected edges (paper Table 2).
    """
    A = np.zeros((N_NODES, N_NODES), dtype=np.float32)

    def _add(edges: List[Tuple[int, int]], offset: int = 0):
        for u, v in edges:
            i, j = u + offset, v + offset
            A[i, j] = 1.0
            A[j, i] = 1.0

    _add(HAND_CONNECTIONS, LH_OFF)
    _add(HAND_CONNECTIONS, RH_OFF)
    _add(FACE_CONNECTIONS, FACE_OFF)
    _add(BODY_CONNECTIONS, BODY_OFF)
    _add(CROSS_STREAM_EDGES)           # already absolute indices

    if add_self_loops:
        np.fill_diagonal(A, 1.0)

    return torch.tensor(A, dtype=torch.float32)


def normalize_adjacency(A: torch.Tensor) -> torch.Tensor:
    """Symmetric normalisation: D^{-1/2} A D^{-1/2}"""
    deg = A.sum(dim=1)
    d   = deg.pow(-0.5)
    d[torch.isinf(d)] = 0.0
    return torch.diag(d) @ A @ torch.diag(d)


def flat279_to_nodes(flat: torch.Tensor) -> torch.Tensor:
    """
    (..., 279) → (..., 93, 3)
    Splits: hand(126) | face(120) | body(33)
    """
    bs   = flat.shape[:-1]
    hand = flat[..., :126].reshape(*bs, 42, 3)
    face = flat[..., 126:246].reshape(*bs, 40, 3)
    body = flat[..., 246:279].reshape(*bs, 11, 3)
    return torch.cat([hand, face, body], dim=-2)  # (..., 93, 3)