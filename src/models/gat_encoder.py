import torch
import torch.nn as nn
import torch.nn.functional as F
from config.config import (
    GAT_IN_DIM, GAT_HIDDEN_DIM, GAT_OUT_DIM,
    GAT_HEADS, GAT_LAYERS, GRAPH_POOL_DIM, NUM_NODES,
)


class GATLayer(nn.Module):
    
    def __init__(self, in_dim: int, out_dim: int, heads: int = 4,
                 concat: bool = True, dropout: float = 0.1,
                 leaky_slope: float = 0.2):
        super().__init__()
        self.heads   = heads
        self.out_dim = out_dim
        self.concat  = concat

        # W^l — linear projection per head
        self.W = nn.Linear(in_dim, heads * out_dim, bias=False)
        # Attention vector a (2 * out_dim per head)
        self.a = nn.Parameter(torch.empty(heads, 2 * out_dim))
        nn.init.xavier_uniform_(self.a.unsqueeze(0))

        self.leaky  = nn.LeakyReLU(leaky_slope)
        self.drop   = nn.Dropout(dropout)
        self.bn     = nn.BatchNorm1d(heads * out_dim if concat else out_dim)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        B, N, _ = h.shape
        H, D    = self.heads, self.out_dim

        # input: (B, N, H*D)  →  (B, H, N, D)
        Wh = self.W(h).view(B, N, H, D).permute(0, 2, 1, 3)  # (B, H, N, D)

        # Compute attention scores e_ij for every (i,j) pair
        # Wh_i : (B, H, N, 1, D)  broadcasts over j-dimension
        # Wh_j : (B, H, 1, N, D)  broadcasts over i-dimension
        Wh_i = Wh.unsqueeze(3)   # (B, H, N, 1, D)
        Wh_j = Wh.unsqueeze(2)   # (B, H, 1, N, D)

        # Concatenate along feature dim: (B, H, N, N, 2D)
        e_input = torch.cat([Wh_i.expand(B, H, N, N, D),
                              Wh_j.expand(B, H, N, N, D)], dim=-1)

        # a : (H, 2D)  →  (1, H, 1, 1, 2D)
        a_ = self.a.unsqueeze(0).unsqueeze(2).unsqueeze(3)     # (1,H,1,1,2D)
        e  = self.leaky((e_input * a_).sum(dim=-1))             # (B, H, N, N)

        # Zero out non-edges before softmax
        mask  = (adj == 0).unsqueeze(0).unsqueeze(0)            # (1,1,N,N)
        e = e.masked_fill(mask, float("-inf"))
        alpha = F.softmax(e, dim=-1)                            # (B, H, N, N)
        # Replace any NaN rows (fully masked nodes) with 0
        alpha = torch.nan_to_num(alpha, nan=0.0)
        alpha = self.drop(alpha)

        # Aggregate neighbours: (B, H, N, N) × (B, H, N, D) → (B, H, N, D)
        h_new = torch.matmul(alpha, Wh)                         # (B, H, N, D)
        h_new = h_new.permute(0, 2, 1, 3)                      # (B, N, H, D)

        if self.concat:
            h_new = h_new.reshape(B, N, H * D)                  # (B, N, H*D)
        else:
            h_new = h_new.mean(dim=2)                           # (B, N, D)

        # BatchNorm over (B*N, F)
        h_new = self.bn(h_new.reshape(B * N, -1)).reshape(B, N, -1)
        return F.elu(h_new)


class StreamGATEncoder(nn.Module):
    def __init__(self, n_nodes: int, in_dim: int = 3,
                 hidden_dim: int = 64, out_dim: int = 64,
                 heads: int = 4, pool_dim: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        self.n_nodes = n_nodes

        # Layer 1: 3 → 64  (concat → 4*64=256 intermediate... but paper says dout=64)
        # We use non-concat on final layer so output stays 64-D
        self.gat1 = GATLayer(in_dim,    hidden_dim, heads, concat=True,  dropout=dropout)
        self.gat2 = GATLayer(hidden_dim * heads, out_dim, heads, concat=False, dropout=dropout)

        self.proj = nn.Sequential(
            nn.Linear(out_dim, pool_dim),
            nn.LayerNorm(pool_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        B, T, N, F = x.shape
        # Merge batch and time: (B*T, N, F)
        x_flat = x.reshape(B * T, N, F)

        h = self.gat1(x_flat, adj)    # (B*T, N, heads*64)
        h = self.gat2(h, adj)         # (B*T, N, 64)

        # Global mean pooling over nodes
        h = h.mean(dim=1)             # (B*T, 64)
        h = self.proj(h)              # (B*T, pool_dim)
        return h.reshape(B, T, -1)   # (B, T, pool_dim)


class MultiStreamGAT(nn.Module):
    def __init__(self, dropout: float = 0.1):
        super().__init__()

        # Separate per-stream adjacency matrices — registered as buffers
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
        from src.graph_builder import build_adjacency_matrix
        A_full = build_adjacency_matrix()

        hand_adj = A_full[:42, :42].clone()
        face_adj = A_full[42:82, 42:82].clone()
        body_adj = A_full[82:93, 82:93].clone()

        self.register_buffer("hand_adj", hand_adj)
        self.register_buffer("face_adj", face_adj)
        self.register_buffer("body_adj", body_adj)

        # Per-stream encoders
        self.hand_enc = StreamGATEncoder(42,  3, 64, 64, GAT_HEADS, GRAPH_POOL_DIM, dropout)
        self.face_enc = StreamGATEncoder(40,  3, 64, 64, GAT_HEADS, GRAPH_POOL_DIM, dropout)
        self.body_enc = StreamGATEncoder(11,  3, 64, 64, GAT_HEADS, GRAPH_POOL_DIM, dropout)

        # Gated fusion: Eq. (viii)
        fused_dim = GRAPH_POOL_DIM * 3   # 768
        self.gate   = nn.Linear(fused_dim, fused_dim)
        self.proj_out = nn.Sequential(
            nn.Linear(fused_dim, GRAPH_POOL_DIM),
            nn.LayerNorm(GRAPH_POOL_DIM),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, T, 279) flat landmark vector
        Returns: (B, T, 256)
        """
        # Split into streams and reshape to (B, T, N, 3)
        hand_flat = x[..., :126]        # (B,T,126)
        face_flat = x[..., 126:246]     # (B,T,120)
        body_flat = x[..., 246:279]     # (B,T,33)

        B, T = x.shape[:2]
        hand_nodes = hand_flat.reshape(B, T, 42, 3)
        face_nodes = face_flat.reshape(B, T, 40, 3)
        body_nodes = body_flat.reshape(B, T, 11, 3)

        # Per-stream encoding: each → (B, T, 256)
        s_hand = self.hand_enc(hand_nodes, self.hand_adj)
        s_face = self.face_enc(face_nodes, self.face_adj)
        s_body = self.body_enc(body_nodes, self.body_adj)

        # Gated fusion: concat → gate → element-wise ⊙ → project
        cat  = torch.cat([s_hand, s_face, s_body], dim=-1)   # (B,T,768)
        gate = torch.sigmoid(self.gate(cat))                   # (B,T,768)
        fused = gate * cat                                     # (B,T,768)
        return self.proj_out(fused)                            # (B,T,256)
