"""

  Stage 1: l_t = M(I_t)                                      [landmark extraction]
  Stage 2: s_t^m = GAT_m(l_t^m, A_m), mE{hand,face,body}    [spatial encoding]
  Stage 3: s_t = GatedFusion(s_t^hand, s_t^face, s_t^body)   [multi-stream fusion]
  Stage 4: F = MHSA(TCN({s_1...s_T}))                          [temporal encoding]
  Stage 5: G = CTCDecode(Linear(F))                          [CTC decoding]
  Stage 6: Y = Refiner(G)                                    [semantic refinement]

"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# Ensure project root is on path so all imports resolve cleanly
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.models.gat_encoder import MultiStreamGAT
from src.models.tcn_encoder import TCNMHSAEncoder
from config.config import TOTAL_DIM, TCN_CHANNELS, GRAPH_POOL_DIM


class LandmarkFusionST(nn.Module):
    def __init__(self,
                 num_classes: int,
                 dropout: float = 0.1,
                 use_ctc: bool = False):
        super().__init__()
        self.num_classes = num_classes
        self.use_ctc     = use_ctc

        # Stage 2+3: Multi-stream GAT + gated fusion → (B, T, 256)
        self.spatial_encoder = MultiStreamGAT(dropout=dropout)

        # Stage 4: TCN + MHSA → (B, T, 256)
        self.temporal_encoder = TCNMHSAEncoder(
            in_dim   = GRAPH_POOL_DIM,
            channels = TCN_CHANNELS,
            dropout  = dropout,
        )

        if use_ctc:
            # Stage 5a: per-frame CTC logits
            self.ctc_head = nn.Linear(TCN_CHANNELS, num_classes)
        else:
            # Stage 5b: global-average-pool → classification
            self.cls_head = nn.Sequential(
                nn.Linear(TCN_CHANNELS, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(256, num_classes),
            )

        # Auxiliary head used when use_ctc=True (paper Eq. L_total = L_CTC + λ L_cls)
        # Projects mean-pooled temporal features to class logits
        if use_ctc:
            self.aux_head = nn.Sequential(
                nn.Linear(TCN_CHANNELS, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, num_classes),
            )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        x      : (B, T, 279)
        Returns: (logits, feat)
          logits : (B, T, C) if use_ctc else (B, C)
          feat   : (B, T, 256) — temporal features for auxiliary loss
        """
        spatial = self.spatial_encoder(x)       # (B, T, 256)
        feat    = self.temporal_encoder(spatial) # (B, T, 256)

        if self.use_ctc:
            logits = self.ctc_head(feat)         # (B, T, C)
        else:
            pooled = feat.mean(dim=1)            # (B, 256)
            logits = self.cls_head(pooled)       # (B, C)

        return logits, feat


class LandmarkFusionLoss(nn.Module):
    def __init__(self,
                 blank_id: int       = 0,
                 lam: float          = 0.3,
                 label_smoothing: float = 0.1,
                 use_ctc: bool       = False):
        super().__init__()
        self.lam     = lam
        self.use_ctc = use_ctc

        self.ctc_loss = nn.CTCLoss(blank=blank_id, reduction="mean",
                                    zero_infinity=True)
        self.cls_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.aux_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self,
                logits: torch.Tensor,
                feat: Optional[torch.Tensor],
                targets: torch.Tensor,
                input_lengths:  Optional[torch.Tensor] = None,
                target_lengths: Optional[torch.Tensor] = None,
                aux_targets:    Optional[torch.Tensor] = None,
                aux_head: Optional[nn.Module]           = None,
                ) -> torch.Tensor:
        
        if not self.use_ctc:
            return self.cls_loss(logits, targets)

        # CTC loss 
        B, T, C = logits.shape
        log_p   = F.log_softmax(logits, dim=-1).permute(1, 0, 2)  # (T,B,C)

        if input_lengths is None:
            input_lengths = torch.full((B,), T, dtype=torch.long,
                                        device=logits.device)
        if target_lengths is None:
            target_lengths = torch.tensor([targets.numel() // B] * B,
                                           dtype=torch.long,
                                           device=logits.device)

        loss = self.ctc_loss(log_p, targets, input_lengths, target_lengths)

        # classification loss
        if (self.lam > 0 and feat is not None
                and aux_targets is not None and aux_head is not None):
            pooled   = feat.mean(dim=1)               # (B, 256)
            aux_logits = aux_head(pooled)              # (B, C)
            loss = loss + self.lam * self.aux_loss(aux_logits, aux_targets)

        return loss
