
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config.config import (
    TCN_CHANNELS, TCN_KERNEL, TCN_BLOCKS,
    MHSA_HEADS, MHSA_D_MODEL, MHSA_FFN_DIM,
)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(x + self.pe[:, :x.size(1)])



class DilatedCausalBlock(nn.Module):
    def __init__(self, channels: int, kernel: int,
                 dilation: int, dropout: float = 0.1):
        super().__init__()
        pad = (kernel - 1) * dilation  # causal: left-only padding

        self.conv = nn.Conv1d(
            channels, channels, kernel,
            dilation=dilation,
            padding=pad,
        )
        self.bn   = nn.BatchNorm1d(channels)
        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)
        self._causal_pad = pad   # amount to trim from right

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, C, T)"""
        out = self.conv(x)
        # Remove future leak (causal convolution)
        if self._causal_pad > 0:
            out = out[:, :, :-self._causal_pad]
        out = self.bn(out)
        out = self.relu(out)
        out = self.drop(out)
        return out + x   # residual



class TCNMHSAEncoder(nn.Module):
    
    def __init__(self,
                 in_dim: int    = TCN_CHANNELS,
                 channels: int  = TCN_CHANNELS,
                 kernel: int    = TCN_KERNEL,
                 n_blocks: int  = TCN_BLOCKS,
                 n_heads: int   = MHSA_HEADS,
                 ffn_dim: int   = MHSA_FFN_DIM,
                 dropout: float = 0.1):
        super().__init__()

        # Input projection (in case in_dim ≠ channels)
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, channels),
            nn.LayerNorm(channels),
        ) if in_dim != channels else nn.Identity()

        # TCN blocks with exponentially increasing dilation
        self.tcn_blocks = nn.ModuleList([
            DilatedCausalBlock(channels, kernel, dilation=2**b, dropout=dropout)
            for b in range(n_blocks)
        ])

        # Positional encoding before MHSA
        self.pos_enc = PositionalEncoding(channels, dropout=dropout)

        # Multi-head self-attention + FFN (Pre-LN)
        self.norm1 = nn.LayerNorm(channels)
        self.attn  = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(channels)
        self.ffn   = nn.Sequential(
            nn.Linear(channels, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, channels),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, T, in_dim) → (B, T, channels)"""
        # Project input
        x = self.input_proj(x)       # (B, T, C)

        # TCN: operates on (B, C, T)
        z = x.transpose(1, 2)        # (B, C, T)
        for block in self.tcn_blocks:
            z = block(z)
        z = z.transpose(1, 2)        # (B, T, C)

        # Add positional encoding before attention
        z = self.pos_enc(z)

        # MHSA with Pre-LN
        residual = z
        z = self.norm1(z)
        attn_out, _ = self.attn(z, z, z)
        z = residual + attn_out

        # FFN with Pre-LN
        residual = z
        z = self.norm2(z)
        z = residual + self.ffn(z)

        return z   # (B, T, 256)
