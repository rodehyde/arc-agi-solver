"""
model.py — ARC encoder-decoder with CLIP-style alignment.

Architecture:
  GridEncoder          40×40×10 one-hot → 256-dim embedding + UNet skip features
  TransformationEncoder  K (input_emb, output_emb) pairs → 256-dim transformation vector
  Decoder              transformation vector + query skip features → 40×40×10 logits
                       (FiLM modulation at each decoder scale)

The transformation vector is trained with two losses:
  1. Reconstruction loss   cross-entropy on predicted vs actual output grid
  2. Alignment loss        InfoNCE — transformation vector should be cosine-similar
                           to the MiniLM description embedding for the same task
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class GridEncoder(nn.Module):
    """Encode a padded 40×40 one-hot grid to a 256-dim embedding.

    Also returns intermediate feature maps for UNet skip connections.

    Input:  (B, 10, 40, 40) float — one-hot colour channels
    Output: embedding (B, embed_dim), skips tuple (s0, s1, s2, s3)
              s0: (B, C,   40, 40)
              s1: (B, C*2, 20, 20)
              s2: (B, C*4, 10, 10)
              s3: (B, C*8,  5,  5)
    """

    def __init__(self, in_channels: int = 10, base_channels: int = 32,
                 embed_dim: int = 256):
        super().__init__()
        C = base_channels
        self.stem  = ConvBlock(in_channels, C)       # 40×40
        self.down1 = ConvBlock(C,    C * 2, stride=2)  # 20×20
        self.down2 = ConvBlock(C * 2, C * 4, stride=2) # 10×10
        self.down3 = ConvBlock(C * 4, C * 8, stride=2) #  5×5
        self.pool  = nn.AdaptiveAvgPool2d(1)
        self.proj  = nn.Linear(C * 8, embed_dim)

    def forward(self, x: torch.Tensor):
        s0 = self.stem(x)
        s1 = self.down1(s0)
        s2 = self.down2(s1)
        s3 = self.down3(s2)
        emb = self.proj(self.pool(s3).flatten(1))
        return emb, (s0, s1, s2, s3)


class TransformationEncoder(nn.Module):
    """Encode K (input_emb, output_emb) pairs to a transformation vector.

    Input:  input_embs  (B, K, embed_dim)
            output_embs (B, K, embed_dim)
    Output: (B, embed_dim)
    """

    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.pair_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, input_embs: torch.Tensor,
                output_embs: torch.Tensor) -> torch.Tensor:
        pairs = torch.cat([input_embs, output_embs], dim=-1)  # (B, K, 2D)
        return self.pair_proj(pairs).mean(dim=1)               # (B, D)


class FiLM(nn.Module):
    """Feature-wise Linear Modulation: condition a feature map on a vector."""

    def __init__(self, condition_dim: int, num_channels: int):
        super().__init__()
        self.gamma_proj = nn.Linear(condition_dim, num_channels)
        self.beta_proj  = nn.Linear(condition_dim, num_channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W),  cond: (B, D)
        gamma = self.gamma_proj(cond).unsqueeze(-1).unsqueeze(-1)
        beta  = self.beta_proj(cond).unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta


class Decoder(nn.Module):
    """UNet decoder: upsample from 5×5 to 40×40, conditioned on transformation vector.

    Skip connections come from the query input's GridEncoder.
    FiLM modulation is applied after each upsampling stage.

    Input:  skips from query GridEncoder (s0, s1, s2, s3)
            transform_vec (B, embed_dim)
    Output: logits (B, 10, 40, 40)
    """

    def __init__(self, base_channels: int = 32, embed_dim: int = 256,
                 out_channels: int = 10):
        super().__init__()
        C = base_channels

        # Each ConvTranspose doubles spatial size; input channels = concat of up + skip
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(C * 8, C * 4, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(C * 4), nn.GELU(),
        )  # s3 (C*8, 5) → (C*4, 10); then concat s2 (C*4) → C*8 into up2

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(C * 8, C * 2, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(C * 2), nn.GELU(),
        )  # (C*8, 10) → (C*2, 20); then concat s1 (C*2) → C*4 into up1

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(C * 4, C, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(C), nn.GELU(),
        )  # (C*4, 20) → (C, 40); then concat s0 (C) → C*2 into out_conv

        self.out_conv = nn.Conv2d(C * 2, out_channels, kernel_size=1)

        self.film3 = FiLM(embed_dim, C * 4)
        self.film2 = FiLM(embed_dim, C * 2)
        self.film1 = FiLM(embed_dim, C)

    @staticmethod
    def _crop(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """Crop x to the spatial size of ref (handles odd-dim rounding in stride-2 ops)."""
        return x[:, :, :ref.shape[2], :ref.shape[3]]

    def forward(self, skips: tuple, transform_vec: torch.Tensor) -> torch.Tensor:
        s0, s1, s2, s3 = skips

        x = self._crop(self.up3(s3), s2)            # (B, C*4, h2, w2)
        x = self.film3(x, transform_vec)
        x = torch.cat([x, s2], dim=1)               # (B, C*8, h2, w2)

        x = self._crop(self.up2(x), s1)             # (B, C*2, h1, w1)
        x = self.film2(x, transform_vec)
        x = torch.cat([x, s1], dim=1)               # (B, C*4, h1, w1)

        x = self._crop(self.up1(x), s0)             # (B, C, h0, w0)
        x = self.film1(x, transform_vec)
        x = torch.cat([x, s0], dim=1)               # (B, C*2, h0, w0)

        return self.out_conv(x)                     # (B, 10, h0, w0)


class ARCSolver(nn.Module):
    """Full ARC solver model.

    Forward pass:
      context_inputs  (B, K, 10, 40, 40) — K demonstration input grids
      context_outputs (B, K, 10, 40, 40) — K demonstration output grids
      query_input     (B, 10, 40, 40)    — test input grid

    Returns:
      logits          (B, 10, 40, 40)    — predicted output (before softmax)
      transform_vec   (B, embed_dim)     — for alignment loss
    """

    def __init__(self, base_channels: int = 32, embed_dim: int = 256):
        super().__init__()
        self.grid_encoder = GridEncoder(
            in_channels=10, base_channels=base_channels, embed_dim=embed_dim
        )
        self.transform_encoder = TransformationEncoder(embed_dim=embed_dim)
        self.decoder = Decoder(
            base_channels=base_channels, embed_dim=embed_dim, out_channels=10
        )

    def encode_grid(self, grid: torch.Tensor):
        """Encode a (B, 10, 40, 40) grid; returns (emb, skips)."""
        return self.grid_encoder(grid)

    def forward(self, context_inputs: torch.Tensor,
                context_outputs: torch.Tensor,
                query_input: torch.Tensor):
        B, K = context_inputs.shape[:2]

        # Encode K context inputs and outputs
        ctx_in  = context_inputs.view(B * K, *context_inputs.shape[2:])
        ctx_out = context_outputs.view(B * K, *context_outputs.shape[2:])
        in_embs,  _ = self.grid_encoder(ctx_in)   # (B*K, D)
        out_embs, _ = self.grid_encoder(ctx_out)
        in_embs  = in_embs.view(B, K, -1)          # (B, K, D)
        out_embs = out_embs.view(B, K, -1)

        # Build transformation vector
        transform_vec = self.transform_encoder(in_embs, out_embs)  # (B, D)

        # Encode query input (keeping skip features for decoder)
        _, query_skips = self.grid_encoder(query_input)

        # Decode
        logits = self.decoder(query_skips, transform_vec)  # (B, 10, 40, 40)
        return logits, transform_vec


class AlignmentProjection(nn.Module):
    """Project transformation vectors and MiniLM embeddings to a shared space.

    CLIP uses separate linear projections for each modality. We do the same:
      transform_vec (embed_dim=256) → proj_dim
      desc_emb      (desc_dim=384)  → proj_dim
    """

    def __init__(self, embed_dim: int = 256, desc_dim: int = 384,
                 proj_dim: int = 128):
        super().__init__()
        self.transform_proj = nn.Linear(embed_dim, proj_dim)
        self.desc_proj      = nn.Linear(desc_dim, proj_dim)

    def forward(self, transform_vec: torch.Tensor,
                desc_emb: torch.Tensor):
        return self.transform_proj(transform_vec), self.desc_proj(desc_emb)


def alignment_loss(tv_proj: torch.Tensor,
                   de_proj: torch.Tensor,
                   temperature: float = 0.1) -> torch.Tensor:
    """InfoNCE contrastive loss on projected vectors (both already in proj_dim space).

    Positive pairs: same task (diagonal).
    Negative pairs: all other tasks in the batch.
    """
    tv = F.normalize(tv_proj, dim=-1)
    de = F.normalize(de_proj, dim=-1)

    logits = torch.matmul(tv, de.T) / temperature  # (B, B)
    labels = torch.arange(logits.size(0), device=logits.device)

    # Symmetric cross-entropy (same as CLIP)
    loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
    return loss
