"""
transformer_model.py — Decoder-only transformer for ARC in-context learning.

Design:
  • Input: (B, T, 5) integer tensor — one row per token, 5 features each:
      [0] token_id     (colour 0-9 or special token)
      [1] col          (1-based column within grid, 0 for special tokens)
      [2] row          (1-based row within grid, 0 for special tokens)
      [3] color_change (1 if colour changed from previous cell, else 0)
      [4] grid_number  (which grid in the sequence, 0 = before first grid)

  • Token embedding (full d_model) + sinusoidal row/col encodings (d_model//4
    each, projected) + learned grid-number segment embedding (d_model//4,
    projected) + learned color-change embedding (d_model//8, projected).
    Learnable scale parameters balance the spatial/structural encodings against
    the token embedding (following the approach in Michael Hodel's ARC solver).

  • Pre-norm decoder-only transformer with causal self-attention.
    Uses F.scaled_dot_product_attention for memory efficiency.

  • Loss computed only on test output tokens (caller provides loss_mask).
  • No weight tying between input embedding and output head — with a small
    vocabulary the tied weights can conflate input-recognition and output-
    generation representations.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.arc_tokenizer import VOCAB_SIZE, PAD


# ---------------------------------------------------------------------------
# Sinusoidal positional encoding (fixed, not learned)
# ---------------------------------------------------------------------------

def make_sinusoidal_encoding(max_len: int, dim: int) -> torch.Tensor:
    """Return (max_len, dim) sinusoidal positional encoding."""
    pe  = torch.zeros(max_len, dim)
    pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe   # (max_len, dim)


# ---------------------------------------------------------------------------
# Attention and transformer block
# ---------------------------------------------------------------------------

class MultiHeadCausalAttention(nn.Module):
    """Multi-head causal self-attention using scaled_dot_product_attention."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.dropout  = dropout
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, D = x.shape
        H = self.n_heads
        qkv = self.qkv(x).reshape(B, T, 3, H, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        dp  = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=dp)
        return self.out(out.transpose(1, 2).reshape(B, T, D))

    def forward_and_cache(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None):
        """Like forward() but also returns K, V for KV caching."""
        B, T, D = x.shape
        H = self.n_heads
        qkv = self.qkv(x).reshape(B, T, 3, H, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        dp  = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=dp)
        return self.out(out.transpose(1, 2).reshape(B, T, D)), k, v

    def forward_with_kv(self, x_new: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor):
        """Decode a single new token attending to all cached past tokens.

        x_new   : (B, 1, D)
        k_cache : (B, H, T_past, head_dim)
        v_cache : (B, H, T_past, head_dim)
        Returns : (output (B, 1, D), k_full, v_full) — cache extended by 1
        """
        B, _, D = x_new.shape
        H = self.n_heads
        qkv = self.qkv(x_new).reshape(B, 1, 3, H, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k_new, v_new = qkv[0], qkv[1], qkv[2]
        k = torch.cat([k_cache, k_new], dim=2)
        v = torch.cat([v_cache, v_new], dim=2)
        # Single query attends to all past tokens — no causal mask needed
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        return self.out(out.transpose(1, 2).reshape(B, 1, D)), k, v


class TransformerBlock(nn.Module):
    """Pre-norm: LN → attn → residual, LN → FFN → residual."""

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.attn = MultiHeadCausalAttention(d_model, n_heads, dropout)
        self.ln2  = nn.LayerNorm(d_model)
        self.ffn  = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), pad_mask)
        x = x + self.ffn(self.ln2(x))
        return x

    def forward_and_cache(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None):
        normed = self.ln1(x)
        attn_out, k, v = self.attn.forward_and_cache(normed, pad_mask)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, k, v

    def forward_with_kv(self, x_new: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor):
        normed = self.ln1(x_new)
        attn_out, k, v = self.attn.forward_with_kv(normed, k_cache, v_cache)
        x_new = x_new + attn_out
        x_new = x_new + self.ffn(self.ln2(x_new))
        return x_new, k, v


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class ArcTransformer(nn.Module):
    """Decoder-only transformer for ARC in-context grid prediction.

    Input:
        x        (B, T, 5) int64  — 5 features per token (see module docstring)
        pad_mask (B, T)    bool   — True where token is padding

    Output:
        logits   (B, T, vocab_size)

    Embedding construction (following Hodel 2024):
        spatial_dim  = d_model // 4
        segment_dim  = d_model // 4
        change_dim   = d_model // 8

        row_enc  = sinusoidal(row,  spatial_dim)   projected to d_model
        col_enc  = sinusoidal(col,  spatial_dim)   projected to d_model
        seg_enc  = learned grid_number embedding   projected to d_model
        chg_enc  = learned color_change embedding  (dim=2 → d_model)
        seq_pos  = learned absolute sequence pos   d_model

        x = tok_emb + scale[0]*row_enc + scale[1]*col_enc
              + scale[2]*seg_enc + chg_enc + seq_pos

    scale is a learnable 3-vector (initialised from Hodel's tuned values).
    """

    def __init__(
        self,
        vocab_size:    int   = VOCAB_SIZE,
        d_model:       int   = 256,
        n_heads:       int   = 8,
        n_layers:      int   = 6,
        max_seq_len:   int   = 2048,
        max_grid_dim:  int   = 32,    # max rows or cols in any grid
        max_grids:     int   = 24,    # max grid_number value (supports K≤11 pairs)
        ffn_dim:       int | None = None,
        dropout:       float = 0.1,
    ):
        super().__init__()
        if ffn_dim is None:
            ffn_dim = 4 * d_model

        self.d_model = d_model
        spatial_dim  = max(d_model // 4, 8)
        spatial_dim  = spatial_dim if spatial_dim % 2 == 0 else spatial_dim + 1

        # ----- token embedding -----
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD)

        # ----- absolute sequence position (learned) -----
        self.seq_pos_emb = nn.Embedding(max_seq_len, d_model)

        # ----- sinusoidal row/col encodings (fixed, registered as buffers) -----
        self.register_buffer("row_sin",
                             make_sinusoidal_encoding(max_grid_dim + 1, spatial_dim))
        self.register_buffer("col_sin",
                             make_sinusoidal_encoding(max_grid_dim + 1, spatial_dim))
        self.row_proj = nn.Linear(spatial_dim, d_model, bias=False)
        self.col_proj = nn.Linear(spatial_dim, d_model, bias=False)

        # ----- grid-number segment embedding (learned) -----
        segment_dim = max(d_model // 4, 8)
        self.seg_emb  = nn.Embedding(max_grids, segment_dim)
        self.seg_proj = nn.Linear(segment_dim, d_model, bias=False)

        # ----- color-change binary embedding (learned) -----
        self.chg_emb = nn.Embedding(2, d_model)

        # ----- learnable scale balancing spatial/structural vs token -----
        # Initialised to Hodel's tuned values (row, col, segment)
        self.enc_scale = nn.Parameter(torch.tensor([2.4002, 2.6549, 1.2614]))

        # ----- transformer blocks -----
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ffn_dim, dropout)
            for _ in range(n_layers)
        ])

        self.ln_final = nn.LayerNorm(d_model)

        # ----- output head (NOT weight-tied — see module docstring) -----
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()

    def _build_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Construct the full per-token embedding from the 5-feature input.

        x : (B, T, 5) int64
        Returns (B, T, d_model)
        """
        B, T, _ = x.shape
        x = x.long()             # ensure int64 for all embedding/index ops
        tok_ids  = x[:, :, 0]   # (B, T)
        col_ids  = x[:, :, 1]   # (B, T) — 1-based, 0 for specials
        row_ids  = x[:, :, 2]   # (B, T)
        chg_ids  = x[:, :, 3]   # (B, T) — 0 or 1
        seg_ids  = x[:, :, 4]   # (B, T)
        max_pos  = self.seq_pos_emb.num_embeddings - 1
        seq_pos  = torch.arange(T, device=x.device).clamp(0, max_pos).unsqueeze(0)  # (1, T)

        # Clamp ids to valid ranges (safety for any out-of-range tokens)
        row_ids = row_ids.clamp(0, self.row_sin.shape[0] - 1)
        col_ids = col_ids.clamp(0, self.col_sin.shape[0] - 1)
        seg_ids = seg_ids.clamp(0, self.seg_emb.num_embeddings - 1)
        chg_ids = chg_ids.clamp(0, 1)

        row_enc = self.row_proj(self.row_sin[row_ids])   # (B, T, d_model)
        col_enc = self.col_proj(self.col_sin[col_ids])   # (B, T, d_model)
        seg_enc = self.seg_proj(self.seg_emb(seg_ids))   # (B, T, d_model)

        s = self.enc_scale
        emb = (self.tok_emb(tok_ids)
               + s[0] * row_enc
               + s[1] * col_enc
               + s[2] * seg_enc
               + self.chg_emb(chg_ids)
               + self.seq_pos_emb(seq_pos))
        return emb

    def forward(
        self,
        x:        torch.Tensor,   # (B, T, 5) int64
        pad_mask: torch.Tensor,   # (B, T) bool — True = padding
    ) -> torch.Tensor:            # (B, T, vocab_size)
        emb = self._build_embedding(x)
        for block in self.blocks:
            emb = block(emb, pad_mask)
        return self.head(self.ln_final(emb))

    # ------------------------------------------------------------------
    # KV-cached greedy inference
    # ------------------------------------------------------------------

    def _prefill(self, prefix: torch.Tensor, pad_mask: torch.Tensor):
        """Encode the entire prefix in one forward pass.

        Returns (logits_at_last_pos, kv_caches) where:
          logits_at_last_pos : (B, vocab_size) — prediction after the last prefix token
          kv_caches          : list of (k, v) per transformer block,
                               each (B, n_heads, T_prefix, head_dim)
        """
        emb = self._build_embedding(prefix)
        h   = emb
        kv_caches = []
        for block in self.blocks:
            h, k, v = block.forward_and_cache(h, pad_mask)
            kv_caches.append((k, v))
        logits = self.head(self.ln_final(h[:, -1:, :]))   # (B, 1, vocab_size)
        return logits[:, 0, :], kv_caches

    def _build_embedding_at_pos(self, feat: torch.Tensor, abs_pos: int) -> torch.Tensor:
        """Build embedding for a single token at a specific absolute sequence position.

        feat    : (B, 1, 5) int64
        abs_pos : absolute index of this token in the full sequence
        Returns : (B, 1, d_model)
        """
        x = feat.long()
        tok_ids = x[:, :, 0]
        col_ids = x[:, :, 1].clamp(0, self.col_sin.shape[0] - 1)
        row_ids = x[:, :, 2].clamp(0, self.row_sin.shape[0] - 1)
        chg_ids = x[:, :, 3].clamp(0, 1)
        seg_ids = x[:, :, 4].clamp(0, self.seg_emb.num_embeddings - 1)

        max_pos = self.seq_pos_emb.num_embeddings - 1
        pos_idx = torch.tensor([[abs_pos]], device=feat.device).clamp(0, max_pos)

        s = self.enc_scale
        return (self.tok_emb(tok_ids)
                + s[0] * self.row_proj(self.row_sin[row_ids])
                + s[1] * self.col_proj(self.col_sin[col_ids])
                + s[2] * self.seg_proj(self.seg_emb(seg_ids))
                + self.chg_emb(chg_ids)
                + self.seq_pos_emb(pos_idx))

    def _decode_step(self, feat: torch.Tensor, abs_pos: int, kv_caches: list):
        """Process one new token, extend KV caches, return logit for the next token.

        feat      : (B, 1, 5) int64 — features of the new token
        abs_pos   : absolute sequence position of this token
        kv_caches : list of (k, v) per block (from _prefill or previous step)
        Returns   : (logits (B, vocab_size), new_kv_caches)
        """
        h = self._build_embedding_at_pos(feat, abs_pos)
        new_kv = []
        for block, (k_cache, v_cache) in zip(self.blocks, kv_caches):
            h, k, v = block.forward_with_kv(h, k_cache, v_cache)
            new_kv.append((k, v))
        logits = self.head(self.ln_final(h[:, -1:, :]))[:, 0, :]   # (B, vocab_size)
        return logits, new_kv

    @torch.no_grad()
    def generate(
        self,
        prefix:      torch.Tensor,   # (1, T_prefix, 5)
        pad_mask:    torch.Tensor,   # (1, T_prefix) — all False
        out_height:  int,
        out_width:   int,
        grid_number: int,
    ) -> np.ndarray:
        """Greedily generate one output grid using pre-allocated KV buffers.

        Phase 1 — prefill: encode the full prefix in one forward pass,
        storing K and V for every layer into pre-allocated buffers.

        Phase 2 — decode: for each output cell, run a single-token forward
        pass that writes the new K/V directly into the buffer (no allocation,
        no copying of previous cache data) and reads a contiguous view for
        the attention computation.

        Cost: O(T_prefix²) for prefill + O(H·W·T_total) for decode,
        with zero dynamic memory allocation during decode.

        Returns a (out_height, out_width) uint8 array.
        """
        from src.arc_tokenizer import ROW_SEP

        T_prefix  = prefix.shape[1]
        B         = prefix.shape[0]
        H         = self.blocks[0].attn.n_heads
        hd        = self.blocks[0].attn.head_dim
        D         = self.d_model
        device    = prefix.device
        max_new   = out_height * out_width + (out_height - 1)   # cells + row-seps
        T_total   = T_prefix + max_new

        # ── Phase 1: prefill ─────────────────────────────────────────────────
        emb = self._build_embedding(prefix)
        h   = emb
        k_bufs: list[torch.Tensor] = []
        v_bufs: list[torch.Tensor] = []
        for block in self.blocks:
            h, k, v = block.forward_and_cache(h, pad_mask)
            k_buf = torch.empty(B, H, T_total, hd, device=device, dtype=k.dtype)
            v_buf = torch.empty(B, H, T_total, hd, device=device, dtype=v.dtype)
            k_buf[:, :, :T_prefix, :].copy_(k)
            v_buf[:, :, :T_prefix, :].copy_(v)
            k_bufs.append(k_buf)
            v_bufs.append(v_buf)

        current_logits = self.head(self.ln_final(h[:, -1:, :]))[:, 0, :]

        # ── Phase 2: decode ───────────────────────────────────────────────────
        generated: list[int] = []
        prev_color = -1
        fill_pos   = T_prefix   # next slot to write in the K/V buffers

        def _step(feat: torch.Tensor) -> torch.Tensor:
            """One decode step: write K/V at fill_pos, return next-token logit."""
            nonlocal fill_pos
            h = self._build_embedding_at_pos(feat, fill_pos)
            for i, block in enumerate(self.blocks):
                normed = block.ln1(h)
                qkv    = block.attn.qkv(normed).reshape(
                             B, 1, 3, H, hd).permute(2, 0, 3, 1, 4)
                q, k_new, v_new = qkv[0], qkv[1], qkv[2]
                # In-place write — no allocation, no copy of existing cache
                k_bufs[i][:, :, fill_pos:fill_pos + 1, :] = k_new
                v_bufs[i][:, :, fill_pos:fill_pos + 1, :] = v_new
                # Contiguous view of filled portion — no copy
                k_act = k_bufs[i][:, :, :fill_pos + 1, :]
                v_act = v_bufs[i][:, :, :fill_pos + 1, :]
                attn  = F.scaled_dot_product_attention(q, k_act, v_act, is_causal=False)
                attn  = block.attn.out(attn.transpose(1, 2).reshape(B, 1, D))
                h     = h + attn
                h     = h + block.ffn(block.ln2(h))
            fill_pos += 1
            return self.head(self.ln_final(h[:, -1:, :]))[:, 0, :]

        for r in range(out_height):
            for c in range(out_width):
                next_tok = int(current_logits[0].argmax().item())
                next_tok = max(0, min(9, next_tok))
                generated.append(next_tok)
                change     = 1 if (next_tok != prev_color and prev_color >= 0) else 0
                prev_color = next_tok
                feat = torch.tensor(
                    [[[next_tok, c + 1, r + 1, change, grid_number]]],
                    device=device, dtype=torch.int64)
                current_logits = _step(feat)

            if r < out_height - 1:
                sep = torch.tensor(
                    [[[ROW_SEP, 0, 0, 0, grid_number]]],
                    device=device, dtype=torch.int64)
                current_logits = _step(sep)

        return np.array(generated, dtype=np.uint8).reshape(out_height, out_width)
