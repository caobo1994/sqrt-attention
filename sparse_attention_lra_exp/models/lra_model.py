"""
√Block Attention for Pathfinder-X (LRA).

Key difference from standard attention:
- √Block attention handles 16K length with O(L) memory
- Each block samples √(block_size) representatives
- Total representatives ≈ O(√L) ≈ 126 for L=16384
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def create_blocks(seq_len, growth_factor=2.0, min_block_size=1):
    """Create blocks from end to start with exponential growth."""
    blocks = []
    pos = seq_len - 1
    block_id = 0
    while pos >= 0:
        block_size = max(min_block_size, int(min_block_size * (growth_factor ** block_id)))
        start = max(0, pos - block_size + 1)
        blocks.append((start, pos + 1))
        pos = start - 1
        block_id += 1
    blocks.reverse()
    return blocks


def uniform_sampling(seq_len, block_start, block_end, k, device='cpu'):
    """Uniformly sample k positions within a block."""
    b_len = block_end - block_start
    if b_len <= k:
        return torch.arange(block_start, block_end, device=device)
    step = b_len / k
    offset = torch.randint(0, max(1, int(step)), (1,), device=device).item()
    indices = (torch.arange(k, device=device) * step + offset).long()
    return torch.clamp(indices, 0, b_len - 1) + block_start


class SparseAttention(nn.Module):
    """√Block sparse attention for LRA long sequences."""

    def __init__(self, d_model, n_heads, growth_factor=2.0, sampling='uniform'):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.growth_factor = growth_factor
        self.sampling = sampling

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def _get_representative_indices(self, seq_len, device):
        """Get all representative token indices for all blocks."""
        blocks = create_blocks(seq_len, growth_factor=self.growth_factor)
        all_indices = []
        for start, end in blocks:
            block_len = end - start
            k = max(1, int(math.sqrt(block_len)))
            idx = uniform_sampling(seq_len, start, end, k, device)
            all_indices.append(idx)
        indices = torch.cat(all_indices).unique()
        # Always include last token
        last = torch.tensor([seq_len - 1], device=device)
        return torch.cat([indices, last]).unique()

    def forward(self, x):
        B, T, C = x.shape
        Q = self.W_q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # No block for dense mode
        if self.sampling == 'full':
            out = F.scaled_dot_product_attention(Q, K, V)
            out = out.transpose(1, 2).contiguous().view(B, T, C)
            return self.out_proj(out)

        # √Block: select representatives
        indices = self._get_representative_indices(T, x.device)
        K_rep = K[:, :, indices, :]
        V_rep = V[:, :, indices, :]

        scale = math.sqrt(self.head_dim)
        attn = torch.matmul(Q, K_rep.transpose(-2, -1)) / scale
        attn_w = F.softmax(attn, dim=-1)
        out = torch.matmul(attn_w, V_rep)

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, growth_factor, sampling, dropout=0.1):
        super().__init__()
        self.attn = SparseAttention(d_model, n_heads, growth_factor, sampling)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x


class PathfinderModel(nn.Module):
    """
    Transformer model for Pathfinder-X (16K length).
    
    Uses a lightweight architecture to handle long sequences:
    - d_model=128 or 256 (small to keep memory manageable)
    - 2-4 layers (depth helps, but layers are expensive at 16K)
    - √Block attention for O(L) complexity
    
    Input: (batch, 128*128) flattened grayscale images
    Output: (batch, 2) logits for binary classification
    """

    def __init__(self, d_model=128, n_heads=4, n_layers=2,
                 growth_factor=2.0, sampling='uniform',
                 seq_len=16384, num_classes=2):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model

        # Pixel embedding (1 channel → d_model)
        self.pixel_emb = nn.Linear(1, d_model)

        # Learned positional encoding
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.dropout = nn.Dropout(0.1)

        # Transformer encoder
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, growth_factor, sampling, dropout=0.1)
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)

        # Classification head
        # We'll take the CLS token or mean pool
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x, pos1=None, pos2=None, return_attn=False):
        """
        Args:
            x: (batch, seq_len) or (batch, seq_len, 1) pixel values
            pos1, pos2: (batch,) marker positions for classification
        """
        B, T = x.shape[0], x.shape[1]
        self._pos1 = pos1
        self._pos2 = pos2

        # Ensure shape: (B, T, 1)
        if x.dim() == 2:
            x = x.unsqueeze(-1)

        # Embed pixels + positions
        positions = torch.arange(T, device=x.device).unsqueeze(0)  # (1, T)
        h = self.pixel_emb(x) + self.pos_emb(positions)
        h = self.dropout(h)

        # Transformer blocks
        for block in self.blocks:
            h = block(h)

        h = self.ln_f(h)

        # Global pooling for classification
        # Mean pooling works well for LRA tasks
        pooled = h.mean(dim=1)  # (B, d_model)

        logits = self.classifier(pooled)
        return logits


class DensePathfinderModel(PathfinderModel):
    """
    Dense-attention baseline for Pathfinder-X.
    Same architecture but uses full attention.
    Only feasible for small L (e.g., 1024). Use with --sampling full.
    """

    def __init__(self, **kwargs):
        kwargs['sampling'] = 'full'
        super().__init__(**kwargs)
