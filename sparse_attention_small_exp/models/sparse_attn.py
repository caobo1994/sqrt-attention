import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple

def create_blocks(seq_len: int, growth_factor: float = 2.0, min_block_size: int = 1) -> List[Tuple[int, int]]:
    """
    生成“近细远粗”的块边界列表。
    每个块表示为 (start, end) 左闭右开区间。
    """
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


def uniform_sampling(x: torch.Tensor, block_start: int, block_end: int, k: int) -> torch.Tensor:
    """
    块内均匀采样 k 个位置。
    """
    seq_len = block_end - block_start
    if seq_len <= k:
        indices = torch.arange(block_start, block_end, device=x.device)
    else:
        step = seq_len / k
        offset = torch.randint(0, max(1, int(step)), (1,), device=x.device).item()
        indices = (torch.arange(k, device=x.device) * step + offset).long()
        indices = torch.clamp(indices, 0, seq_len - 1) + block_start
    return indices


def topk_norm_sampling(x: torch.Tensor, block_start: int, block_end: int, k: int, use_key_norm: bool = True) -> torch.Tensor:
    """
    按 Key 向量的 L2 范数取前 k 个 token。
    """
    block_keys = x[:, :, block_start:block_end, :]         # (B, H, L_blk, d)
    norms = block_keys.norm(dim=-1)                         # (B, H, L_blk)
    avg_norms = norms.mean(dim=(0, 1))                      # (L_blk,)
    k_actual = min(k, avg_norms.size(0))
    _, top_indices = torch.topk(avg_norms, k=k_actual)
    global_indices = top_indices + block_start
    return global_indices


class SparseAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, growth_factor: float = 2.0, sampling: str = 'uniform'):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.growth_factor = growth_factor
        self.sampling = sampling  # 'uniform', 'topk_norm', or 'full'

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # 投影并分割多头
        Q = self.W_q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, T, d)
        K = self.W_k(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        scale = math.sqrt(self.head_dim)

        # ---------- 全注意力模式 ----------
        if self.sampling == 'full':
            attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / scale
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_output = torch.matmul(attn_weights, V)
            attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
            return self.out_proj(attn_output)

        # ---------- 稀疏注意力（均匀或 Top‑K 范数） ----------
        blocks = create_blocks(T, growth_factor=self.growth_factor)

        all_indices = []
        for (start, end) in blocks:
            block_len = end - start
            k = max(1, int(math.sqrt(block_len)))
            if self.sampling == 'uniform':
                idx = uniform_sampling(K, start, end, k)
            elif self.sampling == 'topk_norm':
                idx = topk_norm_sampling(K, start, end, k, use_key_norm=True)
            else:
                raise ValueError(f"Unknown sampling method: {self.sampling}")
            all_indices.append(idx)

        all_indices = torch.cat(all_indices).unique()
        last_token = torch.tensor([T - 1], device=x.device)
        all_indices = torch.cat([all_indices, last_token]).unique()

        K_rep = K[:, :, all_indices, :]
        V_rep = V[:, :, all_indices, :]

        attn_scores = torch.matmul(Q, K_rep.transpose(-2, -1)) / scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V_rep)

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(attn_output)
