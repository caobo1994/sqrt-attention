"""
√Block with Alternating Sampling — 层间交替采样实验

核心问题：不同层的稀疏注意力应该用相同的采样策略吗？
  
假设：
  - 浅层（底部）：Uniform 采样更好（覆盖面广，提取基础特征）
  - 深层（顶部）：TopK-Norm 采样更好（聚焦重要 token，精细化）

实验方案：对比三种配置
  1. Uniform-Only:  所有层使用 uniform 采样
  2. TopK-Only:     所有层使用 topk_norm 采样
  3. Alternating:   层间交替 uniform / topk_norm
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def create_blocks(seq_len, growth_factor=2.0, min_block_size=1):
    """生成指数增长的块边界（近小远大）。"""
    blocks, pos, bid = [], seq_len - 1, 0
    while pos >= 0:
        bsize = max(min_block_size, int(min_block_size * (growth_factor ** bid)))
        start = max(0, pos - bsize + 1)
        blocks.append((start, pos + 1))
        pos = start - 1
        bid += 1
    blocks.reverse()
    return blocks


def uniform_sampling(x, block_start, block_end, k):
    """块内均匀采样 k 个位置，含随机偏移。"""
    b_len = block_end - block_start
    if b_len <= k:
        return torch.arange(block_start, block_end, device=x.device)
    step = b_len / k
    offset = torch.randint(0, max(1, int(step)), (1,), device=x.device).item()
    indices = (torch.arange(k, device=x.device) * step + offset).long()
    return torch.clamp(indices, 0, b_len - 1) + block_start


def topk_norm_sampling(x, block_start, block_end, k):
    """按 Key 向量的 L2 范数取 Top-K。"""
    block_keys = x[:, :, block_start:block_end, :]
    norms = block_keys.norm(dim=-1).mean(dim=(0, 1))
    k_actual = min(k, norms.size(0))
    _, top_idx = torch.topk(norms, k=k_actual)
    return top_idx + block_start


class SparseAttention(nn.Module):
    """√Block 稀疏注意力（支持 uniform / topk_norm 单模式）。"""

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

    def _get_rep_indices(self, seq_len, k_tensor, device):
        blocks = create_blocks(seq_len, self.growth_factor)
        all_idx = []
        for start, end in blocks:
            b_len = end - start
            k = max(1, int(math.sqrt(b_len)))
            if self.sampling == 'uniform':
                idx = uniform_sampling(k_tensor, start, end, k)
            elif self.sampling == 'topk_norm':
                idx = topk_norm_sampling(k_tensor, start, end, k)
            all_idx.append(idx)
        indices = torch.cat(all_idx).unique()
        last = torch.tensor([seq_len - 1], device=device)
        return torch.cat([indices, last]).unique()

    def forward(self, x):
        B, T, C = x.shape
        Q = self.W_q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        if self.sampling == 'full':
            out = F.scaled_dot_product_attention(Q, K, V)
            out = out.transpose(1, 2).contiguous().view(B, T, C)
            return self.out_proj(out)

        indices = self._get_rep_indices(T, K, x.device)
        K_rep, V_rep = K[:, :, indices, :], V[:, :, indices, :]
        scale = math.sqrt(self.head_dim)
        out = torch.matmul(F.softmax(torch.matmul(Q, K_rep.transpose(-2, -1)) / scale, dim=-1), V_rep)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x):
        return self.net(x)


class AlternatingGPT(nn.Module):
    """
    GPT 解码器，支持每层独立指定采样策略。

    sampling_config:
        - 'uniform':    所有层 uniform
        - 'topk_norm':  所有层 topk_norm
        - 'alternate':  层间交替（even=uniform, odd=topk_norm）
        - 'alternate_rev': 交替（even=topk_norm, odd=uniform）
        - [str]:        每层策略列表，如 ['uniform','topk_norm','uniform',...]
    """

    def __init__(self, vocab_size, d_model=256, n_heads=8, n_layers=6,
                 growth_factor=2.0, sampling_config='uniform', max_seq_len=1024):
        super().__init__()

        # 将 sampling_config 解析为每层的采样策略
        if isinstance(sampling_config, list):
            layer_samplings = sampling_config
        elif sampling_config == 'uniform':
            layer_samplings = ['uniform'] * n_layers
        elif sampling_config == 'topk_norm':
            layer_samplings = ['topk_norm'] * n_layers
        elif sampling_config == 'alternate':
            layer_samplings = ['uniform' if i % 2 == 0 else 'topk_norm' for i in range(n_layers)]
        elif sampling_config == 'alternate_rev':
            layer_samplings = ['topk_norm' if i % 2 == 0 else 'uniform' for i in range(n_layers)]
        else:
            raise ValueError(f"Unknown sampling config: {sampling_config}")

        self.sampling_config = layer_samplings

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        self.blocks = nn.ModuleList([
            nn.ModuleList([
                SparseAttention(d_model, n_heads, growth_factor, s),
                nn.LayerNorm(d_model),
                FeedForward(d_model, 4 * d_model),
                nn.LayerNorm(d_model),
            ])
            for s in layer_samplings
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids):
        B, T = input_ids.shape
        positions = torch.arange(0, T, device=input_ids.device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        for attn, ln1, ffn, ln2 in self.blocks:
            x = x + attn(ln1(x))
            x = x + ffn(ln2(x))
        x = self.ln_f(x)
        return self.head(x)

    def get_config_str(self):
        """返回可读的配置字符串。"""
        parts = []
        for i, s in enumerate(self.sampling_config):
            short = 'U' if s == 'uniform' else 'T'
            parts.append(f"L{i}={short}")
        return '[' + ', '.join(parts) + ']'
