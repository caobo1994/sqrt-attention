"""
√Block with Custom Alternating Sampling — 任意交错采样实验

接收一个由 U 和 T 组成的配置字符串，例如：
  "UUUUUU"  → 全部 uniform
  "TTTTTT"  → 全部 topk_norm
  "UTUTUT"  → uniform/topk 交替
  "TUUTUU"  → 任意自定义模式

自动检测已有检查点，支持从中断处继续训练。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def create_blocks(seq_len, growth_factor=2.0, min_block_size=1):
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
    b_len = block_end - block_start
    if b_len <= k:
        return torch.arange(block_start, block_end, device=x.device)
    step = b_len / k
    offset = torch.randint(0, max(1, int(step)), (1,), device=x.device).item()
    indices = (torch.arange(k, device=x.device) * step + offset).long()
    return torch.clamp(indices, 0, b_len - 1) + block_start


def topk_norm_sampling(x, block_start, block_end, k):
    block_keys = x[:, :, block_start:block_end, :]
    norms = block_keys.norm(dim=-1).mean(dim=(0, 1))
    k_actual = min(k, norms.size(0))
    _, top_idx = torch.topk(norms, k=k_actual)
    return top_idx + block_start


class SparseAttention(nn.Module):
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


def parse_config_string(config_str, n_layers=6):
    """
    将配置字符串解析为每层的采样策略。
    
    例如:
      "UTUTUT" → ['uniform','topk_norm','uniform','topk_norm','uniform','topk_norm']
      "UUUTTT" → ['uniform','uniform','uniform','topk_norm','topk_norm','topk_norm']
      "TTTTTT" → ['topk_norm'] * 6
    """
    config_str = config_str.upper().strip()
    if len(config_str) != n_layers:
        raise ValueError(f"Config string must have exactly {n_layers} chars, got {len(config_str)}: '{config_str}'")
    mapping = {'U': 'uniform', 'T': 'topk_norm'}
    result = []
    for c in config_str:
        if c not in mapping:
            raise ValueError(f"Invalid char '{c}' in config string. Use only U or T.")
        result.append(mapping[c])
    return result


class CustomAlternatingGPT(nn.Module):
    """GPT 解码器，接收 U/T 字符串配置每层的采样策略。"""

    def __init__(self, vocab_size, d_model=256, n_heads=8, n_layers=6,
                 growth_factor=2.0, config_str='UTUTUT', max_seq_len=1024):
        super().__init__()
        self.config_str = config_str.upper()
        self.n_layers = n_layers
        layer_samplings = parse_config_string(config_str, n_layers)

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
