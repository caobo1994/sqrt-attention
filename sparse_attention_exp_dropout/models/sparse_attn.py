import torch
import torch.nn.functional as F
import math

def create_blocks(seq_len, growth_factor=2.0, min_block_size=1):
    """
    生成块边界列表 (start, end)，从后往前划分（近端块小，远端块大）。
    返回: List[(start, end)]，包含最后一个 token 自身。
    """
    blocks = []
    pos = seq_len - 1
    block_id = 0
    while pos >= 0:
        # 块大小按几何增长
        block_size = max(min_block_size, int(min_block_size * (growth_factor ** block_id)))
        start = max(0, pos - block_size + 1)
        blocks.append((start, pos + 1))
        pos = start - 1
        block_id += 1
    blocks.reverse()  # 从远到近 (或保持任意顺序，只要统一)
    return blocks

def uniform_sampling(x, block_start, block_end, k):
    """
    块内均匀采样 k 个位置。x 形状: (batch, heads, seq_len, dim)
    """
    seq_len = block_end - block_start
    if seq_len <= k:
        indices = torch.arange(block_start, block_end, device=x.device)
    else:
        step = seq_len / k
        indices = (torch.arange(k, device=x.device) * step).long()
        # 随机偏移起点以减少偏差
        offset = torch.randint(0, max(1, int(step)), (1,), device=x.device).item()
        indices = (indices + offset).clamp(0, seq_len - 1) + block_start
    return indices

def topk_norm_sampling(x, block_start, block_end, k, use_key_norm=True):
    """
    按 Key 向量的 L2 范数取 Top-K，x 为 key 序列。
    输入 x: (batch, heads, seq_len, head_dim)
    """
    # 取块内 key 向量
    block_keys = x[:, :, block_start:block_end, :]   # (B, H, L_blk, d)
    norms = block_keys.norm(dim=-1)                   # (B, H, L_blk)
    # 在批次和头维度上取平均，得到全局重要性（也可按头独立，这里为简化取平均）
    norms = norms.mean(dim=(0, 1))                    # (L_blk,)
    _, top_indices = torch.topk(norms, k=min(k, norms.size(0)))
    # 映射回全局索引
    global_indices = top_indices + block_start
    return global_indices

class SparseAttention(torch.nn.Module):
    def __init__(self, d_model, n_heads, growth_factor=2.0, sampling='uniform'):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.growth_factor = growth_factor
        self.sampling = sampling  # 'uniform' 或 'topk_norm'

        self.W_q = torch.nn.Linear(d_model, d_model, bias=False)
        self.W_k = torch.nn.Linear(d_model, d_model, bias=False)
        self.W_v = torch.nn.Linear(d_model, d_model, bias=False)
        self.out_proj = torch.nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        # 线性投影并分头
        Q = self.W_q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, T, d)
        K = self.W_k(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # 生成块
        blocks = create_blocks(T, growth_factor=self.growth_factor)

        # 收集每个块的代表索引
        all_indices = []
        for (start, end) in blocks:
            block_len = end - start
            k = max(1, int(math.sqrt(block_len)))  # K = ceil(sqrt(block_size))
            if self.sampling == 'uniform':
                idx = uniform_sampling(K, start, end, k)
            elif self.sampling == 'topk_norm':
                idx = topk_norm_sampling(K, start, end, k, use_key_norm=True)
            else:
                raise ValueError
            all_indices.append(idx)
        # 合并并去重（各块可能有重叠，但去重不影响正确性）
        all_indices = torch.cat(all_indices).unique()
        # 强制包含最后一个 token (当前位置)
        all_indices = torch.cat([all_indices, torch.tensor([T-1], device=x.device)]).unique()

        # 提取代表 token 的 K, V
        K_rep = K[:, :, all_indices, :]   # (B, H, L', d)
        V_rep = V[:, :, all_indices, :]

        # 计算注意力分数 (Q 全部，与代表 K 交互)
        scale = math.sqrt(self.head_dim)
        attn_scores = torch.matmul(Q, K_rep.transpose(-2, -1)) / scale  # (B, H, T, L')
        attn_weights = F.softmax(attn_scores, dim=-1)

        # 聚合 Value
        attn_output = torch.matmul(attn_weights, V_rep)  # (B, H, T, d)

        # 恢复形状
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(attn_output)

