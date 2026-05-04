#!/usr/bin/env python3
"""
√Block Attention — Pathfinder-X 长距离依赖实验（单文件版）

任务：128×128 图像 → 16384 像素序列
判断两个标记点是否在同一路径上（二分类）

用法：
    python pathfinder_experiment.py                    # 完整实验
    python pathfinder_experiment.py --quick            # 快速验证
    python pathfinder_experiment.py --sampling topk_norm  # 指定采样策略
    python pathfinder_experiment.py --eval              # 仅评估已有模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import numpy as np
import math
import time
import os
import sys
import argparse
import re
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════
#  Part 1: √Block 稀疏注意力
# ═══════════════════════════════════════════════════════════════════════════


def create_blocks(seq_len, growth_factor=2.0, min_block_size=1):
    """从序列末尾开始，块大小指数增长（近小远大）。"""
    blocks, pos, bid = [], seq_len - 1, 0
    while pos >= 0:
        bsize = max(min_block_size, int(min_block_size * (growth_factor ** bid)))
        start = max(0, pos - bsize + 1)
        blocks.append((start, pos + 1))
        pos = start - 1
        bid += 1
    blocks.reverse()
    return blocks


def uniform_sampling(block_start, block_end, k, device='cpu'):
    """块内均匀采样 k 个位置，含随机偏移。"""
    b_len = block_end - block_start
    if b_len <= k:
        return torch.arange(block_start, block_end, device=device)
    step = b_len / k
    offset = torch.randint(0, max(1, int(step)), (1,), device=device).item()
    indices = (torch.arange(k, device=device) * step + offset).long()
    return torch.clamp(indices, 0, b_len - 1) + block_start


def topk_norm_sampling(k_vals, block_start, block_end, k, device='cpu'):
    """
    按 Key 向量的 L2 范数取 Top-K。
    k_vals: (B, H, seq_len, head_dim) 或 (seq_len,)
    """
    block_norms = k_vals[:, :, block_start:block_end, :].norm(dim=-1)  # (B,H,blk)
    avg_norms = block_norms.mean(dim=(0, 1))  # (blk,)
    k_actual = min(k, avg_norms.size(0))
    _, top_idx = torch.topk(avg_norms, k=k_actual)
    return top_idx + block_start


class SparseAttention(nn.Module):
    """√Block 稀疏注意力层（支持 uniform / topk_norm / full 三种模式）"""

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
        """获取所有块的代表索引。"""
        blocks = create_blocks(seq_len, self.growth_factor)
        all_idx = []
        for start, end in blocks:
            b_len = end - start
            k = max(1, int(math.sqrt(b_len)))
            if self.sampling == 'uniform':
                idx = uniform_sampling(start, end, k, device)
            elif self.sampling == 'topk_norm':
                idx = topk_norm_sampling(k_tensor, start, end, k, device)
            all_idx.append(idx)
        indices = torch.cat(all_idx).unique()
        last = torch.tensor([seq_len - 1], device=device)
        return torch.cat([indices, last]).unique()

    def forward(self, x):
        B, T, C = x.shape
        Q = self.W_q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # 全注意力模式
        if self.sampling == 'full':
            out = F.scaled_dot_product_attention(Q, K, V)
            out = out.transpose(1, 2).contiguous().view(B, T, C)
            return self.out_proj(out)

        # √Block: 选代表
        indices = self._get_rep_indices(T, K, x.device)
        K_rep, V_rep = K[:, :, indices, :], V[:, :, indices, :]
        attn = torch.matmul(Q, K_rep.transpose(-2, -1)) / math.sqrt(self.head_dim)
        out = torch.matmul(F.softmax(attn, dim=-1), V_rep)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, growth_factor, sampling, dropout=0.1):
        super().__init__()
        self.attn = SparseAttention(d_model, n_heads, growth_factor, sampling)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(),
            nn.Linear(4 * d_model, d_model), nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x


class PathfinderModel(nn.Module):
    """Pathfinder-X 分类模型：√Block Transformer + 标记点位置提取 + 分类头

    关键改进：用标记点的像素值 (0.75, 0.5) 在序列中定位两个点，
    提取对应位置的隐状态做分类，而不是均值池化。
    """

    def __init__(self, d_model=128, n_heads=4, n_layers=2,
                 growth_factor=2.0, sampling='uniform',
                 seq_len=16384, num_classes=2):
        super().__init__()
        self.seq_len = seq_len
        self.pixel_emb = nn.Linear(1, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.dropout = nn.Dropout(0.1)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, growth_factor, sampling, 0.1)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        # 分类头：输入是 A 和 B 标记点的拼接隐状态
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.GELU(),
            nn.Dropout(0.1), nn.Linear(d_model, num_classes),
        )

    def forward(self, x, pos1=None, pos2=None):
        """
        x: (B, T) 像素值
        pos1, pos2: (B,) 两个标记点的序列位置
        """
        B, T = x.shape[0], x.shape[1]
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.dropout(self.pixel_emb(x) + self.pos_emb(pos))
        for block in self.blocks:
            h = block(h)
        h = self.ln_f(h)

        if pos1 is not None and pos2 is not None:
            # 提取标记点位置的隐状态
            idx = torch.arange(B, device=x.device)
            h_a = h[idx, pos1, :]  # (B, d_model)
            h_b = h[idx, pos2, :]
            pooled = torch.cat([h_a, h_b], dim=-1)  # (B, 2*d_model)
        else:
            # 回退到均值池化
            pooled = torch.cat([h.mean(dim=1), h.mean(dim=1)], dim=-1)
        return self.classifier(pooled)


# ═══════════════════════════════════════════════════════════════════════════
#  Part 2: Pathfinder-X 数据生成
# ═══════════════════════════════════════════════════════════════════════════

IMG_SIZE = 128
MARKER_VALUES = {1: 0.75, 2: 0.5}


def _random_walk_path(size=128, length=None):
    if length is None:
        length = np.random.randint(size // 2, size * 2)
    x, y = np.random.randint(size // 3, 2 * size // 3, size=2)
    path, steps = {(x, y)}, 0
    while len(path) < length and steps < length * 10:
        steps += 1
        dx, dy = np.random.choice([-1, 0, 1], size=2)
        if dx == 0 and dy == 0:
            continue
        nx, ny = x + dx, y + dy
        if 0 <= nx < size and 0 <= ny < size:
            x, y = nx, ny
            path.add((x, y))
    return path


def _curved_path(size=128):
    n_ctrl = np.random.randint(3, 7)
    cx = np.sort(np.random.randint(0, size, n_ctrl))
    cy = np.random.randint(0, size, n_ctrl)
    t = np.linspace(0, 1, np.random.randint(size // 2, size * 2))
    x = np.zeros_like(t)
    y = np.zeros_like(t)
    for i in range(n_ctrl):
        coeff = np.prod([(t - j / (n_ctrl - 1)) for j in range(n_ctrl) if j != i], axis=0)
        denom = np.prod([(i / (n_ctrl - 1) - j / (n_ctrl - 1)) for j in range(n_ctrl) if j != i])
        x += cx[i] * coeff / denom
        y += cy[i] * coeff / denom
    x = np.clip(np.round(x).astype(int), 0, size - 1)
    y = np.clip(np.round(y).astype(int), 0, size - 1)
    return set(zip(x, y))


def generate_sample(size=128):
    """生成一张 Pathfinder 图像，返回 (flat_image, label, pos1, pos2)。"""
    path = _curved_path(size) if np.random.random() > 0.3 else _random_walk_path(size)
    img = np.zeros((size, size), dtype=np.float32)
    for x, y in path:
        img[y, x] = 1.0
    path_list = list(path)
    same = np.random.random() > 0.5
    if same and len(path_list) >= 2:
        idx = np.random.choice(len(path_list), 2, replace=False)
        p1, p2 = path_list[idx[0]], path_list[idx[1]]
    else:
        p1 = path_list[np.random.randint(len(path_list))]
        attempts = 0
        p2 = p1
        while (p2 == p1 or (p2[0], p2[1]) in path) and attempts < 100:
            px, py = np.random.randint(0, size), np.random.randint(0, size)
            p2 = (px, py)
            attempts += 1
        same = (p2 in path)
    img[p1[1], p1[0]] = MARKER_VALUES[1]
    img[p2[1], p2[0]] = MARKER_VALUES[2]
    # 返回展平图像、标签、以及两个标记点的字节位置
    pos1_flat = p1[1] * size + p1[0]
    pos2_flat = p2[1] * size + p2[0]
    return img.flatten(), 1 if same else 0, pos1_flat, pos2_flat


class PathfinderDataset(Dataset):
    def __init__(self, num_samples=5000, cache=True):
        self.num_samples = num_samples
        self.cache = cache
        if cache:
            self.data = [generate_sample() for _ in range(num_samples)]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.cache:
            seq, label, p1, p2 = self.data[idx]
        else:
            seq, label, p1, p2 = generate_sample()
        return (torch.tensor(seq, dtype=torch.float32),
                torch.tensor(p1, dtype=torch.long),
                torch.tensor(p2, dtype=torch.long),
                torch.tensor(label, dtype=torch.long))


# ═══════════════════════════════════════════════════════════════════════════
#  Part 3: 训练 + 评估
# ═══════════════════════════════════════════════════════════════════════════

def get_scheduler(optimizer, warmup_steps, total_steps):
    warmup = LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])


def train_epoch(model, loader, opt, sched, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch in loader:
        if len(batch) == 4:
            x, p1, p2, y = batch
            p1, p2 = p1.to(device), p2.to(device)
        else:
            x, y = batch
            p1, p2 = None, None
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x, pos1=p1, pos2=p2)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        if sched:
            sched.step()
        total_loss += loss.item()
        correct += (logits.argmax(-1) == y).sum().item()
        total += y.size(0)
    return total_loss / len(loader), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, device, profile=False):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    times, mems = [], []
    for batch in loader:
        if len(batch) == 4:
            x, p1, p2, y = batch
            p1, p2 = p1.to(device), p2.to(device)
        else:
            x, y = batch
            p1, p2 = None, None
        x, y = x.to(device), y.to(device)
        if profile and device.type == "mps":
            torch.mps.synchronize()
            t0 = time.perf_counter()
        logits = model(x, pos1=p1, pos2=p2)
        if profile and device.type == "mps":
            torch.mps.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
            mems.append(torch.mps.current_allocated_memory() / 1024**2)
        loss = F.cross_entropy(logits, y)
        total_loss += loss.item()
        correct += (logits.argmax(-1) == y).sum().item()
        total += y.size(0)
    acc = correct / max(total, 1)
    avg_time = np.mean(times) if times else 0
    avg_mem = np.mean(mems) if mems else 0
    return total_loss / len(loader), acc, avg_time, avg_mem


def clean_log_output(text):
    """清理 tqdm 的 \\r 噪声，保留干净输出。"""
    segments = text.split('\r')
    keep = []
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        clean = re.sub(r'\033\[[0-9;]*m', '', seg)
        if any(p in clean for p in [
            '=== Epoch', 'Train Loss:', 'Val', 'Best', 'Using',
            'Model', 'Parameters', 'Sampling', 'Data', 'Seq',
            'Training complete', '=== Training',
        ]):
            keep.append(clean)
        elif 'it/s' in clean:
            # Keep only final tqdm state: shorten to loss/speed
            m = re.search(r'(Training|Evaluating).*?(loss=[\d.]+).*?([\d.]+it/s)', clean)
            if m:
                keep.append(f"  {m.group(1)}: {m.group(2)}, {m.group(3)}")
    return '\n'.join(keep) + '\n'


# ═══════════════════════════════════════════════════════════════════════════
#  Part 4: Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="√Block Pathfinder-X 实验（单文件版）")
    parser.add_argument('--sampling', default='uniform', choices=['uniform', 'topk_norm', 'full'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--num_train', type=int, default=5000)
    parser.add_argument('--num_val', type=int, default=1000)
    parser.add_argument('--seq_len', type=int, default=16384)
    parser.add_argument('--quick', action='store_true', help='快速模式: 2 epochs, 4096 seq')
    parser.add_argument('--eval', action='store_true', help='仅评估已有模型')
    args = parser.parse_args()

    if args.quick:
        args.epochs = 2
        args.num_train = 500
        args.num_val = 200
        args.seq_len = 4096

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Sampling: {args.sampling}")
    print(f"Model: d_model={args.d_model}, heads={args.n_heads}, layers={args.n_layers}")
    print(f"Seq len: {args.seq_len}, Batch: {args.batch_size}")
    print(f"Data: train={args.num_train}, val={args.num_val}")
    print(f"Epochs: {args.epochs}")

    # 数据
    print("\nGenerating data...")
    train_ds = PathfinderDataset(num_samples=args.num_train)
    val_ds = PathfinderDataset(num_samples=args.num_val)
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False)

    # 模型
    ckpt_name = f"pathfinder_{args.sampling}_best.pt"

    if args.eval:
        model = PathfinderModel(
            d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
            sampling=args.sampling, seq_len=args.seq_len,
        ).to(device)
        model.load_state_dict(torch.load(ckpt_name, map_location=device))
        _, val_acc, inf_time, mem = evaluate(model, val_loader, device, profile=True)
        print(f"\n{'='*50}")
        print(f"  {args.sampling} — Accuracy: {val_acc*100:.2f}%")
        print(f"  Inference: {inf_time:.2f} ms | Memory: {mem:.1f} MB")
        print(f"  Random baseline: 50.00%")
        return

    model = PathfinderModel(
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        sampling=args.sampling, seq_len=args.seq_len,
    ).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    # 内存对比
    R = int(math.sqrt(args.seq_len)) * 2  # rough estimate
    dense_mem = args.seq_len ** 2 * 2 / 1024**3
    block_mem = args.seq_len * R * 2 / 1024**3
    print(f"Memory estimate — Dense: {dense_mem:.2f} GB, √Block: {block_mem:.3f} GB")
    print(f"Representatives: ~{R} of {args.seq_len}\n")

    # 训练
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    steps = len(train_loader) * args.epochs
    sched = get_scheduler(opt, min(200, steps // 10), steps)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        train_loss, train_acc = train_epoch(model, train_loader, opt, sched, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, device)
        print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), ckpt_name)
            print(f"  ★ New best! Saved to {ckpt_name}")

    # 最终结果
    print(f"\n{'='*50}")
    print(f"  Best Val Accuracy ({args.sampling}): {best_acc*100:.2f}%")
    print(f"  Random baseline: 50.00%")
    print(f"  {'='*50}")

    # 如果跑两种采样，做对比
    if args.sampling != 'topk_norm':
        print(f"\n  Also try: python {sys.argv[0]} --sampling topk_norm{' --quick' if args.quick else ''}")
    print(f"\n  Evaluate: python {sys.argv[0]} --eval{' --quick' if args.quick else ''}")


if __name__ == "__main__":
    main()
