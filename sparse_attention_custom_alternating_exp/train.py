#!/usr/bin/env python3
"""
训练任意交错采样配置。

用法：
    python train.py --config UTUTUT          # U/T 交替，6 层
    python train.py --config UUUTTT --epochs 5  # 前 3 层 uniform，后 3 层 topk
    python train.py --config UTUTUT --resume   # 从已有检查点恢复

特性：
  - 检查点按 config 字符串命名：ckpt_UTUTUT.pt
  - 自动检测已有检查点，支持恢复训练
  - train_log.json 记录所有已完成的训练历史
"""

import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import math
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.custom_gpt import CustomAlternatingGPT, parse_config_string
from transformers import AutoTokenizer
from datasets import load_dataset


class WikiTextDataset(Dataset):
    def __init__(self, split, tokenizer, seq_len=256):
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        tokenized = dataset.map(
            lambda x: tokenizer(x["text"], truncation=True, max_length=None, return_overflowing_tokens=True),
            batched=True, remove_columns=dataset.column_names
        )
        all_ids = []
        for sample in tokenized:
            all_ids.extend(sample["input_ids"])
        self.data = torch.tensor(all_ids, dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.data) - 1) // self.seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        chunk = self.data[start:end]
        return chunk[:-1], chunk[1:]


def get_dataloaders(tokenizer, seq_len=256, batch_size=4):
    train_ds = WikiTextDataset("train", tokenizer, seq_len)
    val_ds = WikiTextDataset("validation", tokenizer, seq_len)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
    )


def get_scheduler(optimizer, warmup_steps, total_steps):
    warmup = LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])


def get_checkpoint_info(config_str):
    """查找已有检查点。返回 (ckpt_path, log_entry) 或 (None, None)。"""
    ckpt_path = Path(f"ckpt_{config_str}.pt")
    log_path = Path("train_log.json")

    # Load log
    log = {}
    if log_path.exists():
        log = json.loads(log_path.read_text())

    prev_entry = log.get(config_str)
    if prev_entry and ckpt_path.exists():
        return ckpt_path, prev_entry
    return None, None


def save_checkpoint(model, optimizer, epoch, config_str, best_ppl, path=None):
    """保存检查点。"""
    path = path or Path(f"ckpt_{config_str}.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'best_ppl': best_ppl,
        'config_str': config_str,
    }, path)


def update_log(config_str, epochs_completed, best_ppl, best_epoch):
    """更新训练日志。"""
    log_path = Path("train_log.json")
    log = {}
    if log_path.exists():
        log = json.loads(log_path.read_text())
    log[config_str] = {
        'epochs_completed': epochs_completed,
        'best_ppl': best_ppl,
        'best_epoch': best_epoch,
    }
    log_path.write_text(json.dumps(log, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Train √Block with custom U/T config")
    parser.add_argument('--config', type=str, default='UTUTUT', help='U/T 配置字符串，如 UTUTUT')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--seq_len', type=int, default=256)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--n_heads', type=int, default=8)
    # n_layers inferred from config string length
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--resume', action='store_true', help='从已有检查点恢复')
    parser.add_argument('--quick', action='store_true', help='2 epoch 快速测试')
    args = parser.parse_args()

    config_str = args.config.upper().strip()
    if args.quick:
        args.epochs = 2

    # 解析配置
    layer_samplings = parse_config_string(config_str, args.n_layers)
    readable = ''.join('U' if s == 'uniform' else 'T' for s in layer_samplings)
    print(f"Config: {config_str} → {readable} ({args.n_layers} layers)")
    print(f"Layers: ['{' | '.join(readable)}']")

    # 检查已有检查点
    ckpt_path, prev_entry = get_checkpoint_info(config_str)
    start_epoch = 1
    best_ppl = float('inf')

    if ckpt_path and args.resume:
        print(f"  Found existing checkpoint: {ckpt_path}")
        if prev_entry:
            print(f"  Previously trained: {prev_entry['epochs_completed']} epochs, best PPL={prev_entry['best_ppl']:.2f}")
            start_epoch = prev_entry['epochs_completed'] + 1
            best_ppl = prev_entry['best_ppl']
    elif ckpt_path:
        print(f"  Found checkpoint {ckpt_path} (use --resume to continue)")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model = CustomAlternatingGPT(
        vocab_size=len(tokenizer),
        d_model=args.d_model,
        n_heads=args.n_heads,
        # n_layers inferred from config_str
        config_str=config_str,
        max_seq_len=args.seq_len,
    ).to(device)

    # 加载已有权重
    if ckpt_path and args.resume:
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Resumed from epoch {checkpoint.get('epoch', 0)}")
        if 'best_ppl' in checkpoint:
            best_ppl = checkpoint['best_ppl']

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    train_loader, val_loader = get_dataloaders(tokenizer, args.seq_len, args.batch_size)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    # 恢复优化器状态
    if ckpt_path and args.resume and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    steps = len(train_loader) * args.epochs
    scheduler = get_scheduler(optimizer, args.warmup_steps, steps)

    # 训练
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc="Training")
        for input_ids, targets in pbar:
            input_ids, targets = input_ids.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(input_ids)
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler:
                scheduler.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        # 评估
        model.eval()
        val_loss = 0.0
        for input_ids, targets in tqdm(val_loader, desc="Evaluating"):
            input_ids, targets = input_ids.to(device), targets.to(device)
            with torch.no_grad():
                logits = model(input_ids)
                loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            val_loss += loss.item()
        ppl = math.exp(val_loss / len(val_loader))
        print(f"Train Loss: {total_loss/len(train_loader):.4f}, Val Perplexity: {ppl:.2f}")

        if ppl < best_ppl:
            best_ppl = ppl
            best_epoch = epoch
            save_checkpoint(model, optimizer, epoch, config_str, best_ppl)
            print(f"  ★ New best! PPL={ppl:.2f}")
            update_log(config_str, epoch, best_ppl, best_epoch)

    print(f"\nBest PPL ({config_str}): {best_ppl:.2f}")


if __name__ == "__main__":
    main()
