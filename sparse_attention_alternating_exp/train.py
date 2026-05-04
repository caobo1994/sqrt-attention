#!/usr/bin/env python3
"""
Train √Block GPT with alternating layer sampling.

Compares three strategies:
- uniform-all:  all layers uniform
- topk-all:     all layers topk_norm
- alternate:    even=uniform, odd=topk_norm
"""

import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.alternating_gpt import AlternatingGPT
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


def train_one_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    pbar = tqdm(dataloader, desc="Training")
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
    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    for input_ids, targets in tqdm(dataloader, desc="Evaluating"):
        input_ids, targets = input_ids.to(device), targets.to(device)
        logits = model(input_ids)
        loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        total_loss += loss.item()
    return math.exp(total_loss / len(dataloader))


def main():
    parser = argparse.ArgumentParser(description="Train √Block with alternating layer sampling")
    parser.add_argument('--config', type=str, default='uniform',
                        choices=['uniform', 'topk_norm', 'alternate', 'alternate_rev'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--seq_len', type=int, default=256)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()

    if args.quick:
        args.epochs = 2

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Sampling config: {args.config}")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model = AlternatingGPT(
        vocab_size=len(tokenizer),
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        sampling_config=args.config,
        max_seq_len=args.seq_len,
    ).to(device)

    config_str = model.get_config_str()
    print(f"Layer sampling: {config_str}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_loader, val_loader = get_dataloaders(tokenizer, args.seq_len, args.batch_size)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    steps = len(train_loader) * args.epochs
    scheduler = get_scheduler(optimizer, args.warmup_steps, steps)

    best_ppl = float('inf')
    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        ppl = evaluate(model, val_loader, device)
        print(f"Train Loss: {train_loss:.4f}, Val Perplexity: {ppl:.2f}")
        if ppl < best_ppl:
            best_ppl = ppl
            torch.save(model.state_dict(), f"alternating_{args.config}_best.pt")
            print(f"New best! PPL: {ppl:.2f}")

    print(f"\nBest PPL ({args.config}): {best_ppl:.2f}")


if __name__ == "__main__":
    main()
