#!/usr/bin/env python3
"""
Train √Block Pathfinder model on Pathfinder-X (16K length).

Usage:
    python train.py                              # Default: √Block uniform, 10 epochs
    python train.py --sampling full              # Dense baseline (small seq only!)
    python train.py --sampling topk_norm         # TopK-Norm sampling
    python train.py --quick                      # Quick test: 2 epochs, 1000 samples
    python train.py --task copying               # Copying task
"""

import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.lra_model import PathfinderModel
from utils import get_dataloaders, CopyingDataset


def get_scheduler(optimizer, warmup_steps, total_steps):
    warmup = LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])


def train_one_epoch(model, dataloader, optimizer, scheduler, device, task='pathfinder'):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        if task == 'pathfinder':
            if len(batch) == 4:
                x, p1, p2, y = batch
                p1, p2 = p1.to(device), p2.to(device)
            else:
                x, y = batch
                p1, p2 = None, None
            x, y = x.to(device), y.to(device)
        else:  # copying
            x, y = batch
            x, y = x.to(device), y.to(device)
            p1, p2 = None, None

        optimizer.zero_grad()

        if task == 'pathfinder':
            logits = model(x, pos1=p1, pos2=p2)
            loss = nn.functional.cross_entropy(logits, y)
            preds = logits.argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += y.size(0)
        else:  # copying
            logits = model(x, return_logits=True)
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()
        total_loss += loss.item()

        if task == 'pathfinder':
            acc = correct / max(total, 1)
            pbar.set_postfix(loss=loss.item(), acc=f"{acc:.3f}")
        else:
            pbar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    if task == 'pathfinder':
        return avg_loss, correct / max(total, 1)
    return avg_loss, 0.0


@torch.no_grad()
def evaluate(model, dataloader, device, task='pathfinder'):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in tqdm(dataloader, desc="Evaluating"):
        if task == 'pathfinder':
            x, y = batch
            x, y = x.to(device), y.to(device)
            logits = model(x, pos1=p1, pos2=p2)
            loss = nn.functional.cross_entropy(logits, y)
            preds = logits.argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += y.size(0)
        else:  # copying
            x, y = batch
            x, y = x.to(device), y.to(device)
            logits = model(x, return_logits=True)
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            preds = logits.argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += y.numel()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description="Train √Block on Pathfinder-X")
    parser.add_argument('--sampling', type=str, default='uniform',
                        choices=['uniform', 'topk_norm', 'full'])
    parser.add_argument('--task', type=str, default='pathfinder',
                        choices=['pathfinder', 'copying'])
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--warmup_steps', type=int, default=200)
    parser.add_argument('--seq_len', type=int, default=16384)
    parser.add_argument('--growth_factor', type=float, default=2.0)
    parser.add_argument('--num_train', type=int, default=5000)
    parser.add_argument('--num_val', type=int, default=1000)
    parser.add_argument('--quick', action='store_true', help='Quick test mode')
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    # Quick mode overrides
    if args.quick:
        args.epochs = 2
        args.num_train = 500
        args.num_val = 200
        args.seq_len = 4096  # shorter for quick test

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Task: {args.task}")
    print(f"Sampling: {args.sampling}")
    print(f"Model: d_model={args.d_model}, heads={args.n_heads}, layers={args.n_layers}")
    print(f"Seq len: {args.seq_len}, Batch: {args.batch_size}")
    print(f"Data: train={args.num_train}, val={args.num_val}")

    # Model
    if args.task == 'pathfinder':
        model = PathfinderModel(
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            growth_factor=args.growth_factor,
            sampling=args.sampling,
            seq_len=args.seq_len,
            num_classes=2,
        ).to(device)

        train_loader, val_loader = get_dataloaders(
            batch_size=args.batch_size,
            num_train=args.num_train,
            num_val=args.num_val,
        )
    else:  # copying task
        # For copying, use a decoder-style model
        from models.lra_model import PathfinderModel as CopyModel
        model = CopyModel(
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            growth_factor=args.growth_factor,
            sampling=args.sampling,
            seq_len=args.seq_len,
            num_classes=11,  # vocab=8 + delim + blank + end = 11
        ).to(device)

        dataset = CopyingDataset(num_samples=args.num_train, seq_len=args.seq_len)
        val_dataset = CopyingDataset(num_samples=args.num_val, seq_len=args.seq_len)
        from torch.utils.data import DataLoader
        train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    if args.resume:
        print(f"Resuming from {args.resume}")
        model.load_state_dict(torch.load(args.resume, map_location=device))

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    scheduler = get_scheduler(optimizer, args.warmup_steps, total_steps)

    best_acc = 0.0
    ckpt_name = f"pathfinder_{args.sampling}_best.pt"

    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, args.task
        )
        val_loss, val_acc = evaluate(model, val_loader, device, args.task)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f}, Val   Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), ckpt_name)
            print(f"New best model saved (Acc: {val_acc:.4f})")

    print(f"\n=== Training Complete ===")
    print(f"Best Val Accuracy ({args.sampling}): {best_acc:.4f} ({best_acc * 100:.2f}%)")

    # Quick density comparison
    if args.task == 'pathfinder' and args.sampling == 'uniform':
        R_est = int(args.seq_len ** 0.5) * int(args.growth_factor ** 0.5)
        dense_mem = args.seq_len ** 2 * 2 / 1024**3  # GB for fp16
        block_mem = (args.seq_len * R_est * 2) / 1024**3
        print(f"\nMemory comparison (est.):")
        print(f"  Dense attention: {dense_mem:.2f} GB")
        print(f"  √Block attention: {block_mem:.2f} GB")
        print(f"  Representatives: ~{R_est} (of {args.seq_len})")


if __name__ == "__main__":
    main()
