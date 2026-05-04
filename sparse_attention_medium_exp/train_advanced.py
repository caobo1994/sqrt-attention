#!/usr/bin/env python3
"""
Medium-scale training on WikiText-103 with d_model=512, 8 layers.
Improved training with warmup, cosine annealing, checkpoint resumption.
"""
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
import math
import os

from models.gpt_sparse import SparseGPT
from utils import get_dataloaders
from transformers import AutoTokenizer


def get_scheduler(optimizer, warmup_steps, total_steps):
    """Linear warmup -> cosine annealing"""
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
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
    parser = argparse.ArgumentParser(description="Train √Block GPT on WikiText-103 (medium scale)")
    parser.add_argument('--sampling', type=str, default='uniform', choices=['uniform', 'topk_norm'])
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size (default: 4 for medium model)')
    parser.add_argument('--seq_len', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4, help='Lower LR for larger model')
    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=8)
    parser.add_argument('--growth_factor', type=float, default=2.0)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--patience', type=int, default=0, help='Early stopping patience (0=disabled)')
    parser.add_argument('--log_interval', type=int, default=10, help='Log every N batches')
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Model config: d_model={args.d_model}, n_heads={args.n_heads}, n_layers={args.n_layers}")
    print(f"Dataset: WikiText-103, seq_len={args.seq_len}")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)

    model = SparseGPT(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        growth_factor=args.growth_factor,
        sampling=args.sampling,
        max_seq_len=args.seq_len
    ).to(device)

    if args.resume:
        print(f"Resuming from {args.resume}")
        model.load_state_dict(torch.load(args.resume, map_location=device))

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_loader, val_loader = get_dataloaders(tokenizer, seq_len=args.seq_len, batch_size=args.batch_size)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    scheduler = get_scheduler(optimizer, args.warmup_steps, total_steps)

    best_ppl = float('inf')
    best_epoch = 0
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        ppl = evaluate(model, val_loader, device)
        print(f"Train Loss: {train_loss:.4f}, Val Perplexity: {ppl:.2f}")

        if ppl < best_ppl:
            best_ppl = ppl
            best_epoch = epoch
            no_improve = 0
            save_path = f"sparse_gpt_{args.sampling}_wt103_best.pt"
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved to {save_path} (PPL: {ppl:.2f})")
        else:
            no_improve += 1
            if args.patience > 0 and no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch}, best was epoch {best_epoch} (PPL: {best_ppl:.2f})")
                break

    print(f"\nTraining finished. Best PPL: {best_ppl:.2f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()
