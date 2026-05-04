import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
import math

from models.gpt_sparse import SparseGPT
from utils import get_dataloaders
from transformers import AutoTokenizer


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc="Training")
    for input_ids, targets in pbar:
        input_ids, targets = input_ids.to(device), targets.to(device)
        optimizer.zero_grad()
        logits = model(input_ids)
        loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())
    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    for input_ids, targets in tqdm(dataloader, desc="Evaluating"):
        input_ids, targets = input_ids.to(device), targets.to(device)
        logits = model(input_ids)
        loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    return math.exp(avg_loss)  # perplexity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sampling', type=str, default='uniform', choices=['uniform', 'topk_norm'])
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--seq_len', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=8)
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Model: d_model={args.d_model}, n_heads={args.n_heads}, n_layers={args.n_layers}")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)

    model = SparseGPT(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        growth_factor=2.0,
        sampling=args.sampling,
        max_seq_len=args.seq_len
    ).to(device)

    train_loader, val_loader = get_dataloaders(tokenizer, seq_len=args.seq_len, batch_size=args.batch_size)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training samples: {len(train_loader)} batches")

    best_ppl = float('inf')
    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        ppl = evaluate(model, val_loader, device)
        print(f"Train Loss: {train_loss:.4f}, Val Perplexity: {ppl:.2f}")

        if ppl < best_ppl:
            best_ppl = ppl
            torch.save(model.state_dict(), f"sparse_gpt_{args.sampling}_wt103_best.pt")

    print(f"\nBest Perplexity ({args.sampling}): {best_ppl:.2f}")


if __name__ == "__main__":
    main()
