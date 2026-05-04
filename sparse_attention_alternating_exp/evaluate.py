#!/usr/bin/env python3
"""
Evaluate and compare alternating vs uniform vs topk sampling.
"""

import torch
import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.alternating_gpt import AlternatingGPT
from train import get_dataloaders
from transformers import AutoTokenizer


@torch.no_grad()
def evaluate_ppl(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    for input_ids, targets in dataloader:
        input_ids, targets = input_ids.to(device), targets.to(device)
        logits = model(input_ids)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1)
        )
        total_loss += loss.item()
    return math.exp(total_loss / len(dataloader))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--seq_len', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    _, val_loader = get_dataloaders(tokenizer, args.seq_len, args.batch_size)

    configs = [
        ('uniform',     'uniform'),
        ('topk_norm',   'topk_norm'),
        ('alternate',   'alternate'),
        ('alternate_rev', 'alternate_rev'),
    ]

    results = []
    for name, config in configs:
        ckpt = f"alternating_{config}_best.pt"
        print(f"Loading {name} ({config}) from {ckpt}...")

        try:
            model = AlternatingGPT(
                vocab_size=len(tokenizer),
                d_model=args.d_model,
                n_heads=args.n_heads,
                n_layers=args.n_layers,
                sampling_config=config,
                max_seq_len=args.seq_len,
            ).to(device)
            model.load_state_dict(torch.load(ckpt, map_location=device))
            model.eval()

            ppl = evaluate_ppl(model, val_loader, device)
            config_str = model.get_config_str()
            results.append((name, config_str, ppl))
            print(f"  PPL: {ppl:.2f} | Layers: {config_str}")
        except FileNotFoundError:
            print(f"  Checkpoint not found: {ckpt}")
            results.append((name, config, None))

    # Summary
    print(f"\n{'='*50}")
    print(f"  Alternating Sampling — Final Comparison")
    print(f"{'='*50}")
    print(f"\n{'Config':<20} {'Layer Pattern':<40} {'PPL':<10}")
    print(f"{'-'*70}")
    best_ppl = float('inf')
    best_name = ''
    for name, pattern, ppl in results:
        if ppl is not None:
            marker = ' ★' if ppl == min(r[2] for r in results if r[2] is not None) else ''
            print(f"{name:<20} {pattern:<40} {ppl:<10.2f}{marker}")
            if ppl < best_ppl:
                best_ppl = ppl
                best_name = name

    if best_name:
        print(f"\n  ★ Best config: {best_name} (PPL: {best_ppl:.2f})")

    # 交替 vs 纯策略的相对差距
    uniform_ppl = next((r[2] for r in results if r[0] == 'uniform'), None)
    topk_ppl = next((r[2] for r in results if r[0] == 'topk_norm'), None)
    alt_ppl = next((r[2] for r in results if r[0] == 'alternate'), None)

    if uniform_ppl and alt_ppl:
        gap = (alt_ppl - uniform_ppl) / uniform_ppl * 100
        print(f"\n  Alternating vs Uniform: {gap:+.2f}%")
    if topk_ppl and alt_ppl:
        gap = (alt_ppl - topk_ppl) / topk_ppl * 100
        print(f"  Alternating vs TopK-Norm: {gap:+.2f}%")


if __name__ == "__main__":
    main()
