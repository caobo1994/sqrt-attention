#!/usr/bin/env python3
"""评估多个 U/T 配置，对比 PPL。"""

import torch
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.custom_gpt import CustomAlternatingGPT
from train import get_dataloaders
from transformers import AutoTokenizer


@torch.no_grad()
def evaluate_ppl(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    for input_ids, targets in dataloader:
        input_ids, targets = input_ids.to(device), targets.to(device)
        logits = model(input_ids)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        total_loss += loss.item()
    return math.exp(total_loss / len(dataloader))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', type=str, default='UTUTUT', help='逗号分隔的配置')
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--seq_len', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()

    configs = args.configs.split(',')
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    _, val_loader = get_dataloaders(tokenizer, args.seq_len, args.batch_size)

    results = []
    for config_str in configs:
        ckpt = Path(f"ckpt_{config_str}.pt")
        if not ckpt.exists():
            print(f"  {config_str}: checkpoint not found")
            continue
        model = CustomAlternatingGPT(
            vocab_size=len(tokenizer), d_model=args.d_model,
            n_heads=args.n_heads,
            config_str=config_str, max_seq_len=args.seq_len,
        ).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device)['model_state_dict'])
        ppl = evaluate_ppl(model, val_loader, device)
        results.append((config_str, ppl))
        print(f"  {config_str}: PPL={ppl:.2f}")

    if results:
        best = min(results, key=lambda x: x[1])
        print(f"\n  ★ Best: {best[0]} (PPL={best[1]:.2f})")


if __name__ == "__main__":
    main()
