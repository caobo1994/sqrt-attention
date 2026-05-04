#!/usr/bin/env python3
"""
Evaluate √Block Pathfinder models on Pathfinder-X.
Compares accuracy between Uniform and TopK-Norm sampling.
"""

import torch
import argparse
import math
import time
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.lra_model import PathfinderModel
from utils import get_dataloaders


@torch.no_grad()
def evaluate_accuracy(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    for batch in dataloader:
        if len(batch) == 4:
            x, p1, p2, y = batch
            p1, p2 = p1.to(device), p2.to(device)
        else:
            x, y = batch
            p1, p2 = None, None
        x, y = x.to(device), y.to(device)
        logits = model(x, pos1=p1, pos2=p2)
        preds = logits.argmax(dim=-1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return correct / max(total, 1)


@torch.no_grad()
def profile_inference(model, dataloader, device, burn_in=1, num_batches=5):
    model.eval()
    loader_iter = iter(dataloader)
    for _ in range(burn_in):
        try:
            x, _ = next(loader_iter)
            _ = model(x.to(device))
        except StopIteration:
            break
    if device.type == "mps":
        torch.mps.synchronize()

    timings = []
    mems = []
    for _ in range(num_batches):
        try:
            x, _ = next(loader_iter)
        except StopIteration:
            break
        x = x.to(device)
        if device.type == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()
        _ = model(x)
        if device.type == "mps":
            torch.mps.synchronize()
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1000)
        if device.type == "mps":
            mems.append(torch.mps.current_allocated_memory() / 1024**2)

    avg_time = np.mean(timings) if timings else 0
    avg_mem = np.mean(mems) if mems else 0
    return avg_time, avg_mem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--uniform_ckpt', type=str, default='pathfinder_uniform_best.pt')
    parser.add_argument('--topk_ckpt', type=str, default='pathfinder_topk_norm_best.pt')
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--seq_len', type=int, default=16384)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_val', type=int, default=1000)
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}\n")

    _, val_loader = get_dataloaders(
        batch_size=args.batch_size, num_val=args.num_val
    )

    results = []
    for name, ckpt, sampling in [
        ("√Block Uniform", args.uniform_ckpt, 'uniform'),
        ("√Block TopK-Norm", args.topk_ckpt, 'topk_norm'),
    ]:
        print(f"Loading {name} from {ckpt}...")
        try:
            model = PathfinderModel(
                d_model=args.d_model,
                n_heads=args.n_heads,
                n_layers=args.n_layers,
                sampling=sampling,
                seq_len=args.seq_len,
                num_classes=2,
            ).to(device)
            model.load_state_dict(torch.load(ckpt, map_location=device))
            model.eval()

            acc = evaluate_accuracy(model, val_loader, device)
            _, val_loader = get_dataloaders(
                batch_size=args.batch_size, num_val=args.num_val
            )
            inf_time, mem = profile_inference(model, val_loader, device)
            results.append((name, acc, inf_time, mem))

            print(f"  Accuracy: {acc:.4f} ({acc*100:.2f}%)")
            print(f"  Inference: {inf_time:.2f} ms, Memory: {mem:.2f} MB\n")
        except FileNotFoundError:
            print(f"  {RED}Checkpoint not found: {ckpt}{RESET}\n")
            results.append((name, 0.0, 0.0, 0.0))

    # Summary
    print("=" * 55)
    print(f"{'Method':<22} {'Accuracy':<10} {'Time(ms)':<10} {'Mem(MB)':<10}")
    print("=" * 55)
    for name, acc, t, m in results:
        print(f"{name:<22} {acc*100:<9.2f}% {t:<10.2f} {m:<10.1f}")
    print("=" * 55)

    # Compare with random baseline (50%)
    if len(results) >= 1 and results[0][1] > 0:
        print(f"\nRandom baseline: 50.00%")
        for name, acc, _, _ in results:
            if acc > 0:
                gap = (acc - 0.5) * 100
                print(f"{name}: {gap:+.1f}% above random")


if __name__ == "__main__":
    main()
