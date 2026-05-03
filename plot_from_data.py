#!/usr/bin/env python3
"""
Generate benchmark chart from JSON data.
Usage: python plot_from_data.py
Requires: matplotlib
"""

import json
import matplotlib.pyplot as plt
import os

def load_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, 'benchmark_data.json')
    if not os.path.exists(data_path):
        raise FileNotFoundError("benchmark_data.json not found. Please run minimal_test.py first.")
    with open(data_path, 'r') as f:
        raw_data = json.load(f)
    lengths = sorted([int(L) for L in raw_data.keys()])
    dense_times = []
    block_times = []
    speedups = []
    for L in lengths:
        data = raw_data[str(L)]
        dense = data.get('dense')
        block = data.get('block')
        if dense is None:
            dense = float('inf')
        if block is None:
            block = float('inf')
        dense_times.append(dense if dense != float('inf') else None)
        block_times.append(block if block != float('inf') else None)
        if dense != float('inf') and block != float('inf'):
            speedups.append(dense / block)
        else:
            speedups.append(None)
    return lengths, dense_times, block_times, speedups

def plot():
    lengths, dense_times, block_times, speedups = load_data()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # 左图：延迟对比
    valid_len = [l for l, d, b in zip(lengths, dense_times, block_times) if d is not None and b is not None]
    valid_dense = [d for d in dense_times if d is not None]
    valid_block = [b for b in block_times if b is not None]
    if valid_len:
        ax1.plot(valid_len, valid_dense, 'o-', label='Dense Attention')
        ax1.plot(valid_len, valid_block, 's-', label='√Block Attention')
    ax1.set_xlabel('Sequence Length')
    ax1.set_ylabel('Latency (ms)')
    ax1.set_title('Latency on M2 Pro (32GB)')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 右图：加速比柱状图
    valid_len_speed = [l for l, s in zip(lengths, speedups) if s is not None]
    valid_speed = [s for s in speedups if s is not None]
    if valid_len_speed:
        ax2.bar([str(l) for l in valid_len_speed], valid_speed, color='steelblue')
    ax2.set_xlabel('Sequence Length')
    ax2.set_ylabel('Speedup (x)')
    ax2.set_title('√Block vs Dense Attention')
    ax2.axhline(y=1, color='r', linestyle='--', label='Baseline (1x)')
    ax2.legend()
    ax2.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'benchmark.png')
    plt.savefig(out_path, dpi=150)
    print(f"Chart saved to {out_path}")

if __name__ == "__main__":
    plot()
