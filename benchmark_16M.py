#!/usr/bin/env python3
"""
√Block Attention benchmark from 1K to 16M (strictly doubling).
Only √Block is tested (Dense would be impossible).
Results saved to long_benchmark.json.
Usage: python benchmark_16m.py
"""

import torch
import torch.nn.functional as F
import time
import os
import gc
import json
import sys

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

def get_memory_usage():
    """Return current memory usage in MB."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024**2
    except ImportError:
        return -1

def chunked_scaled_dot_product_attention(query, key, value, chunk_size=4096):
    """
    Split query into chunks to avoid MPS long-sequence issues.
    chunk_size increased to 4096 for better performance on huge lengths.
    """
    B, H, L, D = query.shape
    if L <= chunk_size:
        return F.scaled_dot_product_attention(query, key, value)
    outputs = []
    for i in range(0, L, chunk_size):
        q_chunk = query[:, :, i:min(i+chunk_size, L), :]
        out_chunk = F.scaled_dot_product_attention(q_chunk, key, value)
        outputs.append(out_chunk)
    return torch.cat(outputs, dim=2)

def sqrt_block_attention(q, k, v, use_chunked=False):
    B, H, L, D = q.shape
    num_blocks = max(1, int(L ** 0.5))
    block_size = L // num_blocks
    indices = []
    for i in range(num_blocks):
        start = i * block_size
        end = min((i+1) * block_size, L)
        n_repr = max(1, int((end - start) ** 0.5))
        step = max(1, (end - start) // n_repr)
        indices.extend(range(start, end, step))
    indices = sorted(set(indices))
    R = len(indices)
    q_sel = q[:, :, indices, :]
    k_sel = k[:, :, indices, :]
    v_sel = v[:, :, indices, :]
    if use_chunked:
        out_sel = chunked_scaled_dot_product_attention(q_sel, k_sel, v_sel)
    else:
        out_sel = F.scaled_dot_product_attention(q_sel, k_sel, v_sel)
    out = torch.zeros_like(q)
    out[:, :, indices, :] = out_sel
    return out, R

def test_length(device, L, force_chunked=False):
    torch.manual_seed(42)
    # Use float16
    try:
        q = torch.randn(1, 8, L, 64, device=device, dtype=torch.float16)
        k = q.clone()
        v = q.clone()
    except RuntimeError as e:
        print(f"  Allocation failed: {e}")
        return None

    mem_before = get_memory_usage()
    torch.mps.empty_cache()
    gc.collect()
    torch.mps.synchronize()

    # Force chunked for lengths > 64k
    use_chunked = force_chunked or (L > 65536)
    start = time.time()
    try:
        _, R = sqrt_block_attention(q, k, v, use_chunked=use_chunked)
        torch.mps.synchronize()
        elapsed = time.time() - start
    except Exception as e:
        print(f"  Attention failed: {e}")
        return None
    finally:
        del q, k, v
        torch.mps.empty_cache()
        gc.collect()

    # Estimate peak memory: QKV (3 * L * 64 * 2 bytes) + overhead
    mem_qkv = 3 * L * 64 * 2 / 1024**2
    mem_repr = R * 64 * 2 / 1024**2 if R else 0
    est_mem = mem_qkv + mem_repr + 100  # 100MB overhead

    return {
        'latency_ms': elapsed * 1000,
        'representatives': R,
        'est_memory_mb': est_mem,
        'use_chunked': use_chunked,
        'success': True
    }

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type != 'mps':
        print("MPS not available. Exiting.")
        return

    # Generate lengths: 1K, 2K, 4K, ..., 16M
    lengths = [1024]
    while lengths[-1] < 16777216:
        lengths.append(lengths[-1] * 2)
    # Ensure 16M is exactly included
    if lengths[-1] != 16777216:
        lengths.append(16777216)
    print(f"Testing lengths (doubling): {lengths}")

    results = {}
    for L in lengths:
        print(f"\n--- Length {L} ---")
        # For very large lengths, might OOM; we try anyway
        res = test_length(device, L)
        if res is None:
            print(f"  Failed at {L}")
            results[L] = {'error': 'failed'}
        else:
            print(f"  Latency: {res['latency_ms']:.2f} ms")
            print(f"  Representatives: {res['representatives']} (√L ≈ {int(L**0.5)})")
            print(f"  Est. memory: {res['est_memory_mb']:.0f} MB")
            print(f"  Chunked: {res['use_chunked']}")
            results[L] = res

    # Save JSON
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(script_dir, 'long_benchmark.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Print summary
    print("\n=== Summary (√Block Attention, doubling from 1K to 16M) ===")
    print(f"{'Length':<10} {'Latency(ms)':<12} {'Repr':<8} {'Est.Mem(MB)':<12} {'Chunked':<8}")
    for L in lengths:
        if L in results and 'error' not in results[L]:
            r = results[L]
            print(f"{L:<10} {r['latency_ms']:<12.2f} {r['representatives']:<8} {r['est_memory_mb']:<12.0f} {r['use_chunked']:<8}")
        else:
            print(f"{L:<10} FAILED")

if __name__ == "__main__":
    # Ensure psutil is available
    try:
        import psutil
    except ImportError:
        print("Installing psutil for memory monitoring...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil"])
    main()
