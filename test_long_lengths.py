#!/usr/bin/env python3
"""
Test √Block Attention scalability up to 128k length on M2 Pro.
Only √Block is tested (Dense would OOM).
Usage: python test_long_lengths.py
"""

import torch
import torch.nn.functional as F
import time
import os
import gc
import json
import math

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

def chunked_scaled_dot_product_attention(query, key, value, chunk_size=1024):
    """
    Fallback for MPS: split query into chunks to avoid long-sequence issues.
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
    q_sel = q[:, :, indices, :]
    k_sel = k[:, :, indices, :]
    v_sel = v[:, :, indices, :]
    if use_chunked:
        out_sel = chunked_scaled_dot_product_attention(q_sel, k_sel, v_sel)
    else:
        out_sel = F.scaled_dot_product_attention(q_sel, k_sel, v_sel)
    out = torch.zeros_like(q)
    out[:, :, indices, :] = out_sel
    return out, len(indices)

def test_single_length(device, L, use_chunked=False):
    torch.manual_seed(42)
    # Use float16 to save memory
    q = torch.randn(2, 8, L, 64, device=device, dtype=torch.float16)
    k, v = q, q
    torch.mps.empty_cache()
    gc.collect()
    torch.mps.synchronize()
    try:
        start = time.time()
        _, R = sqrt_block_attention(q, k, v, use_chunked=use_chunked)
        torch.mps.synchronize()
        elapsed = time.time() - start
        # Estimate memory: QKV (3*L*64*2 bytes) + out (same) + intermediate
        # Rough approximation
        mem_mb = (3 * L * 64 * 2) / 1024**2  # Q,K,V in float16
        mem_mb += (R * 64 * 2) / 1024**2     # representatives
        return elapsed, R, mem_mb
    except Exception as e:
        print(f"  Error: {e}")
        return None, None, None

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == 'mps':
        print("Note: MPS may have limitations beyond 64k. Using chunked fallback if needed.")
    
    # Lengths: 8192, 16384, 32768, 65536, 131072
    lengths = [8192, 16384, 32768, 65536, 131072]
    results = {}
    
    for L in lengths:
        print(f"\n--- Testing length {L} (√Block only) ---")
        # For L >= 65536, force chunked mode to be safe
        use_chunked = (L >= 65536)
        elapsed, R, mem_mb = test_single_length(device, L, use_chunked=use_chunked)
        if elapsed is not None:
            print(f"  √Block latency: {elapsed*1000:.2f} ms")
            print(f"  Representatives: {R} (theory √L ≈ {int(L**0.5)})")
            print(f"  Est. memory: {mem_mb:.0f} MB")
            results[L] = {
                'latency_ms': elapsed * 1000,
                'representatives': R,
                'est_memory_mb': mem_mb,
                'use_chunked': use_chunked
            }
        else:
            print(f"  Failed at L={L}")
            results[L] = {'error': 'failed'}
    
    # Save results
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(script_dir, 'long_benchmark.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    
    # Print summary table
    print("\n=== Summary (√Block Attention only) ===")
    print("Length\tLatency(ms)\tRepresentatives\tMemory(MB)")
    for L in lengths:
        if L in results and 'error' not in results[L]:
            r = results[L]
            print(f"{L}\t{r['latency_ms']:.1f}\t\t{r['representatives']}\t\t{r['est_memory_mb']:.0f}")
        else:
            print(f"{L}\tFAILED\t\t-\t\t-")

if __name__ == "__main__":
    main()
