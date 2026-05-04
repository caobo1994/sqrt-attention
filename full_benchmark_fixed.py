#!/usr/bin/env python3
"""
Compare Dense, CSA, HCA, and √Block attention from 1K to 4M (doubling) on M2 Pro.
√Block uses chunked attention for lengths > 65536 to avoid MPS limitations.
Each method stops at first failure.
Results saved to full_comparison_fixed.json.
Usage: python full_benchmark_fixed.py
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
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024**2
    except ImportError:
        return -1

# ---------- 1. Dense Attention ----------
def dense_attention(q, k, v):
    out = F.scaled_dot_product_attention(q, k, v)
    return out, q.size(2)

# ---------- 2. √Block Attention with chunked fallback ----------
def chunked_scaled_dot_product_attention(query, key, value, chunk_size=4096):
    B, H, L, D = query.shape
    if L <= chunk_size:
        return F.scaled_dot_product_attention(query, key, value)
    outputs = []
    for i in range(0, L, chunk_size):
        q_chunk = query[:, :, i:min(i+chunk_size, L), :]
        out_chunk = F.scaled_dot_product_attention(q_chunk, key, value)
        outputs.append(out_chunk)
    return torch.cat(outputs, dim=2)

def sqrt_block_attention(q, k, v, use_chunked=True, chunk_size=4096):
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
        out_sel = chunked_scaled_dot_product_attention(q_sel, k_sel, v_sel, chunk_size)
    else:
        out_sel = F.scaled_dot_product_attention(q_sel, k_sel, v_sel)
    out = torch.zeros_like(q)
    out[:, :, indices, :] = out_sel
    return out, R

# ---------- 3. Simulated CSA (window + global) ----------
def csa_attention(q, k, v, window_size=512, global_tokens=64):
    B, H, L, D = q.shape
    mask = torch.zeros(B, H, L, L, device=q.device, dtype=torch.bool)
    mask[:, :, :, :global_tokens] = True
    half_win = window_size // 2
    for i in range(L):
        left = max(0, i - half_win)
        right = min(L, i + half_win + 1)
        mask[:, :, i, left:right] = True
    attn_mask = torch.zeros(L, L, device=q.device, dtype=q.dtype)
    attn_mask = attn_mask.masked_fill(~mask[0,0], float('-inf'))
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    attended_pairs = mask[0,0].sum().item()
    return out, attended_pairs

# ---------- 4. Simulated HCA (hierarchical coarse) ----------
def hca_attention(q, k, v, coarse_block_size=64):
    B, H, L, D = q.shape
    num_blocks = (L + coarse_block_size - 1) // coarse_block_size
    pad_len = num_blocks * coarse_block_size - L
    if pad_len > 0:
        q_pad = F.pad(q, (0,0,0,pad_len))
        k_pad = F.pad(k, (0,0,0,pad_len))
        v_pad = F.pad(v, (0,0,0,pad_len))
    else:
        q_pad, k_pad, v_pad = q, k, v
    q_blocks = q_pad.view(B, H, num_blocks, coarse_block_size, D).mean(dim=3)
    k_blocks = k_pad.view(B, H, num_blocks, coarse_block_size, D).mean(dim=3)
    v_blocks = v_pad.view(B, H, num_blocks, coarse_block_size, D).mean(dim=3)
    coarse_out = F.scaled_dot_product_attention(q_blocks, k_blocks, v_blocks)
    coarse_out_expanded = coarse_out.unsqueeze(3).expand(-1,-1,-1,coarse_block_size,-1).reshape(B, H, num_blocks*coarse_block_size, D)
    if pad_len > 0:
        coarse_out_expanded = coarse_out_expanded[:, :, :L, :]
    return coarse_out_expanded, num_blocks

# ---------- Benchmark function ----------
def test_method(device, L, method, **kwargs):
    torch.manual_seed(42)
    q = k = v = None
    out = None
    try:
        q = torch.randn(1, 8, L, 64, device=device, dtype=torch.float16)
        k = q.clone()
        v = q.clone()
    except RuntimeError as e:
        return {'error': f'allocation failed: {e}'}
    
    torch.mps.empty_cache()
    gc.collect()
    torch.mps.synchronize()
    
    start = time.time()
    try:
        if method == 'dense':
            out, R = dense_attention(q, k, v)
        elif method == 'sqrt':
            # Automatically use chunked for lengths > 65536
            use_chunked = (L > 65536)
            out, R = sqrt_block_attention(q, k, v, use_chunked=use_chunked)
        elif method == 'csa':
            out, R = csa_attention(q, k, v, **kwargs)
        elif method == 'hca':
            out, R = hca_attention(q, k, v, **kwargs)
        else:
            raise ValueError('Unknown method')
        torch.mps.synchronize()
        elapsed = time.time() - start
    except Exception as e:
        return {'error': str(e)}
    finally:
        if q is not None:
            del q
        if k is not None:
            del k
        if v is not None:
            del v
        if out is not None:
            del out
        torch.mps.empty_cache()
        gc.collect()
    
    # Estimate memory
    mem_qkv = 3 * L * 64 * 2 / 1024**2
    if method == 'dense':
        mem_attn = L * L * 2 / 1024**2
        est_mem = mem_qkv + mem_attn + 100
    elif method == 'csa':
        mem_mask = L * L / 1024**2
        est_mem = mem_qkv + mem_mask + 100
    elif method == 'sqrt':
        mem_repr = R * 64 * 2 / 1024**2
        est_mem = mem_qkv + mem_repr + 100
    else:  # hca
        est_mem = mem_qkv + (R * 64 * 2) / 1024**2 + 100
    
    return {
        'latency_ms': elapsed * 1000,
        'representatives': R,
        'est_memory_mb': est_mem,
        'success': True
    }

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type != 'mps':
        print("MPS not available. Exiting.")
        return
    
    # Lengths: 1K, 2K, 4K, ... up to 4M
    lengths = [1024]
    while lengths[-1] < 4194304:
        lengths.append(lengths[-1] * 2)
    if lengths[-1] != 4194304:
        lengths.append(4194304)
    
    methods = {
        'dense': {'fn': 'dense', 'kwargs': {}},
        'csa':   {'fn': 'csa',   'kwargs': {'window_size': 512, 'global_tokens': 64}},
        'hca':   {'fn': 'hca',   'kwargs': {'coarse_block_size': 64}},
        'sqrt':  {'fn': 'sqrt',  'kwargs': {}}
    }
    
    results = {m: {} for m in methods}
    stop_flag = {m: False for m in methods}
    
    for L in lengths:
        print(f"\n=== Length {L} ===")
        for name, cfg in methods.items():
            if stop_flag[name]:
                print(f"  {name}: skipped (previous failure)")
                results[name][L] = {'error': 'skipped_previous_failure'}
                continue
            
            print(f"  Testing {name}...")
            res = test_method(device, L, cfg['fn'], **cfg['kwargs'])
            if 'error' in res:
                print(f"    {name} failed at L={L}: {res['error']}")
                results[name][L] = {'error': res['error']}
                stop_flag[name] = True
            else:
                print(f"    Latency: {res['latency_ms']:.2f} ms")
                print(f"    Representatives/Attended: {res['representatives']}")
                print(f"    Est. memory: {res['est_memory_mb']:.0f} MB")
                results[name][L] = res
    
    # Save results
    out_path = 'full_comparison_fixed.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    
    # Print summary
    print("\n========== SUMMARY ==========")
    print(f"{'Length':<8} {'Method':<8} {'Latency(ms)':<12} {'Representatives':<18} {'Mem(MB)':<10}")
    for L in lengths:
        for name in methods:
            if L in results[name] and 'error' not in results[name][L]:
                r = results[name][L]
                print(f"{L:<8} {name:<8} {r['latency_ms']:<12.2f} {r['representatives']:<18} {r['est_memory_mb']:<10.0f}")
            elif L in results[name] and 'skipped' in results[name][L].get('error', ''):
                print(f"{L:<8} {name:<8} {'SKIPPED':<12} {'-':<18} {'-':<10}")
            else:
                print(f"{L:<8} {name:<8} {'FAILED':<12} {'-':<18} {'-':<10}")

if __name__ == "__main__":
    try:
        import psutil
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil"])
        import psutil
    main()
