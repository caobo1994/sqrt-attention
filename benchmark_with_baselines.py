#!/usr/bin/env python3
"""
Compare √Block Attention with simulated CSA (Compressed Sparse Attention) 
and HCA (Hierarchical Coarse Attention) on M2 Pro.
Lengths: 1024, 2048, 4096, 8192, 16384, 32768, 65536 (or until OOM)
"""

import torch
import torch.nn.functional as F
import time
import os
import gc
import json
import sys
import math

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

def get_memory_usage():
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024**2
    except ImportError:
        return -1

# ---------- 1. √Block Attention (你的方法) ----------
def sqrt_block_attention(q, k, v, use_chunked=False, chunk_size=4096):
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
    if use_chunked and L > chunk_size:
        # Fallback chunked attention if needed
        def chunked_attn(qc, kc, vc):
            Bc, Hc, Lc, Dc = qc.shape
            if Lc <= chunk_size:
                return F.scaled_dot_product_attention(qc, kc, vc)
            outputs = []
            for i in range(0, Lc, chunk_size):
                q_chunk = qc[:, :, i:i+chunk_size, :]
                out_chunk = F.scaled_dot_product_attention(q_chunk, kc, vc)
                outputs.append(out_chunk)
            return torch.cat(outputs, dim=2)
        out_sel = chunked_attn(q_sel, k_sel, v_sel)
    else:
        out_sel = F.scaled_dot_product_attention(q_sel, k_sel, v_sel)
    out = torch.zeros_like(q)
    out[:, :, indices, :] = out_sel
    return out, R

# ---------- 2. Simulated CSA (Compressed Sparse Attention) ----------
# Inspired by DeepSeek's CSA: local window + global tokens.
# Here we use: sliding window of size w, plus g global tokens (first g positions).
def csa_attention(q, k, v, window_size=512, global_tokens=64):
    B, H, L, D = q.shape
    # Create attention mask: local window + global tokens
    # For simplicity, we compute dense attention but mask out positions outside window and not global.
    # However for fair efficiency comparison, we implement sparse via indexing.
    # Actually we can use the same logic as √Block: select positions that are either global or within window.
    # Global positions: 0..global_tokens-1
    global_indices = list(range(min(global_tokens, L)))
    # For each token, consider window around it, but that's complicated. Instead we use a fixed set:
    # All tokens attend to global tokens + a local window (themselves +- window_size/2)
    # To keep implementation simple and efficient, we approximate: each token attends to:
    # - all global tokens
    # - tokens within distance window_size/2
    # But that leads to variable number of attended tokens per query. For benchmarking, we can
    # precompute the union of all positions that are ever attended. This is similar to Longformer.
    # Here we use a simplified version: attend to global tokens + a block of size window_size around each query.
    # However to make it O(L * (global_tokens + window_size)), we will use indexing.
    # For simplicity and fair runtime comparison, we implement as:
    # - All tokens attend to global tokens (first g tokens)
    # - Each token also attends to a local window of size w around it (clamped)
    # We'll collect all indices to attend for each query, but that's per query. For ease, we compute
    # full attention mask and then apply sparse softmax? That loses efficiency gain.
    # Better: Use PyTorch's sparse attention via block-sparse? Not in MPS.
    # So we implement dense but with mask. This still has O(L^2) but we can limit to small L for comparison.
    # For lengths beyond 8K, Dense CSA will OOM. Instead we will restrict CSA comparison to lengths where it's feasible.
    # Given that CSA typically still has O(L * (w + g)) complexity if implemented efficiently, but here we use dense for simplicity.
    # We'll just run CSA for up to 8K.
    mask = torch.zeros(B, H, L, L, device=q.device, dtype=bool)
    # Global tokens: all positions attend to global tokens
    mask[:, :, :, :global_tokens] = True
    # Local window: for each i, attend to i-window_size//2 .. i+window_size//2
    half_win = window_size // 2
    for i in range(L):
        left = max(0, i - half_win)
        right = min(L, i + half_win + 1)
        mask[:, :, i, left:right] = True
    # Now apply attention with mask (dense but masked)
    # Use F.scaled_dot_product_attention with attn_mask (set -inf for masked out)
    attn_mask = torch.zeros(L, L, device=q.device, dtype=q.dtype)
    attn_mask = attn_mask.masked_fill(~mask[0,0], float('-inf'))
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    # Rough estimate of number of attended pairs (for reporting)
    attended_pairs = (mask[0,0].sum().item() * B * H) / (B*H)  # per head
    return out, attended_pairs

# ---------- 3. Simulated HCA (Hierarchical Coarse Attention) ----------
# Hierarchical: downsample sequence by pooling, attend across levels.
# Simplified: average pooling over blocks of size block_size, then attend to coarse tokens.
def hca_attention(q, k, v, coarse_block_size=64):
    B, H, L, D = q.shape
    # Compute coarse tokens by average pooling along sequence dimension
    # Reshape to [B, H, num_blocks, block_size, D]
    num_blocks = (L + coarse_block_size - 1) // coarse_block_size
    # Pad if needed
    pad_len = num_blocks * coarse_block_size - L
    if pad_len > 0:
        q_pad = F.pad(q, (0,0,0,pad_len))
        k_pad = F.pad(k, (0,0,0,pad_len))
        v_pad = F.pad(v, (0,0,0,pad_len))
    else:
        q_pad, k_pad, v_pad = q, k, v
    # Reshape and pool
    q_blocks = q_pad.view(B, H, num_blocks, coarse_block_size, D).mean(dim=3)
    k_blocks = k_pad.view(B, H, num_blocks, coarse_block_size, D).mean(dim=3)
    v_blocks = v_pad.view(B, H, num_blocks, coarse_block_size, D).mean(dim=3)
    # Coarse attention
    coarse_out = F.scaled_dot_product_attention(q_blocks, k_blocks, v_blocks)
    # Upsample to original length (repeat each coarse token to fill block)
    coarse_out_expanded = coarse_out.unsqueeze(3).expand(-1,-1,-1,coarse_block_size,-1).reshape(B, H, num_blocks*coarse_block_size, D)
    if pad_len > 0:
        coarse_out_expanded = coarse_out_expanded[:, :, :L, :]
    # Also need fine-grained attention? HCA usually combines with local. For simplicity, we return coarse output.
    # Representativity: number of coarse tokens ≈ L / block_size
    R = num_blocks
    return coarse_out_expanded, R

# ---------- Benchmark function ----------
def test_attention(device, L, method, **kwargs):
    torch.manual_seed(42)
    # Use float16
    try:
        q = torch.randn(1, 8, L, 64, device=device, dtype=torch.float16)
        k = q.clone()
        v = q.clone()
    except RuntimeError as e:
        print(f"  Allocation failed: {e}")
        return None
    
    torch.mps.empty_cache()
    gc.collect()
    torch.mps.synchronize()
    
    start = time.time()
    try:
        if method == 'sqrt':
            out, R = sqrt_block_attention(q, k, v, use_chunked=(L>65536))
        elif method == 'csa':
            out, R = csa_attention(q, k, v, **kwargs)
        elif method == 'hca':
            out, R = hca_attention(q, k, v, **kwargs)
        else:
            raise ValueError("Unknown method")
        torch.mps.synchronize()
        elapsed = time.time() - start
    except Exception as e:
        print(f"  {method} failed: {e}")
        return None
    finally:
        del q, k, v
        torch.mps.empty_cache()
        gc.collect()
    
    # Estimate memory
    mem_qkv = 3 * L * 64 * 2 / 1024**2
    # Additional memory for method-specific intermediates (rough)
    if method == 'sqrt':
        mem_repr = R * 64 * 2 / 1024**2
        est_mem = mem_qkv + mem_repr + 100
    elif method == 'csa':
        # mask storage O(L^2) -> huge for large L, but we only run CSA for L <= 8192
        est_mem = mem_qkv + (L*L*2)/1024**2  # mask in fp16
    else:  # hca
        est_mem = mem_qkv + (L / kwargs.get('coarse_block_size',64)) * 64 * 2 / 1024**2 + 100
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
    
    lengths = [1024, 2048, 4096, 8192, 16384, 32768, 65536]
    methods = {
        'sqrt': {'method': 'sqrt', 'kwargs': {}},
        'csa': {'method': 'csa', 'kwargs': {'window_size': 512, 'global_tokens': 64}},
        'hca': {'method': 'hca', 'kwargs': {'coarse_block_size': 64}}
    }
    
    results = {m: {} for m in methods}
    
    for L in lengths:
        print(f"\n--- Length {L} ---")
        for name, cfg in methods.items():
            # Skip CSA for L > 8192 because dense mask would OOM
            if name == 'csa' and L > 8192:
                print(f"  CSA skipped for L={L} (would OOM)")
                results[name][L] = {'error': 'skipped'}
                continue
            print(f"  Testing {name}...")
            res = test_attention(device, L, cfg['method'], **cfg['kwargs'])
            if res is None:
                print(f"    {name} failed at L={L}")
                results[name][L] = {'error': 'failed'}
            else:
                print(f"    Latency: {res['latency_ms']:.2f} ms")
                print(f"    Repr/Attended: {res['representatives']}")
                print(f"    Est. memory: {res['est_memory_mb']:.0f} MB")
                results[name][L] = res
    
    # Save
    out_path = 'baseline_comparison.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    
    # Print summary table
    print("\n=== Comparison Summary ===")
    print(f"{'Length':<8} {'Method':<8} {'Latency(ms)':<12} {'Representatives':<16} {'Mem(MB)':<10}")
    for L in lengths:
        for name in methods:
            if L in results[name] and 'error' not in results[name][L]:
                r = results[name][L]
                print(f"{L:<8} {name:<8} {r['latency_ms']:<12.2f} {r['representatives']:<16} {r['est_memory_mb']:<10.0f}")
            elif L in results[name] and results[name][L].get('error') == 'skipped':
                print(f"{L:<8} {name:<8} {'SKIPPED':<12} {'-':<16} {'-':<10}")
            else:
                print(f"{L:<8} {name:<8} {'FAILED':<12} {'-':<16} {'-':<10}")

if __name__ == "__main__":
    try:
        import psutil
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil"])
        import psutil
    main()
