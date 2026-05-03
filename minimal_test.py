#!/usr/bin/env python3
"""
√Block Attention Benchmark
Usage: python minimal_test.py
Output: benchmark_data.json
"""

import torch
import torch.nn.functional as F
import time
import os
import gc
import json

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

def sqrt_block_attention(q, k, v):
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
    out_sel = F.scaled_dot_product_attention(q_sel, k_sel, v_sel)
    out = torch.zeros_like(q)
    out[:, :, indices, :] = out_sel
    return out

def test_length(device, L, try_dense=True):
    torch.manual_seed(42)
    q = torch.randn(2, 8, L, 64, device=device, dtype=torch.float16)
    k, v = q, q
    result = {}
    
    # Dense
    if try_dense:
        try:
            torch.mps.empty_cache()
            gc.collect()
            torch.mps.synchronize()
            start = time.time()
            out = F.scaled_dot_product_attention(q, k, v)
            torch.mps.synchronize()
            elapsed = time.time() - start
            result['dense'] = elapsed
        except Exception as e:
            print(f"  Dense failed for L={L}: {e}")
            result['dense'] = float('inf')
    else:
        result['dense'] = float('inf')
    
    # Block
    try:
        torch.mps.empty_cache()
        gc.collect()
        torch.mps.synchronize()
        start = time.time()
        out = sqrt_block_attention(q, k, v)
        torch.mps.synchronize()
        elapsed = time.time() - start
        result['block'] = elapsed
    except Exception as e:
        print(f"  Block failed for L={L}: {e}")
        result['block'] = float('inf')
    
    return result

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    lengths = [1024, 2048, 4096, 8192]
    results = {}
    
    for L in lengths:
        print(f"\n--- Length {L} ---")
        try_dense = (L <= 8192)   # 仅在长度≤8192时尝试dense
        res = test_length(device, L, try_dense)
        results[L] = res
        if res.get('dense', float('inf')) != float('inf'):
            print(f"  Dense: {res['dense']*1000:.2f} ms")
        else:
            print(f"  Dense: OOM/skip")
        if res.get('block', float('inf')) != float('inf'):
            print(f"  Block: {res['block']*1000:.2f} ms")
            if res.get('dense', float('inf')) != float('inf'):
                speedup = res['dense'] / res['block']
                print(f"  Speedup: {speedup:.2f}x")
        else:
            print(f"  Block: failed")
    
    # 保存 JSON 数据（将 inf 转为 null）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, 'benchmark_data.json')
    json_results = {}
    for L, res in results.items():
        json_res = {}
        for k, v in res.items():
            if v == float('inf'):
                json_res[k] = None
            else:
                json_res[k] = v
        json_results[L] = json_res
    with open(data_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\nBenchmark data saved to {data_path}")

if __name__ == "__main__":
    main()
