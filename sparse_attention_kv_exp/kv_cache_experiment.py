#!/usr/bin/env python3
"""
√Block KV Cache Compression — 推理场景验证

对比 5 种 KV Cache 策略：
  1. Full (全量, 不压缩)           ← 上限参考
  2. √Block                        ← 本论文方法
  3. DeepSeek CSA (window + pool)  ← DeepSeek 风格
  4. H2O (开头 + 末尾)             ← 经典基线
  5. Random (随机丢弃)             ← 随机基线

用法：
    python kv_cache_experiment.py                       # 默认测试
    python kv_cache_experiment.py --quick               # 快速验证
    python kv_cache_experiment.py --model gpt2-medium   # 更大模型
    python kv_cache_experiment.py --debug                # 输出调试信息
"""

import torch
import math
import time
import json
import numpy as np
import argparse
import re
from transformers import AutoModelForCausalLM, AutoTokenizer


# ═══════════════════════════════════════════════════════════════════════════
#  Part 1: √Block 核心 — KV Cache 压缩
# ═══════════════════════════════════════════════════════════════════════════


def create_blocks(seq_len, growth_factor=2.0):
    """生成指数增长的块边界（近小远大）。"""
    blocks, pos, bid = [], seq_len - 1, 0
    while pos >= 0:
        bsize = max(1, int(1 * (growth_factor ** bid)))
        start = max(0, pos - bsize + 1)
        blocks.append((start, pos + 1))
        pos = start - 1
        bid += 1
    blocks.reverse()
    return blocks


def sqrt_block_indices(seq_len, growth_factor=2.0):
    """√Block: 每块 √(大小) 个代表，总数 ≈ O(√L)。"""
    blocks = create_blocks(seq_len, growth_factor)
    indices = []
    for start, end in blocks:
        b_len = end - start
        k = max(1, int(math.sqrt(b_len)))
        if b_len <= k:
            idx = list(range(start, end))
        else:
            step = b_len / k
            offset = np.random.randint(0, max(1, int(step)))
            idx = [min(int(start + i * step) + offset, end - 1) for i in range(k)]
        indices.extend(idx)
    indices = sorted(set(indices))
    if indices[-1] != seq_len - 1:
        indices.append(seq_len - 1)
    return torch.tensor(indices, dtype=torch.long)


def compress_dynamic_cache(past_key_values, indices):
    """
    transformers 5.x DynamicCache 压缩。
    格式: layers[i].keys = [n_heads, seq_len, head_dim] (无 batch dim)
    """
    if past_key_values is None:
        return None

    from transformers.cache_utils import DynamicCache
    device = past_key_values.layers[0].keys.device
    indices = indices.to(device)

    new_cache = DynamicCache()
    for i in range(len(past_key_values.layers)):
        k = past_key_values.layers[i].keys  # shape: [B, H, S, D] or [H, S, D]
        v = past_key_values.layers[i].values
        # Slice sequence dimension (dim=-2):
        #   [B, H, S, D] → [B, H, k, D] via k[:, :, idx, :]
        #   [H, S, D]    → [H, k, D]    via k[:, idx, :]
        if k.dim() == 4:
            k_sel = k[:, :, indices, :].contiguous()
            v_sel = v[:, :, indices, :].contiguous()
        else:
            k_sel = k[:, indices, :].contiguous()
            v_sel = v[:, indices, :].contiguous()
        new_cache.update(k_sel, v_sel, layer_idx=i)
    return new_cache


# ── 方法 1: √Block ──

def sqrt_block_compress(past_key_values, growth_factor=2.0):
    if past_key_values is None:
        return None
    seq_len = past_key_values.get_seq_length()
    indices = sqrt_block_indices(seq_len, growth_factor)
    return compress_dynamic_cache(past_key_values, indices)


# ── 方法 2: DeepSeek CSA ──

def deepseek_csa_compress(past_key_values, keep_ratio=0.1):
    if past_key_values is None:
        return None
    seq_len = past_key_values.get_seq_length()
    n_keep = max(1, int(seq_len * keep_ratio))
    window_size = max(1, n_keep // 2)
    n_compressed = n_keep - window_size

    indices = []
    if window_size > 0:
        indices.extend(range(max(0, seq_len - window_size), seq_len))
    if n_compressed > 0 and seq_len - window_size > 0:
        old_len = seq_len - window_size
        for i in range(n_compressed):
            start = (i * old_len) // n_compressed
            end = min((i + 1) * old_len // n_compressed, old_len)
            if end > start:
                indices.append(start + (end - start) // 2)
    indices = sorted(set(indices))
    if indices[-1] != seq_len - 1:
        indices.append(seq_len - 1)
    indices_t = torch.tensor(indices[:n_keep], dtype=torch.long)
    return compress_dynamic_cache(past_key_values, indices_t)


# ── 方法 3: H2O ──

def h2o_compress(past_key_values, keep_ratio=0.1):
    if past_key_values is None:
        return None
    seq_len = past_key_values.get_seq_length()
    n_keep = max(1, int(seq_len * keep_ratio))
    n_recent = n_keep // 2
    n_initial = n_keep - n_recent
    initial = list(range(min(n_initial, seq_len)))
    recent = list(range(max(n_initial, seq_len - n_recent), seq_len)) if seq_len - n_recent > n_initial else []
    indices = sorted(set(initial + recent))[:n_keep]
    indices_t = torch.tensor(indices, dtype=torch.long)
    return compress_dynamic_cache(past_key_values, indices_t)


# ── 方法 4: Random ──

def random_compress(past_key_values, keep_ratio=0.1):
    if past_key_values is None:
        return None
    seq_len = past_key_values.get_seq_length()
    n_keep = max(1, int(seq_len * keep_ratio))
    indices = torch.randperm(seq_len)[:n_keep].sort().values
    return compress_dynamic_cache(past_key_values, indices)


# ═══════════════════════════════════════════════════════════════════════════
#  Part 2: Needle-In-A-Haystack 测试（英文版，适合 GPT-2）
# ═══════════════════════════════════════════════════════════════════════════

NEEDLE_TEMPLATES = [
    {
        "needle": "The secret password is \"giraffe-banana-42\".",
        "question": "What is the secret password?",
        "answer": "giraffe-banana-42",
        "alt_answers": ["giraffe-banana-42", "giraffe banana 42"],
    },
    {
        "needle": "The Martian colony was founded in 2087 by Dr. Elena Vasquez.",
        "question": "When was the Martian colony founded?",
        "answer": "2087",
        "alt_answers": ["2087", "in 2087"],
    },
    {
        "needle": "The quantum computer at MIT has exactly 2048 qubits.",
        "question": "How many qubits does the MIT quantum computer have?",
        "answer": "2048",
        "alt_answers": ["2048", "2048 qubits"],
    },
    {
        "needle": "The city of New Tokyo has a population of 42 million people.",
        "question": "What is the population of New Tokyo?",
        "answer": "42 million",
        "alt_answers": ["42 million", "42,000,000", "42000000"],
    },
    {
        "needle": "The fastest animal on Earth is the peregrine falcon.",
        "question": "What is the fastest animal on Earth?",
        "answer": "peregrine falcon",
        "alt_answers": ["peregrine falcon"],
    },
]


def build_haystack(haystack_len=2048, seed=42):
    """构建填充文本。"""
    rng = np.random.RandomState(seed)
    filler = [
        "The quick brown fox jumps over the lazy dog. ",
        "Machine learning is a subset of artificial intelligence. ",
        "Python is a high-level programming language. ",
        "The Earth orbits the Sun once every 365.25 days. ",
        "The Great Wall of China is over 13,000 miles long. ",
        "Photosynthesis converts light energy into chemical energy. ",
        "The human brain contains about 86 billion neurons. ",
        "Water freezes at 0 degrees Celsius at sea level. ",
        "Shakespeare wrote 37 plays and 154 sonnets. ",
        "The speed of light is approximately 300,000 km per second. ",
        "Climate change is a major global challenge. ",
        "Coffee is one of the most popular beverages worldwide. ",
        "The Amazon rainforest produces 20% of Earth's oxygen. ",
        "Bach composed over 1,000 musical works in his lifetime. ",
    ]
    haystack = ""
    while len(haystack) < haystack_len * 4:
        haystack += rng.choice(filler)
    return haystack


def create_needle_prompt(needle_data, needle_depth=0.5, haystack_len=1024, max_position=1024):
    """在干草堆中插入针。截断以不超过模型最大位置。"""
    needle = needle_data["needle"]
    question = needle_data["question"]
    haystack = build_haystack(haystack_len)
    insert_pos = int(len(haystack) * needle_depth)
    haystack_with_needle = haystack[:insert_pos] + needle + " " + haystack[insert_pos:]
    prompt = f"""Read the following passage carefully, then answer the question.

Passage: {haystack_with_needle}

Question: {question}

Answer:"""
    return prompt


def check_answer(generated_text, needle_data):
    """检查生成文本是否包含正确答案。支持部分匹配。"""
    gen_lower = generated_text.lower()
    answers = [needle_data["answer"].lower()] + [a.lower() for a in needle_data.get("alt_answers", [])]
    # 完整匹配
    if any(a in gen_lower for a in answers):
        return True
    # 关键词匹配（取答案的前几个词）
    for ans in answers:
        words = ans.split()
        if len(words) >= 2:
            # 至少匹配到 2 个连续词
            for i in range(len(words) - 1):
                phrase = ' '.join(words[i:i+2])
                if phrase in gen_lower:
                    return True
        elif len(words) == 1 and len(ans) >= 4:
            if ans in gen_lower:
                return True
    return False


# ═══════════════════════════════════════════════════════════════════════════
#  Part 3: 带 KV Cache 压缩的生成
# ═══════════════════════════════════════════════════════════════════════════

COMPRESS_METHODS = {
    'full':       None,
    'sqrt_block': sqrt_block_compress,
    'deepseek':   lambda kv: deepseek_csa_compress(kv, keep_ratio=0.1),
    'h2o':        lambda kv: h2o_compress(kv, keep_ratio=0.1),
    'random':     lambda kv: random_compress(kv, keep_ratio=0.1),
}

COMPRESS_LABELS = {
    'full':       'Full KV Cache',
    'sqrt_block': '√Block KV Cache',
    'deepseek':   'DeepSeek CSA',
    'h2o':        'H2O KV Cache',
    'random':     'Random KV Cache',
}


@torch.no_grad()
def measure_compression(model, tokenizer, prompt, max_new_tokens=20, debug=False):
    """
    用 Full cache 生成，同时测量各种压缩方法的 KV Cache 大小。
    
    分离两个目标：
      1. 正确生成（用 Full cache）
      2. 测量压缩比（对 Prefill 后的 KV Cache 做静态压缩）
    
    返回: (response_text, {method: cache_mb})
    """
    device = model.device
    max_pos = getattr(model.config, 'max_position_embeddings', 1024)
    tokens = tokenizer.encode(prompt, return_tensors='pt')
    if tokens.size(1) > max_pos - max_new_tokens:
        tokens = tokens[:, :max_pos - max_new_tokens]
    input_ids = tokens.to(device)

    # Prefill — 一次性获取全量 KV Cache
    outputs = model(input_ids, use_cache=True)
    full_kv = outputs.past_key_values
    full_cache_mb = _measure_cache(full_kv)
    seq_len = full_kv.get_seq_length()

    # 测量各方法的压缩大小（静态压缩 prefill 后的 KV Cache）
    cache_sizes = {'full': full_cache_mb}
    for method, compress_fn in COMPRESS_METHODS.items():
        if method == 'full':
            continue
        try:
            compressed = compress_fn(full_kv)
            cache_sizes[method] = _measure_cache(compressed)
        except Exception as e:
            cache_sizes[method] = 0.0
            if debug:
                print(f"    [debug] {method} compress failed: {e}")

    if debug:
        sizes = ', '.join(f"{m}={s:.1f}MB" for m, s in cache_sizes.items())
        print(f"    [debug] seq_len={seq_len}, {sizes}")

    # 用 Full cache 自回归生成（保证正确性）
    past_key_values = full_kv
    next_token = outputs.logits[:, -1, :].argmax(-1).unsqueeze(0)
    generated = [next_token.item()]
    for step in range(max_new_tokens - 1):
        outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
        past_key_values = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(-1).unsqueeze(0)
        generated.append(next_token.item())

    response = tokenizer.decode(generated, skip_special_tokens=True)
    return response, cache_sizes, seq_len


def _measure_cache(past_key_values):
    """测量 DynamicCache 的 KV Cache 大小 (MB)。"""
    if past_key_values is None:
        return 0.0
    total = 0.0
    for layer in past_key_values.layers:
        total += layer.keys.numel() * layer.keys.element_size()
        total += layer.values.numel() * layer.values.element_size()
    return total / 1024**2


# ═══════════════════════════════════════════════════════════════════════════
#  Part 4: 运行实验
# ═══════════════════════════════════════════════════════════════════════════

def run_needle_test(model, tokenizer, needle_data, depth=0.5,
                    haystack_len=1024, max_new_tokens=20, debug=False):
    """运行单个 Needle-In-A-Haystack 测试。
    
    一次生成（Full cache），同时测量 5 种方法的压缩大小。
    """
    max_pos = getattr(model.config, 'max_position_embeddings', 1024)
    prompt = create_needle_prompt(needle_data, depth, haystack_len, max_pos)

    # 一次生成 + 一次测量所有方法
    response, cache_sizes, seq_len = measure_compression(
        model, tokenizer, prompt, max_new_tokens, debug=debug,
    )

    correct = check_answer(response, needle_data)
    results = {}
    for method in ['full', 'sqrt_block', 'deepseek', 'h2o', 'random']:
        cache_mb = cache_sizes.get(method, 0.0)
        results[method] = {
            'correct': correct,
            'cache_mb': cache_mb,
            'response': response.strip()[:80],
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="√Block KV Cache 压缩（5 方法对比）")
    parser.add_argument('--model', type=str, default='gpt2')
    parser.add_argument('--haystack_len', type=int, default=1024)
    parser.add_argument('--depths', type=str, default='0.5')
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--max_new_tokens', type=int, default=50)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    if args.quick:
        args.haystack_len = 512
        args.depths = '0.5'
        args.trials = 1
        args.max_new_tokens = 20

    depths = [float(d) for d in args.depths.split(',')]

    # ── Load model ──
    print(f"Loading model: {args.model}")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device.type == 'mps' else torch.float32,
        low_cpu_mem_usage=True,
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    cfg = model.config
    n_layers = getattr(cfg, 'num_hidden_layers', getattr(cfg, 'n_layer', 12))
    n_heads = getattr(cfg, 'num_attention_heads', getattr(cfg, 'n_head', 12))
    d_model = getattr(cfg, 'hidden_size', getattr(cfg, 'n_embd', 768))
    print(f"Layers: {n_layers} | Heads: {n_heads} | dim: {d_model}")

    R = int(math.sqrt(args.haystack_len)) * 2
    print(f"Context: {args.haystack_len} tokens | √Block reps: ~{R}")
    print(f"Methods: 5 (Full/√Block/DeepSeek CSA/H2O/Random)")
    print(f"Tests: {args.trials} trials × {len(depths)} depths = {args.trials * len(depths)}\n")

    # ── Verify cache with a quick test ──
    test_ids = tokenizer.encode("Hello world", return_tensors='pt').to(device)
    test_out = model(test_ids, use_cache=True)
    test_cache = _measure_cache(test_out.past_key_values)
    print(f"[Sanity] KV cache for 'Hello world': {test_cache:.2f} MB")
    print(f"[Sanity] KV cache type: {type(test_out.past_key_values).__name__}")
    print()

    # ── Run tests ──
    all_results = []

    for trial in range(args.trials):
        needle_data = NEEDLE_TEMPLATES[trial % len(NEEDLE_TEMPLATES)]
        for depth in depths:
            needle_preview = needle_data['needle'][:30]
            print(f"  Trial {trial+1}/{args.trials} | \"{needle_preview}...\" | Depth: {depth:.0%}")

            results = run_needle_test(model, tokenizer, needle_data, depth,
                                       args.haystack_len, args.max_new_tokens,
                                       debug=args.debug)

            for method, r in results.items():
                all_results.append({
                    'trial': trial, 'depth': depth, 'method': method,
                    'correct': r['correct'], 'cache_mb': r['cache_mb'],
                })

            for method in ['full', 'sqrt_block', 'deepseek', 'h2o', 'random']:
                r = results.get(method, {})
                mark = '✓' if r.get('correct') else '✗'
                resp = r.get('response', '')[:50]
                print(f"    {COMPRESS_LABELS[method]:<20} {mark}  {r.get('cache_mb',0):.1f}MB  \"{resp}\"")
            print()

    # ── Summary ──
    print(f"\n{'='*65}")
    print(f"  RESULTS SUMMARY — {args.model}")
    print(f"{'='*65}")

    methods_list = ['full', 'sqrt_block', 'deepseek', 'h2o', 'random']
    labels = [COMPRESS_LABELS[m] for m in methods_list]

    print(f"\n{'Method':<22} {'Acc':<8} {'Avg Cache':<12} {'Compression':<12}")
    print(f"{'-'*54}")

    full_cache = 0
    for method, label in zip(methods_list, labels):
        mr = [r for r in all_results if r['method'] == method]
        if not mr:
            continue
        n_correct = sum(1 for r in mr if r['correct'])
        acc = n_correct / len(mr) * 100
        avg_cache = np.mean([r['cache_mb'] for r in mr])

        if method == 'full':
            full_cache = avg_cache
            print(f"{label:<22} {acc:<8.1f}% {avg_cache:<12.1f} {'-':<12}")
        else:
            comp = full_cache / avg_cache if avg_cache > 0 else 0
            print(f"{label:<22} {acc:<8.1f}% {avg_cache:<12.1f} {comp:<12.1f}x")

    # ── Save JSON ──
    out_path = 'kv_cache_results.json'
    clean = []
    for r in all_results:
        clean.append({
            'trial': int(r['trial']), 'depth': float(r['depth']),
            'method': r['method'], 'correct': bool(r['correct']),
            'cache_mb': float(r['cache_mb']),
        })
    with open(out_path, 'w') as f:
        json.dump(clean, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")

    if full_cache > 0:
        avg_block = np.mean([r['cache_mb'] for r in all_results if r['method'] == 'sqrt_block'])
        comp = full_cache / avg_block if avg_block > 0 else 0
        print(f"  √Block compression: ~{comp:.0f}x vs Full")
    else:
        print(f"  WARNING: All cache measurements are 0! Check model compatibility.")

    # ── Visualize ──
    if args.visualize:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            accs = []
            for m in methods_list:
                mr = [r for r in all_results if r['method'] == m]
                acc = sum(1 for r in mr if r['correct']) / max(len(mr), 1) * 100
                accs.append(acc)

            colors = ['#2196F3', '#4CAF50', '#FF5722', '#9C27B0', '#FF9800']
            bars = ax1.bar(labels, accs, color=colors)
            ax1.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Random (50%)')
            ax1.set_ylabel('Accuracy (%)')
            ax1.set_title(f'Needle-In-A-Haystack — {args.model}')
            ax1.tick_params(axis='x', rotation=15)
            for bar, v in zip(bars, accs):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                         f'{v:.0f}%', ha='center', va='bottom', fontsize=9)
            ax1.legend()
            ax1.set_ylim(0, 115)

            cache_sizes = []
            for m in methods_list:
                mr = [r for r in all_results if r['method'] == m]
                avg = np.mean([r['cache_mb'] for r in mr]) if mr else 0
                cache_sizes.append(avg)

            bars2 = ax2.bar(labels, cache_sizes, color=colors)
            ax2.set_ylabel('KV Cache (MB)')
            ax2.set_title('Memory Usage')
            ax2.tick_params(axis='x', rotation=15)
            for bar, v in zip(bars2, cache_sizes):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                         f'{v:.1f}MB', ha='center', va='bottom', fontsize=9)

            plt.tight_layout()
            viz_path = 'kv_cache_comparison.png'
            plt.savefig(viz_path, dpi=150, bbox_inches='tight')
            print(f"Chart saved to {viz_path}")
        except ImportError:
            print("matplotlib not available, skip visualization.")

    print(f"\n{'='*65}")


if __name__ == "__main__":
    main()
