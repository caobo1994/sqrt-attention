import torch
import argparse
import math
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from models.gpt_sparse import SparseGPT
from utils import get_dataloaders
from transformers import AutoTokenizer


@torch.no_grad()
def evaluate_ppl(model, dataloader, device):
    """计算验证集困惑度"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    for input_ids, targets in dataloader:
        input_ids, targets = input_ids.to(device), targets.to(device)
        logits = model(input_ids)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1)
        )
        total_loss += loss.item()
        num_batches += 1
    if num_batches == 0:
        return float('inf')
    return math.exp(total_loss / num_batches)


@torch.no_grad()
def profile_inference(model, dataloader, device, burn_in=2, num_batches=10):
    """测量推理时间与峰值内存"""
    model.eval()
    loader_iter = iter(dataloader)
    # 预热
    for _ in range(burn_in):
        try:
            input_ids, _ = next(loader_iter)
        except StopIteration:
            break
        _ = model(input_ids.to(device))
    if device.type == "mps":
        torch.mps.synchronize()

    timings = []
    for _ in range(num_batches):
        try:
            input_ids, _ = next(loader_iter)
        except StopIteration:
            break
        input_ids = input_ids.to(device)
        if device.type == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()
        _ = model(input_ids)
        if device.type == "mps":
            torch.mps.synchronize()
        t1 = time.perf_counter()
        timings.append(t1 - t0)

    avg_time = np.mean(timings) * 1000 if timings else 0.0   # 毫秒
    if device.type == "mps":
        mem = torch.mps.current_allocated_memory() / 1024**2   # MB
    else:
        mem = 0.0
    return avg_time, mem


@torch.no_grad()
def compute_kl_divergence(model_sparse, model_full, dataloader, device, max_batches=20):
    """计算稀疏模型与全注意力模型输出分布的 KL 散度"""
    kl_sum = 0.0
    count = 0
    loader_iter = iter(dataloader)
    for _ in range(max_batches):
        try:
            input_ids, _ = next(loader_iter)
        except StopIteration:
            break
        input_ids = input_ids.to(device)
        logits_full = model_full(input_ids)
        logits_sparse = model_sparse(input_ids)
        kl = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(logits_sparse, dim=-1),
            torch.nn.functional.softmax(logits_full, dim=-1),
            reduction='batchmean'
        )
        kl_sum += kl.item()
        count += 1
    return kl_sum / count if count > 0 else float('inf')


def plot_comparison(uniform_ppl, topk_ppl, uniform_time, topk_time, save_path="comparison.png"):
    """绘制困惑度与推理时间对比柱状图"""
    labels = ['Uniform', 'TopK-Norm']
    ppl_values = [uniform_ppl, topk_ppl]
    time_values = [uniform_time, topk_time]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # 困惑度
    bars1 = ax1.bar(labels, ppl_values, color=['#2196F3', '#4CAF50'])
    ax1.set_ylabel('Perplexity')
    ax1.set_title('Validation PPL (WikiText-103)')
    for bar, v in zip(bars1, ppl_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{v:.2f}', ha='center', va='bottom')

    # 推理时间
    bars2 = ax2.bar(labels, time_values, color=['#2196F3', '#4CAF50'])
    ax2.set_ylabel('Inference Time (ms)')
    ax2.set_title('Avg Batch Inference Time')
    for bar, v in zip(bars2, time_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{v:.2f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Comparison plot saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate sparse attention models on WikiText-103")
    parser.add_argument('--uniform_ckpt', type=str, default='sparse_gpt_uniform_wt103_best.pt',
                        help='Path to uniform sampling checkpoint')
    parser.add_argument('--topk_ckpt', type=str, default='sparse_gpt_topk_norm_wt103_best.pt',
                        help='Path to TopK-Norm sampling checkpoint')
    parser.add_argument('--full_ckpt', type=str, default=None,
                        help='Optional full-attention checkpoint for KL comparison')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--seq_len', type=int, default=256)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=8)
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Model: d_model={args.d_model}, n_heads={args.n_heads}, n_layers={args.n_layers}")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)

    # 加载验证集 DataLoader
    _, val_loader = get_dataloaders(tokenizer, seq_len=args.seq_len, batch_size=args.batch_size)

    # ================== 加载 uniform 模型 ==================
    uniform_model = SparseGPT(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        sampling='uniform',
        max_seq_len=args.seq_len
    ).to(device)
    print(f"Loading uniform checkpoint: {args.uniform_ckpt}")
    uniform_model.load_state_dict(torch.load(args.uniform_ckpt, map_location=device))
    uniform_model.eval()

    # ================== 加载 topk_norm 模型 ==================
    topk_model = SparseGPT(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        sampling='topk_norm',
        max_seq_len=args.seq_len
    ).to(device)
    print(f"Loading topk_norm checkpoint: {args.topk_ckpt}")
    topk_model.load_state_dict(torch.load(args.topk_ckpt, map_location=device))
    topk_model.eval()

    # ================== Step 1: Perplexity Evaluation ==================
    print("\n=== Step 1: Perplexity Evaluation ===")
    ppl_uniform = evaluate_ppl(uniform_model, val_loader, device)
    _, val_loader = get_dataloaders(tokenizer, seq_len=args.seq_len, batch_size=args.batch_size)
    ppl_topk = evaluate_ppl(topk_model, val_loader, device)
    print(f"Uniform Sampling PPL    : {ppl_uniform:.2f}")
    print(f"TopK-Norm Sampling PPL  : {ppl_topk:.2f}")

    # ================== Step 2: Performance Profiling ==================
    print("\n=== Step 2: Performance Profiling ===")
    _, val_loader = get_dataloaders(tokenizer, seq_len=args.seq_len, batch_size=args.batch_size)
    uniform_time, uniform_mem = profile_inference(uniform_model, val_loader, device)
    _, val_loader = get_dataloaders(tokenizer, seq_len=args.seq_len, batch_size=args.batch_size)
    topk_time, topk_mem = profile_inference(topk_model, val_loader, device)
    print(f"Uniform Sampling    - Avg inference time: {uniform_time:.2f} ms, Memory: {uniform_mem:.2f} MB")
    print(f"TopK-Norm Sampling  - Avg inference time: {topk_time:.2f} ms, Memory: {topk_mem:.2f} MB")

    # ================== Step 3: KL Divergence (optional) ==================
    if args.full_ckpt:
        print("\n=== Step 3: KL Divergence vs Full Attention ===")
        full_model = SparseGPT(
            vocab_size=vocab_size,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            sampling='uniform',
            max_seq_len=args.seq_len
        ).to(device)
        full_model.load_state_dict(torch.load(args.full_ckpt, map_location=device))
        full_model.eval()
        _, val_loader = get_dataloaders(tokenizer, seq_len=args.seq_len, batch_size=args.batch_size)
        kl_uniform = compute_kl_divergence(uniform_model, full_model, val_loader, device)
        _, val_loader = get_dataloaders(tokenizer, seq_len=args.seq_len, batch_size=args.batch_size)
        kl_topk = compute_kl_divergence(topk_model, full_model, val_loader, device)
        print(f"KL(Uniform || Full)  : {kl_uniform:.4f}")
        print(f"KL(TopK-Norm || Full): {kl_topk:.4f}")

    # ================== Step 4: Visualization ==================
    print("\n=== Step 4: Visualization ===")
    plot_comparison(ppl_uniform, ppl_topk, uniform_time, topk_time, save_path="comparison_wt103.png")

    # 总结
    print("\n" + "="*50)
    print(" Final Summary (WikiText-103, d_model=512, 8 layers)")
    print("="*50)
    print(f"Uniform     -> PPL: {ppl_uniform:.2f}, Time: {uniform_time:.2f} ms, Mem: {uniform_mem:.2f} MB")
    print(f"TopK-Norm   -> PPL: {ppl_topk:.2f}, Time: {topk_time:.2f} ms, Mem: {topk_mem:.2f} MB")
    improvement = (ppl_uniform - ppl_topk) / ppl_uniform * 100
    print(f"Relative PPL change (negative = TopK-Norm worse): {improvement:+.1f}%")


if __name__ == "__main__":
    main()
