"""
评估三种注意力模型：Full Attention、√Block Uniform、√Block TopK-Norm
输出验证困惑度（PPL）与首 batch 推理时间。
"""
import torch
import math
import time
from models.gpt_sparse import SparseGPT
from utils import get_dataloaders
from transformers import AutoTokenizer

@torch.no_grad()
def evaluate(model, dataloader, device):
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
    return math.exp(total_loss / num_batches)

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)

    # 模型配置（必须与训练时一致）
    d_model, n_heads, n_layers, seq_len = 128, 8, 2, 256
    batch_size = 8

    # 加载验证集
    _, val_loader = get_dataloaders(tokenizer, seq_len=seq_len, batch_size=batch_size)

    configs = [
        ("Full Attention", "full", "sparse_gpt_full_best.pt"),
        ("√Block Uniform", "uniform", "sparse_gpt_uniform_best.pt"),
        ("√Block TopK-Norm", "topk_norm", "sparse_gpt_topk_norm_best.pt"),
    ]

    results = []
    for name, sampling, ckpt in configs:
        model = SparseGPT(vocab_size, d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                          sampling=sampling, max_seq_len=seq_len).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()
        ppl = evaluate(model, val_loader, device)

        # 测量首 batch 推理时间
        val_iter = iter(val_loader)
        input_ids, _ = next(val_iter)
        input_ids = input_ids.to(device)
        if device.type == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()
        _ = model(input_ids)
        if device.type == "mps":
            torch.mps.synchronize()
        t1 = time.perf_counter()
        elapsed = (t1 - t0) * 1000

        results.append((name, ppl, elapsed))
        print(f"{name:25s} | PPL: {ppl:6.2f} | Time: {elapsed:6.2f} ms")

    print("\n--- Final Comparison ---")
    for name, ppl, t in results:
        print(f"{name:25s} | PPL: {ppl:6.2f} | Time: {t:6.2f} ms")

if __name__ == "__main__":
    main()