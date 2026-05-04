#!/usr/bin/env python3
"""
小型概念验证实验：自动训练并对比三种注意力模式
- Full Attention（全注意力基线）
- √Block Uniform（均匀采样）
- √Block Top‑K 范数（按 Key L2 范数选择代表）

用法：python run_small_exp.py
"""
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
TRAIN_SCRIPT = BASE / "train_advanced.py"
EVAL_SCRIPT = BASE / "eval_three.py"

# 通用训练参数
TRAIN_ARGS = [
    "--epochs", "2",
    "--batch_size", "8",
    "--seq_len", "256",
    "--d_model", "128",
    "--n_heads", "8",
    "--n_layers", "2",
    "--warmup_steps", "100"
]

def run(cmd, description):
    print(f"\n{'='*60}")
    print(f" {description}")
    print(f" Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, cwd=BASE)
    if result.returncode != 0:
        print(f"\n[ERROR] 步骤失败：{description}")
        sys.exit(result.returncode)

def main():
    # 全注意力基线
    run([sys.executable, str(TRAIN_SCRIPT), "--sampling", "full"] + TRAIN_ARGS,
        "1/3: Training Full Attention (baseline)")

    # √Block 均匀采样
    run([sys.executable, str(TRAIN_SCRIPT), "--sampling", "uniform"] + TRAIN_ARGS,
        "2/3: Training √Block Uniform Sampling")

    # √Block Top‑K 范数采样
    run([sys.executable, str(TRAIN_SCRIPT), "--sampling", "topk_norm"] + TRAIN_ARGS,
        "3/3: Training √Block Top‑K Norm Sampling")

    # 评估与对比
    run([sys.executable, str(EVAL_SCRIPT)],
        "Evaluation: comparing PPL and inference time")

    print("\n实验完成！请查看终端输出的对比表格。")

if __name__ == "__main__":
    main()