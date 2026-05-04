#!/usr/bin/env python3
"""
WikiText-103 中型模型全自动实验流水线

目的：在更大数据集（WikiText-103, 1.03 亿 token）上验证 √Block Attention 的可扩展性。
模型：d_model=512, 8 heads, 8 layers (~ 30M 参数)

流水线：
  Phase 1: 训练 Uniform 基线 (50 epochs, warmup=1000)
  Phase 2: 继续微调 Uniform (10 epochs, 从基线恢复)
  Phase 3: 微调 TopK-Norm (10 epochs, 从基线恢复，切换采样策略)
  Phase 4: 评估对比（PPL + 推理时间）
"""
import subprocess
import sys
import shutil
from pathlib import Path

BASE = Path(__file__).resolve().parent
TRAIN_SCRIPT = BASE / "train_advanced.py"
EVAL_SCRIPT = BASE / "eval.py"
BASELINE_CKPT = BASE / "sparse_gpt_uniform_wt103_best.pt"
BASELINE_BACKUP = BASE / "baseline_uniform_wt103.pt"

# 中型模型参数
MODEL_ARGS = [
    "--d_model", "512",
    "--n_heads", "8",
    "--n_layers", "8",
    "--batch_size", "4",
    "--seq_len", "256",
    "--lr", "1e-4",
    "--grad_clip", "1.0",
]


def run(cmd, description):
    """运行命令，失败时终止"""
    print(f"\n{'='*60}")
    print(f" {description}")
    print(f" Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, cwd=BASE)
    if result.returncode != 0:
        print(f"\n[ERROR] Step failed: {description}")
        sys.exit(result.returncode)


def main():
    # ---------- Phase 1: 基线训练 (Uniform, 50 epochs) ----------
    run([
        sys.executable, str(TRAIN_SCRIPT),
        "--sampling", "uniform",
        "--epochs", "50",
        "--warmup_steps", "1000",
        "--patience", "10",
        "--log_interval", "50",
    ] + MODEL_ARGS,
        "Phase 1: Training Uniform Baseline on WikiText-103 (50 epochs)")

    # 备份基线检查点
    if BASELINE_CKPT.exists():
        shutil.copy(BASELINE_CKPT, BASELINE_BACKUP)
        print(f"Baseline checkpoint saved as {BASELINE_BACKUP}")
    else:
        print(f"Warning: Baseline checkpoint not found at {BASELINE_CKPT}")

    # ---------- Phase 2: 微调 Uniform (10 epochs) ----------
    run([
        sys.executable, str(TRAIN_SCRIPT),
        "--sampling", "uniform",
        "--epochs", "10",
        "--warmup_steps", "500",
        "--resume", str(BASELINE_BACKUP),
        "--lr", "5e-5",  # 微调降低学习率
    ] + MODEL_ARGS,
        "Phase 2: Fine-tuning Uniform (10 epochs)")

    # ---------- Phase 3: 微调 TopK-Norm (10 epochs) ----------
    run([
        sys.executable, str(TRAIN_SCRIPT),
        "--sampling", "topk_norm",
        "--epochs", "10",
        "--warmup_steps", "500",
        "--resume", str(BASELINE_BACKUP),
        "--lr", "5e-5",
    ] + MODEL_ARGS,
        "Phase 3: Fine-tuning TopK-Norm (10 epochs)")

    # ---------- Phase 4: 评估对比 ----------
    run([
        sys.executable, str(EVAL_SCRIPT),
        "--uniform_ckpt", "sparse_gpt_uniform_wt103_best.pt",
        "--topk_ckpt", "sparse_gpt_topk_norm_wt103_best.pt",
        "--d_model", "512",
        "--n_heads", "8",
        "--n_layers", "8",
    ], "Phase 4: Evaluation & Comparison")

    print("\n" + "="*60)
    print(" WikiText-103 medium experiment complete!")
    print("="*60)


if __name__ == "__main__":
    main()
