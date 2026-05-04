#!/usr/bin/env python3
"""
全自动稀疏注意力对比实验 (基线训练 + 微调 + 评估)
用法: python run_full_exp.py
"""
import subprocess
import sys
import shutil
from pathlib import Path

BASE = Path(__file__).resolve().parent
TRAIN_SCRIPT = BASE / "train_advanced.py"
EVAL_SCRIPT = BASE / "eval.py"
BASELINE_CKPT = BASE / "sparse_gpt_uniform_best.pt"
BASELINE_BACKUP = BASE / "baseline_uniform.pt"

def run(cmd, description):
    """运行命令，失败时终止整个脚本"""
    print(f"\n{'='*60}")
    print(f" {description}")
    print(f" Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, cwd=BASE)
    if result.returncode != 0:
        print(f"\n[ERROR] Step failed: {description}")
        sys.exit(result.returncode)

def main():
    # ---------- Phase 1: 基线训练 (Uniform, 30 epochs) ----------
    run([
        sys.executable, str(TRAIN_SCRIPT),
        "--sampling", "uniform",
        "--epochs", "30",
        "--warmup_steps", "500",
        "--batch_size", "4",
        "--seq_len", "256"
    ], "Phase 1: Training Baseline (Uniform, 30 epochs)")

    # 复制基线检查点
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
        "--warmup_steps", "200",
        "--resume", str(BASELINE_BACKUP),
        "--batch_size", "4",
        "--seq_len", "256"
    ], "Phase 2: Fine-tuning Uniform (10 epochs, from baseline)")

    # ---------- Phase 3: 微调 TopK-Norm (10 epochs) ----------
    run([
        sys.executable, str(TRAIN_SCRIPT),
        "--sampling", "topk_norm",
        "--epochs", "10",
        "--warmup_steps", "200",
        "--resume", str(BASELINE_BACKUP),
        "--batch_size", "4",
        "--seq_len", "256"
    ], "Phase 3: Fine-tuning TopK-Norm (10 epochs, from baseline)")

    # ---------- Phase 4: 评估 ----------
    run([sys.executable, str(EVAL_SCRIPT)], "Phase 4: Evaluation & Comparison")

    print("\nAll done! Results saved to comparison.png")

if __name__ == "__main__":
    main()
