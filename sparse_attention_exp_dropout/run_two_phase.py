#!/usr/bin/env python3
"""
两阶段推进实验：
  Phase 1: 训练 Uniform 基线 (30 epochs, early stop, dropout=0.2)
  Phase 2: 基于基线分别微调 Uniform 和 TopK-Norm (各10 epochs, 带正则化)
  Phase 3: 评估对比 (使用 eval.py)
"""
import subprocess
import sys
import shutil
from pathlib import Path

BASE = Path(__file__).resolve().parent
TRAIN = BASE / "train_advanced.py"
EVAL = BASE / "eval.py"

# 通用 6 层参数
BASE_ARGS = [
    "--batch_size", "4",
    "--seq_len", "256",
    "--d_model", "256",
    "--n_heads", "8",
    "--n_layers", "6",
    "--warmup_steps", "500",
    "--patience", "5"               # 早停耐心值
]

def run(cmd, desc):
    print(f"\n{'='*60}\n {desc}\n{'='*60}")
    r = subprocess.run(cmd, cwd=BASE)
    if r.returncode != 0:
        print("ERROR, stopping.")
        sys.exit(1)

# Phase 1: 基线 Uniform (30 轮，dropout=0.2)
run([sys.executable, str(TRAIN), "--sampling", "uniform", "--epochs", "30",
     "--dropout", "0.2"] + BASE_ARGS,
    "Phase 1: Uniform baseline (30 ep, dropout=0.2)")

# 保存基线副本
baseline = BASE / "baseline_uniform.pt"
shutil.copy(BASE / "sparse_gpt_uniform_best.pt", baseline)

# Phase 2a: 微调 Uniform (10 轮，从基线开始)
run([sys.executable, str(TRAIN), "--sampling", "uniform", "--epochs", "10",
     "--dropout", "0.2", "--resume", str(baseline)] + BASE_ARGS,
    "Phase 2a: Fine-tune Uniform (10 ep)")

# Phase 2b: 微调 TopK-Norm (10 轮，从基线开始，更强正则化)
run([sys.executable, str(TRAIN), "--sampling", "topk_norm", "--epochs", "10",
     "--dropout", "0.3", "--weight_decay", "1e-3",
     "--resume", str(baseline)] + BASE_ARGS,
    "Phase 2b: Fine-tune TopK-Norm (10 ep, dropout=0.3, wd=1e-3)")

# Phase 3: 评估（eval.py 自动比较两个最佳模型）
run([sys.executable, str(EVAL)], "Phase 3: Final evaluation")

print("\n✅ Done. Check comparison.png and terminal output.")