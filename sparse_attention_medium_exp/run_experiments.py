#!/usr/bin/env python3
"""
√Block Attention — 中型实验 Runner (8层, d_model=512, WikiText-103)

在 WikiText-103 上验证 √Block Attention 的可扩展性。
同主实验的流水线：Uniform 基线 → Uniform 微调 → TopK-Norm 微调

Usage:
    python run_experiments.py              # 完整流水线 (50+10+10)
    python run_experiments.py --quick      # 快速验证 (5+2+2)
    python run_experiments.py --clean-logs # 清理日志
"""

import subprocess
import sys
import re
import shutil
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
TRAIN = BASE / "train_advanced.py"
EVAL = BASE / "eval.py"

GREEN = '\033[92m'
RED = '\033[91m'
BLUE = '\033[94m'
BOLD = '\033[1m'
RESET = '\033[0m'

MODEL_ARGS = [
    "--d_model", "512",
    "--n_heads", "8",
    "--n_layers", "8",
    "--batch_size", "4",
    "--seq_len", "256",
]


def run_with_clean_log(cmd, log_path, description):
    """Run command with clean output + log (handles tqdm \\r noise)."""
    print(f"\n{BOLD}{BLUE}{'='*60}{RESET}")
    print(f"{BOLD}{BLUE} {description}{RESET}")
    print(f"{BOLD}{BLUE}{'='*60}{RESET}\n")
    print(f"  {' '.join(cmd)}\n")

    start = time.time()
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        bufsize=0, text=False, cwd=BASE,
    )

    pending = ''
    collected = []
    chunk_size = 8192

    while True:
        raw = process.stdout.read(chunk_size)
        if not raw:
            break
        text = raw.decode('utf-8', errors='replace')
        segments = text.split('\r')
        for seg in segments:
            if '\n' in seg:
                parts = seg.split('\n')
                for j, part in enumerate(parts):
                    if not part:
                        continue
                    line = part.rstrip()
                    print(f"  {line}")
                    collected.append(line)
                pending = parts[-1] if parts[-1] else ''
            else:
                pending = seg.rstrip()

    if pending and pending.strip():
        print(f"  {pending}")
        collected.append(pending)

    process.wait()
    elapsed = time.time() - start

    with open(log_path, 'w', encoding='utf-8') as f:
        for line in collected:
            clean = re.sub(r'\033\[[0-9;]*m', '', line)
            f.write(clean + '\n')

    mins, secs = divmod(elapsed, 60)
    status = f"{GREEN}✓ Done{RESET}" if process.returncode == 0 else f"{RED}✗ Failed{RESET}"
    print(f"\n  {status} in {int(mins)}m {secs:.0f}s")
    best = re.search(r'Best Perplexity.*?([\d.]+)', log_path.read_text('utf-8'))
    if best:
        print(f"  Best PPL: {GREEN}{best.group(1)}{RESET}")
    print(f"  Log: {log_path.name} ({log_path.stat().st_size:,} bytes)")
    return process.returncode


def main():
    quick = '--quick' in sys.argv
    clean_only = '--clean-logs' in sys.argv

    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  √Block Attention — Medium Experiment (WikiText-103){RESET}")
    print(f"{BOLD}{'='*60}{RESET}")
    print(f"  Mode: {'Quick' if quick else 'Full'}{' (clean only)' if clean_only else ''}\n")

    if clean_only:
        for log_name in ['uniform_wt103.log', 'topk_norm_wt103.log',
                         'finetune_uniform_wt103.log', 'finetune_topk_norm_wt103.log']:
            p = BASE / log_name
            if p.exists():
                p.unlink()
                print(f"  Removed: {log_name}")
        print(f"\n{GREEN}✓ Clean{RESET}")
        return 0

    # Phase 1: Uniform baseline (50 epochs / 5 quick)
    ep1 = 5 if quick else 50
    rc = run_with_clean_log(
        [sys.executable, str(TRAIN), "--sampling", "uniform",
         "--epochs", str(ep1), "--warmup_steps", "1000",
         "--lr", "1e-4", "--patience", "10", "--grad_clip", "1.0"] + MODEL_ARGS,
        BASE / f"uniform_wt103_{ep1}.log",
        f"Phase 1/3: Uniform Baseline ({ep1} epochs, d_model=512, 8 layers)")
    if rc != 0:
        return 1

    # Backup baseline checkpoint (with wt103 prefix)
    best_uniform = BASE / "sparse_gpt_uniform_wt103_best.pt"
    baseline = BASE / "baseline_uniform_wt103.pt"
    if best_uniform.exists():
        shutil.copy(best_uniform, baseline)
        print(f"  Baseline: {baseline}")

    # Also check for the simple-named checkpoint (train.py may have saved without wt103)
    simple_ckpt = BASE / "sparse_gpt_uniform_best.pt"
    if simple_ckpt.exists() and not best_uniform.exists():
        shutil.copy(simple_ckpt, baseline)
        print(f"  Baseline (from simple name): {baseline}")

    # Phase 2: Fine-tune Uniform (10 epochs / 2 quick)
    ep2 = 2 if quick else 10
    rc = run_with_clean_log(
        [sys.executable, str(TRAIN), "--sampling", "uniform",
         "--epochs", str(ep2), "--warmup_steps", "500",
         "--lr", "5e-5", "--resume", str(baseline)] + MODEL_ARGS,
        BASE / f"finetune_uniform_wt103_{ep2}.log",
        f"Phase 2/3: Fine-tune Uniform ({ep2} epochs)")
    if rc != 0:
        return 1

    # Phase 3: Fine-tune TopK-Norm (10 epochs / 2 quick)
    rc = run_with_clean_log(
        [sys.executable, str(TRAIN), "--sampling", "topk_norm",
         "--epochs", str(ep2), "--warmup_steps", "500",
         "--lr", "5e-5", "--resume", str(baseline)] + MODEL_ARGS,
        BASE / f"finetune_topk_norm_wt103_{ep2}.log",
        f"Phase 3/3: Fine-tune TopK-Norm ({ep2} epochs)")
    if rc != 0:
        return 1

    print(f"\n{BOLD}{GREEN}{'='*60}{RESET}")
    print(f"{BOLD}{GREEN} ✓ All experiments complete!{RESET}")
    print(f"{BOLD}{GREEN}{'='*60}{RESET}")
    print(f"\n  Run: python eval.py \\")
    print(f"    --uniform_ckpt sparse_gpt_uniform_wt103_best.pt \\")
    print(f"    --topk_ckpt sparse_gpt_topk_norm_wt103_best.pt \\")
    print(f"    --d_model 512 --n_heads 8 --n_layers 8")
    print(f"  To compare models.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
