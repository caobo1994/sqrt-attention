#!/usr/bin/env python3
"""
√Block Attention — 小型实验 Runner (2层, d_model=128)

训练并对比三种注意力模式：
  1. Full Attention（全注意力基线）
  2. √Block Uniform（均匀采样）
  3. √Block TopK-Norm（按 Key L2 范数选择代表）

Usage:
    python run_experiments.py              # 完整流水线
    python run_experiments.py --quick      # 1 epoch 快速验证
    python run_experiments.py --clean-logs # 清理日志
"""

import subprocess
import sys
import re
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
TRAIN = BASE / "train_advanced.py"
EVAL = BASE / "eval_three.py"

GREEN = '\033[92m'
RED = '\033[91m'
BLUE = '\033[94m'
BOLD = '\033[1m'
RESET = '\033[0m'

BASE_ARGS = [
    "--batch_size", "8",
    "--seq_len", "256",
    "--d_model", "128",
    "--n_heads", "8",
    "--n_layers", "2",
    "--warmup_steps", "100",
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
                    if pending and j == 0:
                        pass  # final tqdm state, already shown
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
    print(f"{BOLD}  √Block Attention — Small Experiment{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")
    print(f"  Mode: {'Quick' if quick else 'Full'}{' (clean only)' if clean_only else ''}\n")

    if clean_only:
        for f in ['full.log', 'uniform.log', 'topk_norm.log',
                  'full_1.log', 'uniform_1.log', 'topk_norm_1.log']:
            p = BASE / f
            if p.exists():
                p.unlink()
                print(f"  Removed: {f}")
        print(f"\n{GREEN}✓ Clean{RESET}")
        return 0

    epochs = 1 if quick else 2

    # Phase 1: Full Attention
    rc = run_with_clean_log(
        [sys.executable, str(TRAIN), "--sampling", "full",
         "--epochs", str(epochs)] + BASE_ARGS,
        BASE / f"full_{epochs}.log",
        f"Phase 1/3: Full Attention ({epochs} epochs)")
    if rc != 0:
        return 1

    # Phase 2: Uniform
    rc = run_with_clean_log(
        [sys.executable, str(TRAIN), "--sampling", "uniform",
         "--epochs", str(epochs)] + BASE_ARGS,
        BASE / f"uniform_{epochs}.log",
        f"Phase 2/3: √Block Uniform ({epochs} epochs)")
    if rc != 0:
        return 1

    # Phase 3: TopK-Norm
    rc = run_with_clean_log(
        [sys.executable, str(TRAIN), "--sampling", "topk_norm",
         "--epochs", str(epochs)] + BASE_ARGS,
        BASE / f"topk_norm_{epochs}.log",
        f"Phase 3/3: √Block TopK-Norm ({epochs} epochs)")
    if rc != 0:
        return 1

    print(f"\n{BOLD}{GREEN}{'='*60}{RESET}")
    print(f"{BOLD}{GREEN} ✓ Training complete!{RESET}")
    print(f"{BOLD}{GREEN}{'='*60}{RESET}")
    print(f"\n  Run: python eval_three.py")
    print(f"  To compare PPL and inference time.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
