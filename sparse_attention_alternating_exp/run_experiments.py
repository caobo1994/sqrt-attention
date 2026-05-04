#!/usr/bin/env python3
"""
√Block Alternating Sampling — 层间交替采样实验 Runner

按层交替 uniform / topk_norm 采样策略，对比三种配置：
  1. uniform-all: 所有层 uniform
  2. topk-all:    所有层 topk_norm
  3. alternate:   even=uniform, odd=topk_norm
  4. alternate_rev: even=topk_norm, odd=uniform

核心假设：
  - 浅层 uniform 覆盖面广，提取基础特征
  - 深层 topk 聚焦重要 token
  - 交替可能优于单一策略

用法：
    python run_experiments.py              # 完整 10 轮
    python run_experiments.py --quick      # 2 轮快速测试
    python run_experiments.py --clean-logs # 清理日志
"""

import subprocess
import sys
import re
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
TRAIN = BASE / "train.py"
EVAL = BASE / "evaluate.py"

GREEN = '\033[92m'
RED = '\033[91m'
BLUE = '\033[94m'
BOLD = '\033[1m'
RESET = '\033[0m'


def run_with_clean_log(cmd, log_path, description):
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
    while True:
        raw = process.stdout.read(8192)
        if not raw:
            break
        text = raw.decode('utf-8', errors='replace')
        for seg in text.split('\r'):
            if '\n' in seg:
                for part in seg.split('\n'):
                    if not part.strip():
                        continue
                    line = part.rstrip()
                    print(f"  {line}")
                    collected.append(line)
            else:
                pending = seg.rstrip()

    process.wait()
    elapsed = time.time() - start

    with open(log_path, 'w', encoding='utf-8') as f:
        for line in collected:
            clean = re.sub(r'\033\[[0-9;]*m', '', line)
            f.write(clean + '\n')

    mins, secs = divmod(elapsed, 60)
    status = f"{GREEN}✓ Done{RESET}" if process.returncode == 0 else f"{RED}✗ Failed{RESET}"
    print(f"\n  {status} in {int(mins)}m {secs:.0f}s")
    best = re.search(r'Best PPL.*?([\d.]+)', log_path.read_text('utf-8'))
    if best:
        print(f"  Best PPL: {GREEN}{best.group(1)}{RESET}")
    return process.returncode


def main():
    quick = '--quick' in sys.argv
    clean_only = '--clean-logs' in sys.argv

    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  √Block Alternating Sampling Experiment{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")
    print(f"  Testing: uniform-only, topk-only, alternate (U/T/U/T...), alternate_rev (T/U/T/U...)\n")

    if clean_only:
        for f in BASE.glob('*.log'):
            f.unlink()
            print(f"  Removed: {f.name}")
        print(f"\n{GREEN}✓ Clean{RESET}")
        return 0

    epochs_arg = ['--epochs', '2'] if quick else ['--epochs', '10']

    # Phase 1: Uniform-only
    rc = run_with_clean_log(
        [sys.executable, str(TRAIN), '--config', 'uniform'] + epochs_arg,
        BASE / f"uniform_{'quick' if quick else 'full'}.log",
        f"Phase 1/4: Uniform-Only ({'2' if quick else '10'} epochs)")
    if rc != 0:
        return 1

    # Phase 2: TopK-only
    rc = run_with_clean_log(
        [sys.executable, str(TRAIN), '--config', 'topk_norm'] + epochs_arg,
        BASE / f"topk_{'quick' if quick else 'full'}.log",
        f"Phase 2/4: TopK-Only ({'2' if quick else '10'} epochs)")
    if rc != 0:
        return 1

    # Phase 3: Alternate (even=U, odd=T)
    rc = run_with_clean_log(
        [sys.executable, str(TRAIN), '--config', 'alternate'] + epochs_arg,
        BASE / f"alternate_{'quick' if quick else 'full'}.log",
        f"Phase 3/4: Alternate (U/T/U/T...) ({'2' if quick else '10'} epochs)")
    if rc != 0:
        return 1

    # Phase 4: Alternate Rev (even=T, odd=U)
    rc = run_with_clean_log(
        [sys.executable, str(TRAIN), '--config', 'alternate_rev'] + epochs_arg,
        BASE / f"alternate_rev_{'quick' if quick else 'full'}.log",
        f"Phase 4/4: Alternate Rev (T/U/T/U...) ({'2' if quick else '10'} epochs)")
    if rc != 0:
        return 1

    # Summary
    print(f"\n{BOLD}{GREEN}{'='*60}{RESET}")
    print(f"{BOLD}{GREEN} ✓ All experiments complete!{RESET}")
    print(f"{BOLD}{GREEN}{'='*60}{RESET}")
    print(f"\n  Run: python evaluate.py")
    print(f"  To compare all 4 configurations.\n")


if __name__ == "__main__":
    main()
