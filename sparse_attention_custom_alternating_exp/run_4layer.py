#!/usr/bin/env python3
"""
4 层 √Block 对比实验 — 一键跑完四种配置

采样策略：
  1. UUUU  → 全部 uniform（基线）
  2. TTTT  → 全部 topk_norm（基线）
  3. UTUT  → alternate (uniform/topk 交替)
  4. TUTU  → alternate_rev (topk/uniform 交替)

用法：
    python run_4layer.py                 # 4 种配置各 10 轮
    python run_4layer.py --quick         # 各 2 轮快速测试
    python run_4layer.py --epochs 30     # 各 30 轮
    python run_4layer.py --resume        # 从已有检查点恢复
    python run_4layer.py --summary       # 汇总已有结果
"""

import subprocess
import sys
import re
import json
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
TRAIN = BASE / "train.py"

GREEN = '\033[92m'
RED = '\033[91m'
BLUE = '\033[94m'
BOLD = '\033[1m'
RESET = '\033[0m'

# 四种 4 层配置
CONFIGS = [
    ('UUUU', '全部 Uniform'),
    ('TTTT', '全部 TopK-Norm'),
    ('UTUT', 'Alternate (U/T/U/T)'),
    ('TUTU', 'Alternate Rev (T/U/T/U)'),
]


def run_with_clean_log(cmd, log_path, description):
    print(f"\n{BOLD}{BLUE}{'='*60}{RESET}")
    print(f"{BOLD}{BLUE}  {description}{RESET}")
    print(f"{BOLD}{BLUE}{'='*60}{RESET}\n")

    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        bufsize=0, text=False, cwd=BASE,
    )
    collected = []
    while True:
        raw = process.stdout.read(8192)
        if not raw:
            break
        for seg in raw.decode('utf-8', errors='replace').split('\r'):
            if '\n' in seg:
                for part in seg.split('\n'):
                    if not part.strip():
                        continue
                    line = part.rstrip()
                    cl = re.sub(r'\033\[[0-9;]*m', '', line); print(f'  {cl}')
                    collected.append(line)
    process.wait()

    with open(log_path, 'w', encoding='utf-8') as f:
        for line in collected:
            cl = re.sub(r'\033\[[0-9;]*m', '', line); f.write(cl + '\n')

    best = re.search(r'Best PPL.*?([\d.]+)', ''.join(collected))
    best_str = f" → PPL={GREEN}{best.group(1)}{RESET}" if best else ""
    status = f"{GREEN}✓{RESET}" if process.returncode == 0 else f"{RED}✗{RESET}"
    print(f"  {status}{best_str}")


def main():
    args = sys.argv[1:]

    if '--summary' in args:
        log_path = BASE / "train_log.json"
        if not log_path.exists():
            print("No results yet.")
            return
        log = json.loads(log_path.read_text())
        print(f"\n{BOLD}{'='*50}{RESET}")
        print(f"{BOLD}  4-Layer Alternating — Summary{RESET}")
        print(f"{BOLD}{'='*50}{RESET}")
        print(f"\n{'Config':<8} {'Pattern':<30} {'Best PPL':<10}")
        print(f"{'-'*48}")
        best_all = min((e.get('best_ppl', float('inf')) for e in log.values()), default=float('inf'))
        for cfg, _ in CONFIGS:
            entry = log.get(cfg, {})
            ppl = entry.get('best_ppl', '-')
            star = ' ★' if ppl == best_all and ppl != '-' else ''
            print(f"  {cfg:<6} {cfg:<30} {ppl:<10}{star}")
        return

    quick = '--quick' in args
    resume = '--resume' in args
    epochs = 2 if quick else 10
    extra = '--resume' if resume else ''

    # 从参数中提取 epochs
    for i, a in enumerate(args):
        if a == '--epochs' and i + 1 < len(args):
            epochs = int(args[i + 1])

    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  4-Layer √Block Alternating Experiment{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")
    print(f"  Epochs per config: {epochs}")
    print(f"  Configs: {len(CONFIGS)}")
    print(f"  Resume: {'yes' if resume else 'no'}\n")

    for cfg, desc in CONFIGS:
        cmd = [sys.executable, str(TRAIN), '--config', cfg, '--epochs', str(epochs)]
        if resume:
            cmd.append('--resume')
        run_with_clean_log(
            cmd, BASE / f"log_{cfg}.log",
            f"{cfg} — {desc}"
        )

    print(f"\n{GREEN}{'='*50}{RESET}")
    print(f"{GREEN}  All done! Run with --summary to compare results.{RESET}")
    print(f"{GREEN}{'='*50}{RESET}")


if __name__ == "__main__":
    main()
