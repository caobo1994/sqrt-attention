#!/usr/bin/env python3
"""
任意交错采样 — 批量探索 Runner

按配置字符串列表依次训练，自动复用已有检查点。

用法：
    python run_experiments.py --configs UTUTUT,TUUTUU,UUUTTT
    python run_experiments.py --configs UTUTUT --epochs 10
    python run_experiments.py --all-alternates    # 所有 6 层 U/T 交替模式
    python run_experiments.py --summary           # 汇总所有已有结果
"""

import subprocess
import sys
import re
import json
import itertools
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
TRAIN = BASE / "train.py"

GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
CYAN = '\033[96m'
BOLD = '\033[1m'
RESET = '\033[0m'


def parse_configs(args):
    """解析配置列表。"""
    if '--all-alternates' in args:
        # 从第一个 --epochs 或默认 6 层推断
        n = 6
        for i, a in enumerate(args):
            if a == '--n' and i + 1 < len(args):
                n = int(args[i + 1])
                break
        return [''.join(p) for p in itertools.product('UT', repeat=n)]
    for i, a in enumerate(args):
        if a == '--configs' and i + 1 < len(args):
            return args[i + 1].split(',')
    return ['UTUTUT']  # default


def run_single(config_str, epochs, extra_args=''):
    """训练单个配置，带 clean logging。"""
    cmd = [sys.executable, str(TRAIN), '--config', config_str, '--epochs', str(epochs)]
    if extra_args:
        cmd.extend(extra_args.split())

    log_path = BASE / f"log_{config_str}.log"
    print(f"\n  {BOLD}[{config_str}]{RESET} {' '.join(cmd[3:])}")

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
                    clean_line = line; print(f"    {clean_line}")
                    collected.append(line)
    process.wait()
    elapsed = time.time() - start

    with open(log_path, 'w', encoding='utf-8') as f:
        for line in collected:
            f.write(re.sub(r'\033\[[0-9;]*m', '', line) + '\n')

    mins, secs = divmod(elapsed, 60)
    status = f"{GREEN}✓{RESET}" if process.returncode == 0 else f"{RED}✗{RESET}"
    best = re.search(r'Best PPL.*?([\d.]+)', log_path.read_text('utf-8'))
    best_str = f" PPL={GREEN}{best.group(1)}{RESET}" if best else ""
    print(f"  {status} {int(mins)}m {secs:.0f}s{best_str}")
    return process.returncode


def print_summary():
    """汇总 train_log.json 中的所有结果。"""
    log_path = BASE / "train_log.json"
    if not log_path.exists():
        print(f"  No results yet. Run some experiments first.")
        return

    log = json.loads(log_path.read_text())
    n_layers = max(len(k) for k in log) if log else 0

    print(f"\n{BOLD}{'='*70}{RESET}")
    print(f"{BOLD}  Custom Alternating — Results Summary{RESET}")
    print(f"{BOLD}{'='*70}{RESET}")
    print(f"\n{'Config':<10} {'Pattern':<30} {'Epochs':<8} {'Best PPL':<10}")
    print(f"{'-'*58}")

    # Sort by PPL
    sorted_configs = sorted(log.items(), key=lambda x: x[1].get('best_ppl', float('inf')))

    for config_str, entry in sorted_configs:
        readable = ''.join('U' if c == 'U' else 'T' for c in config_str.upper())
        pattern = ' | '.join(f"L{i}={c}" for i, c in enumerate(readable))
        epochs = entry.get('epochs_completed', '?')
        ppl = entry.get('best_ppl', '?')
        best_flag = " ★" if ppl == min(e.get('best_ppl', float('inf')) for e in log.values()) else ""
        print(f"  {config_str:<8} {pattern:<30} {epochs:<8} {ppl:<10}{best_flag}")

    print(f"{'-'*58}")
    winner = min(log.items(), key=lambda x: x[1].get('best_ppl', float('inf')))
    print(f"\n  Best config: {winner[0]} (PPL={winner[1]['best_ppl']})")


def main():
    args = sys.argv[1:]

    if '--summary' in args:
        print_summary()
        return 0
    if '--clean' in args:
        for f in BASE.glob('log_*.log'):
            f.unlink()
        print("Logs cleaned.")
        return 0

    configs = parse_configs(args)
    epochs = 10
    extra = ''

    for i, a in enumerate(args):
        if a == '--epochs' and i + 1 < len(args):
            epochs = int(args[i + 1])
        if a == '--resume':
            extra = '--resume'

    # 去掉 --configs 和配置列表，避免传给 train.py
    clean_args = [a for a in args if a != '--all-alternates' and not a.startswith('--configs')]
    # keep epochs and resume
    extra = ' '.join(a for a in clean_args if a in ('--resume',) or a.startswith('--epochs'))

    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  √Block Custom Alternating — Batch Runner{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")
    print(f"  Configs: {len(configs)} ({', '.join(configs[:5])}{'...' if len(configs) > 5 else ''})")
    print(f"  Epochs per config: {epochs}")
    print(f"  Resume: {'yes' if '--resume' in extra else 'no'}")

    for config_str in configs:
        rc = run_single(config_str, epochs, extra)
        if rc != 0:
            print(f"\n{RED}Failed at {config_str}, stopping.{RESET}")
            return 1

    print_summary()
    print(f"\n{GREEN}✓ All done!{RESET}")
    return 0


if __name__ == "__main__":
    main()
