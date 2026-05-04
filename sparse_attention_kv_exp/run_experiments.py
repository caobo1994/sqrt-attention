#!/usr/bin/env python3
"""
√Block KV Cache 压缩实验 — Runner

用法：
    python run_experiments.py                   # 完整实验
    python run_experiments.py --quick           # 快速测试
    python run_experiments.py --model gpt2-medium  # 更大模型
    python run_experiments.py --visualize       # 生成图表
"""

import subprocess
import sys
import re
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
MAIN = BASE / "kv_cache_experiment.py"

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
        segments = text.split('\r')
        for seg in segments:
            if '\n' in seg:
                parts = seg.split('\n')
                for part in parts:
                    if not part.strip():
                        continue
                    line = part.rstrip()
                    print(f"  {line}")
                    collected.append(line)
                pending = ''
            else:
                pending = seg.rstrip()

    if pending and pending.strip():
        print(f"  {pending}")
        collected.append(pending)

    process.wait()
    elapsed = time.time() - start
    mins, secs = divmod(elapsed, 60)

    with open(log_path, 'w', encoding='utf-8') as f:
        for line in collected:
            clean = re.sub(r'\033\[[0-9;]*m', '', line)
            f.write(clean + '\n')

    status = f"{GREEN}✓ Done{RESET}" if process.returncode == 0 else f"{RED}✗ Failed{RESET}"
    print(f"\n  {status} in {int(mins)}m {secs:.0f}s")
    print(f"  Log: {log_path.name} ({log_path.stat().st_size:,} bytes)")
    return process.returncode


def main():
    quick = '--quick' in sys.argv
    viz = '--visualize' in sys.argv
    model = '--model' in sys.argv
    model_arg = ''
    if model:
        idx = sys.argv.index('--model')
        if idx + 1 < len(sys.argv):
            model_arg = f'--model {sys.argv[idx + 1]}'

    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  √Block KV Cache Compression Experiment{RESET}")
    print(f"{BOLD}{'='*60}{RESET}\n")

    cmd = [sys.executable, str(MAIN)]
    if quick:
        cmd.append('--quick')
    if viz:
        cmd.append('--visualize')

    rc = run_with_clean_log(
        cmd,
        BASE / f"kv_cache_results_{'quick' if quick else 'full'}.log",
        f"Running KV Cache experiment ({'quick' if quick else 'full'})"
    )

    if rc == 0:
        print(f"\n{GREEN}✓ Experiment complete!{RESET}")
        print(f"  Check kv_cache_results.json for detailed results.\n")

    return rc


if __name__ == "__main__":
    sys.exit(main())
