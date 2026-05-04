#!/usr/bin/env python3
"""
√Block Attention — Experiment Runner with Clean Logging

Problem: When run interactively, tqdm uses \\r (carriage return) to update
the progress bar in-place. Via subprocess.PIPE, these accumulate into 
giant single lines (~230KB each tqdm epoch).

Solution: Read stdout in binary chunks, split on \\r in real-time,
show only the last (final) state of each progress bar, and save clean logs.

Usage:
    python run_experiments.py              # Full 30+10+10 pipeline
    python run_experiments.py --quick      # Quick 5+2+2 test
    python run_experiments.py --clean-logs # Reset logs
"""

import subprocess
import sys
import re
import shutil
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
TRAIN = BASE / "train_advanced.py"

GREEN = '\033[92m'
RED = '\033[91m'
BLUE = '\033[94m'
GRAY = '\033[90m'
BOLD = '\033[1m'
RESET = '\033[0m'


def run_with_clean_log(cmd, log_path, description):
    """
    Run command with real-time output display + clean log file.
    
    Binary reading approach handles tqdm's \\r correctly:
      - On \\r: accumulate into a buffer (progress bar update)
      - On \\n: flush the buffer as a clean line + newline part
      - At EOF: flush remaining buffer
    """
    print(f"\n{BOLD}{BLUE}{'='*60}{RESET}")
    print(f"{BOLD}{BLUE} {description}{RESET}")
    print(f"{BOLD}{BLUE}{'='*60}{RESET}\n")
    print(f"  {GRAY}{' '.join(cmd)}{RESET}\n")

    start = time.time()

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,       # unbuffered
        text=False,      # binary mode for \\r handling
        cwd=BASE,
    )

    pending = ''  # current tqdm progress bar content
    collected = []  # clean output lines

    chunk_size = 8192
    while True:
        raw = process.stdout.read(chunk_size)
        if not raw:
            break

        text = raw.decode('utf-8', errors='replace')

        # Split on \r: each segment is a progress bar update
        segments = text.split('\r')
        for i, seg in enumerate(segments):
            if '\n' in seg:
                # Newline found: flush pending + write this segment
                parts = seg.split('\n')
                for j, part in enumerate(parts):
                    if not part:
                        continue
                    # The part before \n is final tqdm state or log line
                    line = part.rstrip()
                    if pending and j == 0:
                        # This was the final state of the progress bar
                        pass
                    print(f"  {line}")
                    collected.append(line)
                pending = parts[-1] if parts[-1] else ''
            else:
                # \r-separated tqdm update: replace previous state
                pending = seg.rstrip()

    # Flush remaining pending content
    if pending and pending.strip():
        print(f"  {pending}")
        collected.append(pending)

    process.wait()
    elapsed = time.time() - start

    # Write clean log (strip ANSI escape codes)
    with open(log_path, 'w', encoding='utf-8') as f:
        for line in collected:
            clean = re.sub(r'\033\[[0-9;]*m', '', line)
            f.write(clean + '\n')

    # Status
    mins, secs = divmod(elapsed, 60)
    status = f"{GREEN}✓ Done{RESET}" if process.returncode == 0 else f"{RED}✗ Failed{RESET}"
    print(f"\n  {status} in {int(mins)}m {secs:.0f}s")

    # Best PPL
    text = log_path.read_text(encoding='utf-8')
    best = re.search(r'Best Perplexity.*?([\d.]+)', text)
    if best:
        print(f"  Best PPL: {GREEN}{best.group(1)}{RESET}")

    print(f"  Log: {log_path.name} ({log_path.stat().st_size:,} bytes)")
    return process.returncode


def main():
    quick = '--quick' in sys.argv
    clean_only = '--clean-logs' in sys.argv

    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  √Block Attention — Experiment Runner{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")
    print(f"  Mode: {'Quick' if quick else 'Full'}{' (clean only)' if clean_only else ''}\n")

    if clean_only:
        for log_name in ['uniform_30.log', 'topk_30.log',
                         'uniform_5.log', 'topk_5.log',
                         'finetune_uniform_10.log', 'finetune_uniform_2.log',
                         'finetune_topk_norm_10.log', 'finetune_topk_norm_2.log']:
            p = BASE / log_name
            if p.exists():
                p.unlink()
                print(f"  Removed: {log_name}")
        print(f"\n{GREEN}✓ Clean{RESET}")
        return 0

    # Phase 1: Uniform baseline
    ep1 = 5 if quick else 30
    rc = run_with_clean_log(
        [sys.executable, str(TRAIN), "--sampling", "uniform",
         "--epochs", str(ep1), "--warmup_steps", "500",
         "--batch_size", "4", "--seq_len", "256"],
        BASE / f"uniform_{ep1}.log",
        f"Phase 1/3: Uniform Baseline ({ep1} epochs)")
    if rc != 0:
        return 1

    # Backup baseline checkpoint
    best_uniform = BASE / "sparse_gpt_uniform_best.pt"
    baseline = BASE / "baseline_uniform.pt"
    if best_uniform.exists():
        shutil.copy(best_uniform, baseline)
        print(f"  Baseline: {baseline}")

    # Phase 2: Fine-tune Uniform
    ep2 = 2 if quick else 10
    rc = run_with_clean_log(
        [sys.executable, str(TRAIN), "--sampling", "uniform",
         "--epochs", str(ep2), "--warmup_steps", "200",
         "--resume", str(baseline),
         "--batch_size", "4", "--seq_len", "256"],
        BASE / f"finetune_uniform_{ep2}.log",
        f"Phase 2/3: Fine-tune Uniform ({ep2} epochs)")
    if rc != 0:
        return 1

    # Phase 3: Fine-tune TopK-Norm
    rc = run_with_clean_log(
        [sys.executable, str(TRAIN), "--sampling", "topk_norm",
         "--epochs", str(ep2), "--warmup_steps", "200",
         "--resume", str(baseline),
         "--batch_size", "4", "--seq_len", "256"],
        BASE / f"finetune_topk_norm_{ep2}.log",
        f"Phase 3/3: Fine-tune TopK-Norm ({ep2} epochs)")
    if rc != 0:
        return 1

    print(f"\n{BOLD}{GREEN}{'='*60}{RESET}")
    print(f"{BOLD}{GREEN} ✓ All experiments complete!{RESET}")
    print(f"{BOLD}{GREEN}{'='*60}{RESET}")
    print(f"\n  Run `python eval.py` to compare.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
