#!/usr/bin/env python3
"""
√Block Attention — Long Range Arena (Pathfinder-X) Runner

Trains √Block attention models on Pathfinder-X (16K sequence length).
Compares Uniform vs TopK-Norm sampling.

Usage:
    python run_experiments.py                # Full run (10 epochs each)
    python run_experiments.py --quick        # Quick: 2 epochs, 4096 seq
    python run_experiments.py --clean-logs   # Clean logs
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
    best = re.search(r'Best Val Accuracy.*?([\d.]+)', log_path.read_text('utf-8'))
    if best:
        print(f"  Best Acc: {GREEN}{best.group(1)}{RESET}")
    print(f"  Log: {log_path.name} ({log_path.stat().st_size:,} bytes)")
    return process.returncode


def main():
    quick = '--quick' in sys.argv
    clean_only = '--clean-logs' in sys.argv

    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  √Block Attention — Long Range Arena (Pathfinder-X){RESET}")
    print(f"{BOLD}{'='*60}{RESET}")
    print(f"  Mode: {'Quick' if quick else 'Full'}{' (clean only)' if clean_only else ''}\n")

    if clean_only:
        for f in ['uniform_pathfinder.log', 'topk_norm_pathfinder.log']:
            p = BASE / f
            if p.exists():
                p.unlink()
                print(f"  Removed: {f}")
        print(f"\n{GREEN}✓ Clean{RESET}")
        return 0

    quick_args = ['--quick'] if quick else []
    epochs = 2 if quick else 10

    # Phase 1: √Block Uniform
    rc = run_with_clean_log(
        [sys.executable, str(TRAIN), "--sampling", "uniform",
         "--epochs", str(epochs)] + quick_args,
        BASE / f"uniform_pathfinder_{epochs}.log",
        f"Phase 1/2: √Block Uniform ({epochs} epochs, 16K seq)")
    if rc != 0:
        return 1

    # Phase 2: √Block TopK-Norm
    rc = run_with_clean_log(
        [sys.executable, str(TRAIN), "--sampling", "topk_norm",
         "--epochs", str(epochs)] + quick_args,
        BASE / f"topk_norm_pathfinder_{epochs}.log",
        f"Phase 2/2: √Block TopK-Norm ({epochs} epochs, 16K seq)")
    if rc != 0:
        return 1

    print(f"\n{BOLD}{GREEN}{'='*60}{RESET}")
    print(f"{BOLD}{GREEN} ✓ Training complete!{RESET}")
    print(f"{BOLD}{GREEN}{'='*60}{RESET}")
    print(f"\n  Run: python evaluate.py")
    print(f"  To compare accuracy.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
