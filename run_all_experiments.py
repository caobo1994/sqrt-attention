#!/usr/bin/env python3
"""
√Block Attention — 全实验总调度器

依次运行所有实验目录的 --quick 模式，汇总结果并清理日志。

用法：
    python run_all_experiments.py              # 全部运行（quick 模式）
    python run_all_experiments.py --full       # 非 quick 模式（很慢！）
    python run_all_experiments.py --list       # 仅列出实验
    python run_all_experiments.py --clean      # 仅清理日志
    python run_all_experiments.py --skip kv lra  # 跳过指定实验
"""

import subprocess
import sys
import os
import re
import time
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent

GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
CYAN = '\033[96m'
GRAY = '\033[90m'
BOLD = '\033[1m'
RESET = '\033[0m'

LOG_DIR = ROOT / '.run_all_logs'

# ── 实验注册表 ──
EXPERIMENTS = [
    {
        'name': 'small_exp',
        'label': '🧪 小型验证 (2层, d_model=128, WikiText-2)',
        'dir': 'sparse_attention_small_exp',
        'runner': 'run_experiments.py',
        'quick_flag': '--quick',
        'full_estimate': '~3 min',
    },
    {
        'name': 'main_exp',
        'label': '🔬 主实验 (6层, d_model=256, WikiText-2)',
        'dir': 'sparse_attention_exp',
        'runner': 'run_experiments.py',
        'quick_flag': '--quick',
        'full_estimate': '~8 hours',
    },
    {
        'name': 'dropout_exp',
        'label': '🛡️ Dropout 实验 (6层, d_model=256, WikiText-2)',
        'dir': 'sparse_attention_exp_dropout',
        'runner': 'run_two_phase.py',
        'quick_flag': '',
        'full_estimate': '~8 hours',
        'note': '无 quick 模式，会运行 30+10 轮',
    },
    {
        'name': 'medium_exp',
        'label': '🚀 可扩展性实验 (8层, d_model=512, WikiText-103)',
        'dir': 'sparse_attention_medium_exp',
        'runner': 'run_experiments.py',
        'quick_flag': '--quick',
        'full_estimate': '~24 hours',
    },
    {
        'name': 'lra_exp',
        'label': '🎯 长距离依赖 (Pathfinder-X 16K seq)',
        'dir': 'sparse_attention_lra_exp',
        'runner': 'run_experiments.py',
        'quick_flag': '--quick',
        'full_estimate': '~5 hours',
    },
    {
        'name': 'kv_exp',
        'label': '💾 KV Cache 压缩 (GPT-2, Needle-In-A-Haystack)',
        'dir': 'sparse_attention_kv_exp',
        'runner': 'run_experiments.py',
        'quick_flag': '--quick',
        'full_estimate': '~5 min',
    },
    {
        'name': 'alt_exp',
        'label': '🔄 交替采样 (U/T/U/T vs pure strategies)',
        'dir': 'sparse_attention_alternating_exp',
        'runner': 'run_experiments.py',
        'quick_flag': '--quick',
        'full_estimate': '~2 hours',
    },
]


def print_header(text):
    print(f"\n{BOLD}{BLUE}{'='*65}{RESET}")
    print(f"{BOLD}{BLUE}  {text}{RESET}")
    print(f"{BOLD}{BLUE}{'='*65}{RESET}")


def run_single_experiment(exp, quick=True, log_dir=LOG_DIR):
    """运行单个实验，保存日志，返回结果。"""
    name = exp['name']
    label = exp['label']
    exp_dir = ROOT / exp['dir']
    runner = exp_dir / exp['runner']
    quick_flag = exp.get('quick_flag', '')

    if not runner.exists():
        print(f"  {YELLOW}⚠ Skipped: {runner} not found{RESET}")
        return {'name': name, 'status': 'skipped', 'reason': 'runner not found', 'elapsed': 0}

    if not quick_flag and quick:
        # Try alternate runner
        alt_runner = exp_dir / 'run_full_exp.py'
        if alt_runner.exists():
            runner = alt_runner
            print(f"  {YELLOW}No --quick flag, using {alt_runner.name}{RESET}")
        else:
            print(f"  {YELLOW}⚠ {exp['note']}{RESET}")

    # Build command
    cmd = [sys.executable, str(runner)]
    if quick and quick_flag:
        cmd.append(quick_flag)

    # Log path
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%H%M%S')
    log_path = log_dir / f'{name}_{timestamp}.log'
    status_path = log_dir / f'{name}_status.txt'

    # Run
    print(f"\n  {BOLD}{label}{RESET}")
    print(f"  {GRAY}Command: {' '.join(cmd)}{RESET}")
    print(f"  {GRAY}Log: {log_path}{RESET}")
    sys.stdout.flush()

    start = time.time()
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        bufsize=0, text=False, cwd=exp_dir,
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
                    # Print relevant lines (skip empty tqdm intermediate)
                    if any(kw in line for kw in ['===', 'Phase', 'Training', 'Evaluating',
                                                   'Done', 'Best', 'Val', 'Train Loss',
                                                   'Saved', 'Using', 'Model', 'Parameters',
                                                   'Feature', 'Device', '✓', '✗', 'Log:']):
                        # Clean ANSI for display
                        display = re.sub(r'\033\[[0-9;]*m', '', line)
                        if display.strip():
                            print(f"    {display}")
                    collected.append(line)
                pending = parts[-1] if parts[-1] else ''
            else:
                pending = seg.rstrip()

    if pending and pending.strip():
        collected.append(pending)

    process.wait()
    elapsed = time.time() - start
    mins, secs = divmod(elapsed, 60)

    # Write raw log (before cleaning)
    raw_path = log_path
    with open(raw_path, 'w', encoding='utf-8') as f:
        for line in collected:
            clean = re.sub(r'\033\[[0-9;]*m', '', line)
            f.write(clean + '\n')

    # Determine status
    status = 'completed' if process.returncode == 0 else 'failed'

    # Extract key metrics
    text = raw_path.read_text(encoding='utf-8')
    metrics = {}
    best = re.search(r'Best (Perplexity|Val Accuracy).*?([\d.]+)', text)
    if best:
        metrics[best.group(1)] = best.group(2)
    best_acc = re.search(r'Best Perplexity.*?([\d.]+)', text)
    if best_acc:
        metrics['Best PPL'] = best_acc.group(1)
    best_cache = re.search(r'√Block.*?cache=([\d.]+)', text)
    if best_cache:
        metrics['√Block Cache'] = f"{best_cache.group(1)}MB"

    # Save status
    with open(status_path, 'w') as f:
        f.write(f"Status: {status}\n")
        f.write(f"Elapsed: {int(mins)}m {secs:.0f}s\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    return {
        'name': name,
        'label': label,
        'status': status,
        'elapsed': elapsed,
        'log_path': raw_path,
        'metrics': metrics,
    }


def clean_logs(log_dir=LOG_DIR):
    """清理所有日志文件中的 tqdm 噪声。"""
    if not log_dir.exists():
        print(f"  {GRAY}No logs to clean.{RESET}")
        return

    total_before = 0
    total_after = 0
    for log_path in sorted(log_dir.glob('*.log')):
        before = log_path.stat().st_size
        total_before += before
        content = log_path.read_text(encoding='utf-8', errors='replace')

        # Keep only meaningful lines
        keep_patterns = [
            r'=== Epoch', r'Train Loss:', r'Val', r'Best',
            r'Phase', r'Saved', r'Completed', r'Done',
            r'Using device', r'Model', r'Parameters',
            r'Method', r'Acc', r'Cache', r'✓', r'✗',
            r'Loading', r'Results', r'Conclusion', r'Feature',
        ]
        clean_lines = []
        for line in content.split('\n'):
            line = line.strip()
            if not line:
                continue
            if any(re.search(p, line) for p in keep_patterns):
                clean_lines.append(line)

        clean_content = '\n'.join(clean_lines)
        log_path.write_text(clean_content, encoding='utf-8')
        after = log_path.stat().st_size
        total_after += after
        savings = (1 - after / before) * 100 if before > 0 else 0
        print(f"  {log_path.name}: {before:,}B → {after:,}B ({savings:.0f}% reduction)")

    if total_before > 0:
        print(f"\n  {GREEN}Total: {total_before:,}B → {total_after:,}B ({ (1-total_after/total_before)*100:.0f}% reduction){RESET}")


def main():
    skip_list = []

    # Parse args
    args = sys.argv[1:]
    quick = '--full' not in args
    only_list = '--list' in args
    only_clean = '--clean' in args

    if '--skip' in args:
        idx = args.index('--skip')
        skip_list = args[idx + 1:]

    print(f"\n{BOLD}{'='*65}{RESET}")
    print(f"{BOLD}  √Block Attention — 全实验总调度器{RESET}")
    print(f"{BOLD}{'='*65}{RESET}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Mode: {'QUICK' if quick else 'FULL'}")
    if skip_list:
        print(f"  Skip: {', '.join(skip_list)}")
    print(f"  Total experiments: {len(EXPERIMENTS)}")
    print(f"{'='*65}\n")

    if only_list:
        print(f"{'Name':<20} {'Label':<50} {'Est. time':<12}")
        print(f"{'-'*82}")
        for exp in EXPERIMENTS:
            est = exp.get('full_estimate', '?')
            print(f"{exp['name']:<20} {exp['label']:<50} {est:<12}")
        return 0

    if only_clean:
        print_header("Cleaning All Logs")
        clean_logs(LOG_DIR)
        print(f"\n{GREEN}✓ Done{RESET}")
        return 0

    results = []

    for exp in EXPERIMENTS:
        if exp['name'] in skip_list:
            print(f"\n  {GRAY}Skipping {exp['name']}{RESET}")
            continue

        result = run_single_experiment(exp, quick=quick)
        results.append(result)

        # Stop if experiment failed critically
        if result['status'] == 'failed':
            print(f"\n  {RED}✗ {exp['label']} FAILED{RESET}")
            # Option: continue or break
            # break

    # ── Summary ──
    print(f"\n\n{BOLD}{'='*65}{RESET}")
    print(f"{BOLD}  FINAL SUMMARY{RESET}")
    print(f"{BOLD}{'='*65}{RESET}")
    print(f"\n{'Experiment':<25} {'Status':<12} {'Time':<10} {'Key Metrics'}")
    print(f"{'-'*75}")

    total_elapsed = 0
    for r in results:
        mins, secs = divmod(r['elapsed'], 60)
        total_elapsed += r['elapsed']
        status_symbol = f"{GREEN}✓{RESET}" if r['status'] == 'completed' else f"{RED}✗{RESET}"
        metrics_str = ', '.join(f"{k}={v}" for k, v in r.get('metrics', {}).items())
        print(f"  {r['name']:<23} {status_symbol:<12} {int(mins):>2}m{secs:>4.0f}s {metrics_str}")

    total_mins, total_secs = divmod(total_elapsed, 60)
    print(f"\n  {BOLD}Total time: {int(total_mins)}m {total_secs:.0f}s{RESET}")

    # ── Clean logs ──
    print(f"\n\n{BOLD}Cleaning logs...{RESET}")
    clean_logs(LOG_DIR)

    print(f"\n{GREEN}{'='*65}{RESET}")
    print(f"{GREEN}  All experiments complete!{RESET}")
    print(f"{GREEN}  Logs saved to: {LOG_DIR}{RESET}")
    print(f"{GREEN}{'='*65}{RESET}\n")


if __name__ == "__main__":
    main()
