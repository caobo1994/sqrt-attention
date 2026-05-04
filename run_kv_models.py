#!/usr/bin/env python3
"""
√Block KV Cache — 多模型对比实验

在 5 个预训练小模型上运行 Needle-In-A-Haystack 测试，
对比 Full / √Block / Random / H2O 四种 KV Cache 策略。

用法：
    python run_kv_models.py                  # quick 模式跑 5 个模型
    python run_kv_models.py --full           # 完整模式（更多 trials）
    python run_kv_models.py --model gpt2     # 只跑指定模型
    python run_kv_models.py --summary        # 仅汇总已有结果
    python run_kv_models.py --clean          # 清理日志
"""

import subprocess
import sys
import os
import re
import time
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXP_DIR = ROOT / 'sparse_attention_kv_exp'
LOG_DIR = ROOT / '.kv_model_logs'

MODELS = [
    {'name': 'gpt2',            'label': 'GPT-2 Small',    'params': '124M', 'speed': '✅ 轻松'},
    {'name': 'gpt2-medium',     'label': 'GPT-2 Medium',   'params': '355M', 'speed': '✅ 可行'},
    {'name': 'gpt2-large',      'label': 'GPT-2 Large',    'params': '774M', 'speed': '⚠️ 较慢'},
    {'name': 'facebook/opt-125m','label': 'OPT-125M',       'params': '125M', 'speed': '✅ 轻松'},
    {'name': 'facebook/opt-350m','label': 'OPT-350M',       'params': '350M', 'speed': '✅ 可行'},
]

GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
CYAN = '\033[96m'
GRAY = '\033[90m'
BOLD = '\033[1m'
RESET = '\033[0m'


def run_model(model_name, quick=True, trials=1):
    """运行单个模型的 KV Cache 实验。"""
    model_id = model_name.replace('/', '_')
    log_path = LOG_DIR / f'kv_{model_id}_{"quick" if quick else "full"}.log'

    cmd = [
        sys.executable, str(EXP_DIR / 'kv_cache_experiment.py'),
        '--model', model_name,
        '--trials', str(trials if not quick else 1),
    ]
    if quick:
        cmd.append('--quick')

    print(f"\n  {BOLD}Running {model_name}...{RESET}")
    print(f"  {GRAY}Command: {' '.join(cmd)}{RESET}")

    start = time.time()
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        bufsize=0, text=False, cwd=EXP_DIR,
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
                    if any(kw in line for kw in ['Summary', 'Method', 'Full', '√Block',
                                                   'Random', 'H2O', 'Avg Cache',
                                                   'Accuracy', 'Acc', 'Compression',
                                                   'Needle', 'Device', 'Loading',
                                                   'Model:', 'Context:', '✓']):
                        display = re.sub(r'\033\[[0-9;]*m', '', line)
                        print(f"    {display}")
                    collected.append(line)
            else:
                pending = seg.rstrip()

    process.wait()
    elapsed = time.time() - start

    with open(log_path, 'w', encoding='utf-8') as f:
        for line in collected:
            clean = re.sub(r'\033\[[0-9;]*m', '', line)
            f.write(clean + '\n')

    # Extract results
    text = ''.join(collected)
    results = {'name': model_name, 'status': 'ok' if process.returncode == 0 else 'error', 'elapsed': elapsed}

    # Parse summary table (multi-word method names: "Full KV Cache")
    lines = text.split('\n')
    results_data = {}
    for line in lines:
        # Remove ANSI codes before matching
        clean = re.sub(r'\033\[[0-9;]*m', '', line).strip()
        # Method-specific patterns
        for prefix, key in [('Full KV Cache', 'full'), ('DeepSeek CSA', 'deepseek'),
                             ('\u221aBlock KV Cache', 'block'), ('H2O KV Cache', 'h2o'),
                             ('Random KV Cache', 'random')]:
            if prefix not in clean:
                continue
            # Extract accuracy and cache after the method name
            remainder = clean.replace(prefix, '', 1).strip()
            parts = remainder.split()
            for i, p in enumerate(parts):
                if p == '%' and i > 0:
                    results_data[f'{key}_acc'] = parts[i-1]
                    for pp in parts[i+1:]:
                        try:
                            float(pp)
                            results_data[f'{key}_cache'] = pp
                            break
                        except ValueError:
                            continue
                    break
            break

    for key, val in results_data.items():
        results[key] = val

    # 保存 JSON 结果
    json_path = LOG_DIR / f'{model_id}_results.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    mins, secs = divmod(elapsed, 60)
    status = f"{GREEN}✓{RESET}" if results['status'] == 'ok' else f"{RED}✗{RESET}"
    block_acc = results.get('block_acc', '?')
    print(f"\n  {status} Done in {int(mins)}m {secs:.0f}s | √Block Acc: {block_acc}")
    return results


def print_summary():
    """汇总所有已有结果。"""
    print(f"\n{BOLD}{'='*70}{RESET}")
    print(f"{BOLD}  KV Cache 压缩 — 多模型对比汇总{RESET}")
    print(f"{BOLD}{'='*70}{RESET}")
    print(f"\n{'Model':<18} {'Params':<8} {'Full Acc':<10} {'√Block Acc':<12} {'Random Acc':<12} {'H2O Acc':<10} {'√Block Cache':<12}")
    print('-'*76)

    for mod in MODELS:
        model_id = mod['name'].replace('/', '_')
        json_path = LOG_DIR / f'{model_id}_results.json'
        json_path2 = LOG_DIR / f'kv_{model_id}_quick.json'  # alt name

        data = None
        for p in [json_path, json_path2, LOG_DIR / ("kv_" + model_id + "_quick.json"), LOG_DIR / ("kv_" + model_id + "_full.json")]:
            if p.exists():
                try:
                    data = json.loads(p.read_text())
                except:
                    pass

        if data:
            full_acc = data.get('full_acc', '-')
            block_acc = data.get('block_acc', '-')
            random_acc = data.get('random_acc', '-')
            h2o_acc = data.get('h2o_acc', '-')

            deepseek_acc = data.get('deepseek_acc', '-')
            block_cache = data.get('block_cache', '-')
            print(f"  {mod['name']:<18} {mod['params']:<8} {full_acc:<8} {deepseek_acc:<8} {block_acc:<8} {h2o_acc:<8} {random_acc:<8} {block_cache:<8}")
        else:
            print(f"  {GRAY}{mod['name']:<16} {mod['params']:<8} {'-':<10} {'-':<12} {'-':<12} {'-':<10} {'-':<12}{RESET}")

    print('-'*76)
    print(f"  Random baseline accuracy: ~50.0%")
    print(f"{'='*70}\n")


def clean_logs():
    """清理日志中的 tqdm 噪声。"""
    if not LOG_DIR.exists():
        return
    keep_patterns = [
        'Summary', 'Method', 'Full KV', '√Block', 'Random', 'H2O',
        'Avg Cache', 'Compression', 'Accuracy', 'Acc', 'Device',
        'Loading', 'Model:', 'Context:', 'Done', 'RESULTS',
        '✓', '✗',
    ]
    for f in sorted(LOG_DIR.glob('*.log')):
        before = f.stat().st_size
        lines = f.read_text(encoding='utf-8', errors='replace').split('\n')
        clean = [l for l in lines if any(p in l for p in keep_patterns)]
        f.write_text('\n'.join(clean) + '\n', encoding='utf-8')
        after = f.stat().st_size
        if before > 0:
            print(f"  {f.name}: {before:,}B → {after:,}B ({ (1-after/before)*100:.0f}%)")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="√Block KV Cache — 多模型对比")
    parser.add_argument('--full', action='store_true', help='完整模式')
    parser.add_argument('--model', type=str, default=None, help='只跑指定模型')
    parser.add_argument('--trials', type=int, default=None, help='trials 数量')
    parser.add_argument('--summary', action='store_true', help='汇总已有结果')
    parser.add_argument('--clean', action='store_true', help='清理日志')
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if args.clean:
        print(f"\n{BOLD}Cleaning logs...{RESET}")
        clean_logs()
        print(f"\n{GREEN}✓ Done{RESET}")
        return

    if args.summary:
        print_summary()
        return

    quick = not args.full
    trials = args.trials or (1 if quick else 3)

    selected = [m for m in MODELS if args.model is None or m['name'] == args.model]

    print(f"\n{BOLD}{'='*70}{RESET}")
    print(f"{BOLD}  √Block KV Cache — 多模型对比实验{RESET}")
    print(f"{BOLD}{'='*70}{RESET}")
    print(f"  Mode: {'QUICK' if quick else 'FULL'}")
    print(f"  Trials per model: {trials}")
    print(f"  Models: {len(selected)}")
    print(f"  Total estimated time: ", end='')
    total_est = 0
    for m in selected:
        if 'large' in m['name']:
            total_est += 5 if quick else 40
        elif '350m' in m['name'] or 'medium' in m['name']:
            total_est += 2 if quick else 15
        else:
            total_est += 1 if quick else 8
    print(f"~{total_est} minutes\n")

    all_results = []
    for mod in selected:
        print(f"\n{'─'*55}")
        r = run_model(mod['name'], quick=quick, trials=trials)
        all_results.append(r)

    # 汇总
    print_summary()

    # 清理日志
    print(f"\n{BOLD}Cleaning logs...{RESET}")
    clean_logs()
    print(f"\n{GREEN}✓ All done!{RESET}")


if __name__ == "__main__":
    main()
