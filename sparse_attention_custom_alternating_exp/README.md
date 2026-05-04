# √Block Custom Alternating — 任意 U/T 交错采样实验

## 功能

接受 U/T 字符串配置每层的采样策略，自动复用已有检查点。

## 用法

```bash
# 训练单个配置
python train.py --config UTUTUT --epochs 10

# 从已有检查点恢复
python train.py --config UTUTUT --epochs 20 --resume

# 批量探索
python run_experiments.py --configs UTUTUT,TUUTUU,UUUTTT --epochs 10

# 所有 64 种 6 层 U/T 组合
python run_experiments.py --all-alternates --epochs 3

# 汇总结果
python run_experiments.py --summary
```

## 配置示例

```
UUUUUU  全部 uniform（基线）
TTTTTT  全部 topk_norm（基线）
UTUTUT  uniform/topk 交替
TUUTUU  L0=T, L1=U, L2=U, L3=T, L4=U, L5=U
UUUTTT  前 3 层 uniform, 后 3 层 topk
TTTUUU  前 3 层 topk, 后 3 层 uniform
```

## 自动恢复

- 检查点按 config 命名: `ckpt_UTUTUT.pt`
- 日志文件: `train_log.json`（记录所有已完成训练历史）
- `--resume` 自动从上次中断处继续
