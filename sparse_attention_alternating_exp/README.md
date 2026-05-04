# √Block Alternating Sampling — 层间交替采样实验

**核心问题**：不同 Transformer 层是否应该用不同的稀疏注意力采样策略？

## 假设

- **浅层**（底部 layers）：Uniform 采样更好 → 覆盖面广，提取基础特征
- **深层**（顶部 layers）：TopK-Norm 采样更好 → 聚焦重要 token，精细化表达
- **交替采样**可能比单一策略更好

## 四种配置

| 配置 | Layer 0 | Layer 1 | Layer 2 | Layer 3 | Layer 4 | Layer 5 |
|------|---------|---------|---------|---------|---------|---------|
| uniform-all | U | U | U | U | U | U |
| topk-all | T | T | T | T | T | T |
| alternate | **U** | **T** | **U** | **T** | **U** | **T** |
| alternate_rev | **T** | **U** | **T** | **U** | **T** | **U** |

U = Uniform 采样, T = TopK-Norm 采样

## 文件结构

```
sparse_attention_alternating_exp/
├── models/
│   └── alternating_gpt.py   # 支持逐层配置采样策略的 GPT 模型
├── train.py                 # 训练脚本（支持 --config 参数）
├── evaluate.py              # 评估对比（输出 4 种配置的 PPL）
├── run_experiments.py       # Runner（clean logging）
└── README.md                # 本文件
```

## 模型架构

与主实验相同的 6 层 GPT（d_model=256, n_heads=8），唯一区别：
- 每层独立指定采样策略（uniform / topk_norm）
- 模型通过 `sampling_config` 参数控制

```python
# 示例：逐层配置
model = AlternatingGPT(
    sampling_config=['uniform', 'uniform', 'topk_norm', 'topk_norm', 'alternate', 'alternate'],
)
```

## 运行方式

```bash
# 完整实验（10 epoch × 4 = ~2 小时）
python run_experiments.py

# 快速验证（2 epoch × 4 = ~20 分钟）
python run_experiments.py --quick

# 分步执行
python train.py --config uniform
python train.py --config topk_norm
python train.py --config alternate
python train.py --config alternate_rev
python evaluate.py

# 评估已有结果
python evaluate.py
```

## 预期结果

```
Config               Layer Pattern                  PPL
────                 ─────                           ───
uniform              [U, U, U, U, U, U]             26.5
topk_norm            [T, T, T, T, T, T]             9.6
alternate            [U, T, U, T, U, T]             ???
alternate_rev        [T, U, T, U, T, U]             ???
```

如果交替采样优于单一策略，则说明不同层确实需要不同的注意力模式。
