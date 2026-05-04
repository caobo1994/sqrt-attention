# √Block Attention — 小型验证实验 (sparse_attention_small_exp)

在 WikiText-2 上训练 **小型 2 层 GPT 模型**（d_model=128, n_heads=8, seq_len=256），快速验证三种注意力模式：

1. **Full Attention** — 全注意力基线
2. **√Block Uniform** — 均匀采样
3. **√Block TopK-Norm** — 按 Key 范数采样

## 与主实验的区别

| 特性 | 主实验 | 小型实验 |
|------|--------|---------|
| 层数 | 6 层 | 2 层 |
| 模型维度 | d_model=256 | d_model=128 |
| 训练轮数 | 30+10 | 2 |
| 架构 | 仅稀疏 | 包含 Full Attention 基线 |
| 训练速度 | 慢 | 快（适合快速验证） |
| 实验流程 | 基线→微调→评估 | 并行训练三种模式 |

## 文件结构

```
sparse_attention_small_exp/
├── models/
│   ├── sparse_attn.py   # 支持 'full' 模式的稀疏注意力（与主实验不同！）
│   └── gpt_sparse.py    # 2 层 GPT（与主实验相同架构，参数更小）
├── train.py             # 基础训练（5 epoch）
├── train_advanced.py    # 增强训练（支持三种 sampling 模式）
├── run_full_exp.py      # 完整流水线（全注意力 + Uniform + TopK-Norm）
├── run_small_exp.py     # 小型验证流水线（2 epoch 快速验证）
├── eval.py              # 2 路对比评估（与主实验相同）
├── eval_three.py        # 3 路对比评估（Full + Uniform + TopK-Norm）
├── utils.py             # WikiText-2 数据集加载器
│
├── sparse_gpt_full_best.pt           # Full Attention 最佳模型
├── sparse_gpt_uniform_best.pt        # √Block Uniform 最佳模型
├── sparse_gpt_topk_norm_best.pt      # √Block TopK-Norm 最佳模型
├── uniform_30.log                    # 训练日志
├── comparison.png                    # 对比图表
└── requirements.txt                 # 依赖
```

## 关键代码说明

### models/sparse_attn.py — 支持全注意力模式

与主实验版本的关键区别在于 `SparseAttention.forward()` 支持 `sampling='full'` 模式：

```python
# 全注意力分支
if self.sampling == 'full':
    attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / scale
    attn_weights = F.softmax(attn_scores, dim=-1)
    attn_output = torch.matmul(attn_weights, V)
    return self.out_proj(attn_output)

# 稀疏注意力分支（与主实验相同）
blocks = create_blocks(...)
...
```

其余差异：
- 完整类型注解（Type Hints）
- TopK-Norm 使用局部变量名 `k_actual = min(k, avg_norms.size(0))` 避免越界

### run_small_exp.py — 小型验证

```bash
1/3: Training Full Attention (baseline)    → 2 epochs
2/3: Training √Block Uniform               → 2 epochs
3/3: Training √Block TopK-Norm             → 2 epochs
     Evaluation: comparing PPL and time    → eval_three.py
```

验证参数：batch_size=8, seq_len=256, d_model=128, n_heads=8, n_layers=2

### eval_three.py — 三路对比

对比 Full Attention、Uniform、TopK-Norm 三种模式的：
- **验证集困惑度 (PPL)**
- **首 batch 推理时间 (ms)**

设计为轻量级快速验证，约 2-5 分钟可完成完整实验。

## 运行方式

```bash
# 快速验证（2 epochs，约 2 分钟）
python run_small_exp.py

# 完整流水线（更多 epoch）
python run_full_exp.py

# 分步
python train_advanced.py --sampling full --epochs 2
python train_advanced.py --sampling uniform --epochs 2
python train_advanced.py --sampling topk_norm --epochs 2
python eval_three.py
```
