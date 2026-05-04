# √Block Attention — 主实验 (sparse_attention_exp)

在 WikiText-2 上训练 6 层 GPT 模型（d_model=256, n_heads=8, seq_len=256），对比 **Uniform 采样** 与 **TopK-Norm 采样** 两种稀疏注意力模式。

## 文件结构

```
sparse_attention_exp/
├── models/
│   ├── sparse_attn.py   # √Block 稀疏注意力核心实现
│   └── gpt_sparse.py    # 6 层 GPT 解码器（堆叠 sparse_attn）
├── train.py              # 基础训练脚本（5 epoch）
├── train_advanced.py     # 增强训练（warmup + cosine annealing + checkpoint 恢复）
├── run_full_exp.py       # 全自动流水线（基线训练 → 微调 → 评估）
├── eval.py               # 评估脚本（PPL + 推理时间 + KL 散度 + 可视化）
├── utils.py              # WikiText-2 数据集加载器
│
├── sparse_gpt_uniform_best.pt      # Uniform 最佳模型
├── sparse_gpt_topk_norm_best.pt    # TopK-Norm 最佳模型
├── baseline_uniform.pt             # Uniform 基线备份
├── uniform_30.log                  # Uniform 30 epoch 训练日志
├── topk_30.log                     # TopK-Norm 30 epoch 训练日志
├── comparison.png                  # PPL 与推理时间对比图
└── requirements.txt                # 依赖
```

## 核心代码说明

### models/sparse_attn.py — 稀疏注意力层

- **分块策略** (`create_blocks`): 从序列末尾开始，块大小指数增长（1, 2, 4, 8, ...），近端块小保留细节，远端块大只保留概要。
- **均匀采样** (`uniform_sampling`): 每块内均匀采样 √(块大小) 个 token 作为代表，加入随机偏移减少偏差。
- **TopK-Norm 采样** (`topk_norm_sampling`): 按 Key 向量的 L2 范数取 Top-K，保留信息量最大的 token。
- **注意力计算**: 所有 Q 与代表 K/V 计算点积注意力，非代表位置输出为零。

### models/gpt_sparse.py — GPT 解码器

6 层 Transformer 解码器，每层使用 `SparseAttention` 替代标准全注意力。

### run_full_exp.py — 全自动实验

4 阶段流水线：

| 阶段 | 操作 | 参数 |
|------|------|------|
| 1 | 训练 Uniform 基线 (30 epochs) | sampling=uniform, warmup=500 |
| 2 | 继续微调 Uniform (10 epochs) | 从基线恢复 |
| 3 | 微调 TopK-Norm (10 epochs) | 从基线恢复（切换采样策略） |
| 4 | 评估对比 | eval.py 输出 PPL + 时间 + 图表 |

### eval.py — 评估

- **PPL**: 验证集困惑度（越低越好）
- **Inference Time**: 平均 batch 推理延迟（ms）
- **KL Divergence**: 可选，与全注意力模型的输出分布散度
- **可视化**: 输出 comparison.png

### 模型差异（与 small_exp 对比）

- 不包含 'full' 全注意力模式，只有 'uniform' 和 'topk_norm'
- 不使用 dropout 正则化
- 模型更大（d_model=256, n_layers=6）

## 运行方式

```bash
# 自动完成全实验
python run_full_exp.py

# 或分步执行
python train_advanced.py --sampling uniform --epochs 30
python train_advanced.py --sampling topk_norm --epochs 10 --resume baseline_uniform.pt
python eval.py
```
