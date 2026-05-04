# √Block Attention — Dropout 实验 (sparse_attention_exp_dropout)

在 WikiText-2 上训练 6 层 GPT 模型，与主实验相同的架构（d_model=256, n_heads=8, seq_len=256），**额外引入 dropout 正则化**对比 Uniform 与 TopK-Norm 采样。

## 与主实验的区别

| 特性 | 主实验 | Dropout 实验 |
|------|--------|-------------|
| 正则化 | 无 | embedding dropout + attention/FFN 残差后 dropout |
| TopK-Norm 微调 | 常规 | 更强正则化 (dropout=0.3, weight_decay=1e-3) |
| 早停 | 无 | patience=5，验证集 5 轮不改善则停止 |
| 训练流程 | 基线+微调+评估 | 两阶段推进（run_two_phase.py） |

## 文件结构

```
sparse_attention_exp_dropout/
├── models/
│   ├── sparse_attn.py           # 与主实验相同
│   └── gpt_sparse.py            # 添加 dropout 支持（见下方 diff）
├── train.py                     # 基础训练（5 epoch）
├── train_advanced.py            # 增强训练 + dropout + 早停
├── run_full_exp.py              # 旧版全自动流水线（无 dropout）
├── run_two_phase.py             # 两阶段推进（带 dropout 正则化）
├── eval.py                      # 评估（与主实验相同）
├── utils.py                     # 数据集加载器（与主实验相同）
│
├── sparse_gpt_uniform_best.pt   # Uniform 最佳模型
├── sparse_gpt_topk_norm_best.pt # TopK-Norm 最佳模型
├── baseline_uniform.pt          # Uniform 基线备份
├── uniform_30.log               # Uniform 训练日志
├── topk_30.log                  # TopK-Norm 训练日志
├── comparison.png               # PPL 与推理时间对比图
└── requirements.txt             # 依赖
```

## 关键代码变更 (gpt_sparse.py vs 主实验)

```diff
# 新增 dropout 参数
- class TransformerBlock(...):
+ class TransformerBlock(..., dropout: float = 0.0):
+     self.dropout = nn.Dropout(dropout)
  
  def forward(self, x):
-     x = x + self.attn(self.ln1(x))
-     x = x + self.ffn(self.ln2(x))
+     x = x + self.dropout(self.attn(self.ln1(x)))
+     x = x + self.dropout(self.ffn(self.ln2(x)))

- class SparseGPT(..., dropout: float = 0.0):
+     self.dropout = nn.Dropout(dropout)     # embedding dropout
```

### run_two_phase.py — 两阶段推进

| 阶段 | 操作 | 参数 |
|------|------|------|
| 1 | 训练 Uniform 基线 (30 epochs, 早停) | dropout=0.2, patience=5 |
| 2a | 微调 Uniform (10 epochs) | dropout=0.2，从基线恢复 |
| 2b | 微调 TopK-Norm (10 epochs) | dropout=0.3, weight_decay=1e-3，从基线恢复 |
| 3 | 评估对比 | eval.py |

**设计意图**:
- Uniform 基线在弱正则化下训练，确保充分的模型容量
- TopK-Norm 微调时用更强正则化，防止自适应采样导致的过拟合
- 早停机制避免过训练

## 运行方式

```bash
# 两阶段推进（推荐）
python run_two_phase.py

# 或按旧版流水线
python run_full_exp.py
```
