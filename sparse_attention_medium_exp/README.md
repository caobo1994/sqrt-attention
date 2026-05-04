# √Block Attention — WikiText-103 中型实验 (sparse_attention_medium_exp)

在 **WikiText-103**（1.03 亿 token）上训练 **8 层 GPT 模型（d_model=512, ~40M 参数）**，验证 √Block Attention 在更大规模数据和模型上的可扩展性。

## 实验动机

| 维度 | 小型实验 | 中型实验（本目录） |
|------|---------|-----------------|
| 数据集 | WikiText-2 (~2M tokens) | WikiText-103 (~103M tokens) |
| 模型参数 | d_model=256, 6 layers (~15M) | d_model=512, 8 layers (~40M) |
| 训练规模 | 30 轮基线 + 10 轮微调 | 50 轮基线 + 10 轮微调 |
| 关注点 | 功能验证 | 可扩展性验证 |

## 文件结构

```
sparse_attention_medium_exp/
├── models/
│   ├── sparse_attn.py      # √Block 稀疏注意力（与主实验相同）
│   └── gpt_sparse.py       # GPT 解码器（与主实验相同）
├── train.py                # 基础训练（5 epoch，适合快速调试）
├── train_advanced.py       # 增强训练（warmup + cosine annealing + 梯度裁剪）
├── run_full_exp.py         # 全自动流水线（基线 → 微调 → 评估）
├── eval.py                 # 评估（PPL + 推理时间 + KL 散度 + 可视化）
├── utils.py                # WikiText-103 数据集加载器
├── README.md               # 本文件
└── requirements.txt        # 依赖
```

## 与主实验的关键差异

### 1. 数据集 (utils.py)

```python
# 主实验（~2M tokens）
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)

# 本实验（~103M tokens，50 倍数据）
dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
```

### 2. 模型参数 (所有训练脚本)

| 参数 | 主实验 | 中型实验 |
|------|--------|---------|
| d_model | 256 | 512 |
| n_heads | 8 | 8 |
| n_layers | 6 | 8 |
| 参数量 | ~15M | ~40M |
| 学习率 | 3e-4 | 1e-4 |

### 3. 训练增强点

- **梯度裁剪**：`clip_grad_norm_(model.parameters(), max_norm=1.0)`，防止大模型梯度爆炸
- **更低学习率**：1e-4（大模型需要更小步长）
- **更长预热**：1000 steps（大模型需要更充分预热）
- **早停机制**：patience=10，验证集 10 轮不改善则停止
- **权重衰减**：1e-2，增强泛化

### 4. 检查点命名

为避免与主实验冲突，检查点文件使用 `_wt103_best.pt` 后缀：

```
sparse_gpt_uniform_wt103_best.pt
sparse_gpt_topk_norm_wt103_best.pt
baseline_uniform_wt103.pt
```

## 核心代码说明

### train_advanced.py — 增强训练

```
参数                        默认值      说明
───                         ────      ────
sampling                    uniform   采样策略（uniform / topk_norm）
d_model                     512       模型维度
n_heads                     8         注意力头数
n_layers                    8         层数
batch_size                  4         批次大小
seq_len                     256       序列长度
epochs                      5         训练轮数
lr                          1e-4      学习率
warmup_steps                1000      预热步数
grad_clip                   1.0       梯度裁剪阈值
patience                    0         早停耐心值（0=禁用）
resume                      None      检查点恢复路径
```

### run_full_exp.py — 全自动流水线

```
Phase 1: Uniform 基线训练 (50 epochs)
  ├─ learning rate: 1e-4, warmup: 1000 steps
  ├─ patience=10 (early stopping)
  └─ saves: sparse_gpt_uniform_wt103_best.pt

Phase 2: Uniform 微调 (10 epochs)
  ├─ 从基线恢复
  ├─ learning rate: 5e-5 (减半)
  └─ saves: sparse_gpt_uniform_wt103_best.pt (覆盖)

Phase 3: TopK-Norm 微调 (10 epochs)
  ├─ 从基线恢复，切换采样策略
  ├─ learning rate: 5e-5
  └─ saves: sparse_gpt_topk_norm_wt103_best.pt

Phase 4: 评估对比
  ├─ PPL（困惑度）
  ├─ 推理时间 (ms)
  ├─ 内存占用 (MB)
  └─ 输出 comparison_wt103.png
```

## 运行方式

```bash
# 1. 安装依赖（注意：WikiText-103 需要 datasets 库）
pip install -r requirements.txt

# 2. 全自动实验（建议）
python run_full_exp.py

# 3. 或分步执行
# 训练基线
python train_advanced.py --sampling uniform --epochs 50 --warmup_steps 1000

# 微调
python train_advanced.py --sampling uniform --epochs 10 --resume baseline_uniform_wt103.pt --lr 5e-5
python train_advanced.py --sampling topk_norm --epochs 10 --resume baseline_uniform_wt103.pt --lr 5e-5

# 评估
python eval.py
```

## 预期结果与参考

基于小型实验的外推预期：
- **Uniform PPL**: 略高于 Full Attention（~5-10%），可作为参考基线
- **TopK-Norm PPL**: 与 Uniform 持平或略低（因选择信息量更大的 token）
- **推理时间**: 两种稀疏方法相近，远快于全注意力
- **内存占用**: 稀疏方法 < 全注意力（FlashAttention 已优化，但仍需 O(L²) 显存）

数据量和模型规模增大后，TopK-Norm 的相对优势可能更明显，因为更多数据让自适应采样策略能学到更好的代表性选择。

## 注意事项

1. WikiText-103 首次加载会下载 ~180MB 数据，需要网络连接
2. 8 层 d_model=512 模型约 40M 参数，建议至少 16GB 内存的 MPS 设备
3. 完整 50+10 轮训练可能需要数小时到一天，建议使用 `patience=10` 提前停止
4. MPS 后端对大序列可能存在限制，seq_len=256 是安全的
