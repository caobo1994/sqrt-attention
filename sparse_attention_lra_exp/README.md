# √Block Attention — Long Range Arena (LRA) 实验

在 **Pathfinder-X**（16K 序列长度）上验证 √Block Attention 捕捉长距离依赖的能力。

## 实验动机

| 维度 | 语言建模实验 | LRA 实验（本目录） |
|------|------------|-----------------|
| 任务 | WikiText PPL | Pathfinder-X 分类 / Copying |
| 序列长度 | 256 | 16,384 (16K) |
| 核心能力 | 语言理解 | 长距离依赖捕捉 |
| 挑战 | 困惑度 | 判断两个点是否在同一路径上 |
| 预期 √Block 优势 | 加速比 27x | O(L) vs O(L²) 内存 |

## 文件结构

```
sparse_attention_lra_exp/
├── models/
│   └── lra_model.py      # Pathfinder 模型（√Block + dense 双模式）
├── utils.py              # 数据生成（Pathfinder-X + Copying）
├── train.py              # 训练脚本
├── evaluate.py           # 评估对比
├── run_experiments.py    # Runner（含干净日志）
├── requirements.txt      # 依赖
└── README.md             # 本文件
```

## 任务说明

### Pathfinder-X (16K)

- **输入**: 128×128 二值图像 → 16384 像素序列
- **标签**: 两个标记点是否在同一连续路径上（二分类）
- **正例**: 50% 概率两个标记都在路径上
- **反例**: 50% 概率一个在路径上，另一个不在
- **测评**: 分类准确率（随机基线: 50%）

### Copying Task (备选)

- 输入一串随机数字，经过大量空白后要求复制前段
- 测试模型的长距离记忆能力

## 模型架构

```
Input: (B, 16384) 像素值
  ↓ Linear(1, d_model) + Position Embedding
  ↓ √Block Transformer × n_layers
  ↓ LayerNorm + Mean Pooling
  ↓ Linear(d_model, 2) → 分类
```

- **d_model** = 128（节省显存，16K 长度下仍可训练）
- **n_layers** = 2-4（层数受 16K 序列限制）
- **n_heads** = 4
- **代表数**: 16K seq → ~126 个代表 → QK^T 仅 2M 次操作

## 内存优势

对于 L=16384 的序列长度：

| 指标 | Dense Attention | √Block Attention |
|------|----------------|-----------------|
| QK^T 大小 | 16K × 16K = 268M | 16K × 126 = 2M |
| 计算量 | O(L²) = 268M | O(L) = 2M |
| 估计显存 | 1.1 GB (fp16) | ~8 MB |
| 可行设备 | 需要 GPU | MacBook MPS 即可 |

## 运行方式

```bash
# 完整实验（10 epochs）
python run_experiments.py

# 快速验证（2 epochs, 4096 seq）
python run_experiments.py --quick

# 手动分步
python train.py --sampling uniform --epochs 10
python train.py --sampling topk_norm --epochs 10
python evaluate.py

# Copying 任务验证
python train.py --task copying --sampling uniform --epochs 5
```

## 预期结果

- **√Block Uniform**: 准确率显著高于随机基线（>55%），证明能捕捉长距离依赖
- **√Block TopK-Norm**: 与 Uniform 持平或略高
- **Dense Baseline**: 16K 长度下不可行（MPS OOM），但有理论基准（O(L²) 内存）
- **Random Baseline**: 50.0%
