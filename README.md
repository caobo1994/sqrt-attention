# √Block Attention

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**English** | [**中文**](#中文版)

---

## English

**O(L) sparse attention via sqrt‑block sampling | 27.56x speedup on M2 Pro @8192**

### Introduction

√Block Attention is a mathematically inspired sparse attention mechanism. It partitions a long sequence into blocks of exponentially increasing size (small near the current token, large far away), keeps only **√block_size** representative tokens per block, and then applies standard dense attention on these representatives. The total number of representatives becomes **O(√L)**, reducing overall complexity from **O(L²)** to **O(L)**. No manual specification of global tokens is required.

The core mathematical observation:
$$ \sum_{i=1}^{m} \sqrt{a_i} = \Theta\left(\sqrt{\sum_{i=1}^{m} a_i}\right) \quad (\text{when } a_i \text{ grows exponentially}) $$

### Experimental Results

Measured on **MacBook Pro M2 Pro (32GB)** with `torch.float16`, batch size 2, 8 heads, dimension 64.

#### Speedup & Latency

![Performance benchmark](benchmark.png)

| Sequence Length | Dense Attention (ms) | √Block Attention (ms) | **Speedup** |
|----------------|----------------------|------------------------|--------------|
| 1024           | 45.83                | 20.14                  | 2.28x        |
| 2048           | 48.86                | 11.59                  | 4.21x        |
| 4096           | 48.20                | 8.94                   | 5.39x        |
| 8192           | 488.79               | 17.73                  | **27.56x**   |

*Note: At length 8192, dense attention uses ~15GB+ memory, while √Block uses only ~1GB. The latency of √Block barely grows with L, confirming the O(√L) representative count.*

### Reproduction

#### Requirements

- macOS 12.3+ (Apple Silicon M1/M2/M3)
- Python 3.11+
- PyTorch 2.3+ (with MPS support)
- Matplotlib (for plotting)

#### Setup

```bash
git clone https://github.com/caobo1994/sqrt-attention.git
cd sqrt-attention
pip install -r requirements.txt
```

`requirements.txt`:
```
torch>=2.3.0
matplotlib
```

#### Run Benchmark

```bash
# 1. Run experiment (generates benchmark_data.json)
python minimal_test.py

# 2. Generate chart (outputs benchmark.png)
python plot_from_data.py
```

### Method Details

1. **Block partition**: Starting from the nearest token, block sizes grow exponentially (1, 2, 4, 8, …). Near blocks are small (retain more details), far blocks are large (only summary).
2. **Representative selection**: Uniformly sample √block_size tokens per block (O(block_size) time).
3. **Attention**: Place all representatives in original order and run dense attention using `F.scaled_dot_product_attention` (FlashAttention kernel).
4. **Output broadcast**: Write back outputs to the original positions; unselected positions receive zero (can be recovered via residual connections or light aggregation).

**Complexity**: O(L) computation, O(L) memory.

### Why It Matters

- **Novel theory**: First application of ∑√aᵢ = Θ(√∑aᵢ) to sparse attention.
- **Simple to implement**: Can be added as a pre‑sampling layer before FlashAttention.
- **Ultra‑long sequences**: For L = 100k, R ≈ 316 → computation drops from 1e10 to 1e5.
- **Hardware friendly**: O(L) memory access pattern, efficient on GPUs/NPUs.

### File Structure

```
sqrt-attention/
├── LICENSE                  # MIT license
├── README.md                # This file
├── minimal_test.py          # Benchmark script (outputs JSON)
├── plot_from_data.py        # Plot generator from JSON
├── benchmark_data.json      # Generated after running minimal_test.py
├── benchmark.png            # Generated after running plot_from_data.py
└── requirements.txt         # Dependencies
```

### Contact

Author: Cao Bo (caobo1994)  
GitHub: [caobo1994](https://github.com/caobo1994)  
Email: *[bocao1994@qq.com]*

### License

MIT © 2025 Cao Bo

---

## 中文版

### 简介

√Block Attention 是一种基于数学观测的稀疏注意力机制。它将长序列按指数增长划分为不等长块（**近小远大**），每块只保留 **√块大小** 个代表 token，总代表数自动降至 **O(√L)**，然后对这些代表执行标准自注意力。整体计算复杂度从 **O(L²)** 降到 **O(L)**，且无需手工指定全局 token 数。

核心数学关系：
$$ \sum_{i=1}^{m} \sqrt{a_i} = \Theta\left(\sqrt{\sum_{i=1}^{m} a_i}\right) \quad (\text{当 } a_i \text{ 指数增长}) $$

### 实验结果

在 **MacBook Pro M2 Pro (32GB)** 上实测，使用 `torch.float16`，batch size 2，8 个注意力头，维度 64。

#### 加速比与延迟

![性能对比图](benchmark.png)

| 序列长度 | 全注意力 (ms) | √Block 注意力 (ms) | **加速比** |
|---------|---------------|---------------------|------------|
| 1024    | 45.83         | 20.14               | 2.28x      |
| 2048    | 48.86         | 11.59               | 4.21x      |
| 4096    | 48.20         | 8.94                | 5.39x      |
| 8192    | 488.79        | 17.73               | **27.56x** |

*注：在 8192 长度时，全注意力显存占用约 15GB+，√Block 仅约 1GB。√Block 的延迟几乎不随长度增长，完美验证 O(√L) 代表数的理论。*

### 快速复现

#### 环境要求

- macOS 12.3+ (Apple Silicon M1/M2/M3)
- Python 3.11+
- PyTorch 2.3+ (需支持 MPS)
- Matplotlib (用于绘图)

#### 安装

```bash
git clone https://github.com/caobo1994/sqrt-attention.git
cd sqrt-attention
pip install -r requirements.txt
```

`requirements.txt` 内容：
```
torch>=2.3.0
matplotlib
```

#### 运行基准测试

```bash
# 1. 运行实验（自动生成 benchmark_data.json）
python minimal_test.py

# 2. 生成性能图表（输出 benchmark.png）
python plot_from_data.py
```

### 方法细节

1. **分块策略**：从最近 token 开始，块大小按指数增长（1, 2, 4, 8, …），近处块小（保留更多细节），远处块大（仅保留概要）。
2. **代表选择**：每个块均匀采样 √块大小 个 token（复杂度 O(块大小)）。
3. **注意力计算**：所有代表 token 按原始顺序排列，使用 `F.scaled_dot_product_attention`（FlashAttention 内核）执行密集注意力。
4. **输出广播**：将代表 token 的输出写回原序列对应位置，未选中的 token 输出为零（下游可通过残差连接或轻量聚合恢复）。

**复杂度**：O(L) 计算，O(L) 内存。

### 为什么值得关注

- **理论新颖**：首次将 ∑√aᵢ = Θ(√∑aᵢ) 用于注意力稀疏化。
- **实现简单**：可轻松集成到现有 Transformer 中作为预处理采样层。
- **极致长序列**：当 L = 100k 时，R ≈ 316，计算量从 1e10 降至 1e5。
- **硬件友好**：仅需 O(L) 访存模式，适合 GPU/NPU 高效实现。

### 文件结构

```
sqrt-attention/
├── LICENSE                  # MIT 许可证
├── README.md                # 本文件
├── minimal_test.py          # 主测试脚本（输出 JSON）
├── plot_from_data.py        # 从 JSON 生成图表
├── benchmark_data.json      # 运行 minimal_test.py 后生成
├── benchmark.png            # 运行 plot_from_data.py 后生成
└── requirements.txt         # 依赖列表
```

### 联系

作者：曹博 (caobo1994)  
GitHub：[caobo1994](https://github.com/caobo1994)  
邮箱：*bocao1994@qq.com*

### 许可证

MIT © 2025 曹博
```

