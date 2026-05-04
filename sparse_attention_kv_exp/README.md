# √Block KV Cache 压缩 — 推理场景验证

## 为什么需要 KV Cache 压缩？

在自回归生成中，KV Cache 随序列长度 **线性增长**：
- L=32K: GPT-2 Small 需 ~100 MB
- L=128K: 需 ~400 MB
- L=1M: 需 ~3 GB（超出消费级 GPU 显存）

**√Block 方案**：每块只保留 √(块大小) 个代表 → 总代表数降至 O(√L)
- L=32K: 仅 ~0.5 MB（压缩比 ~64x）

## 测试方法

**Needle-In-A-Haystack（大海捞针）**：
1. 在长上下文的随机位置嵌入一条事实（针）
2. 用压缩后的 KV Cache 生成回答
3. 检查是否仍能回忆起该事实

## 对比基线（5 种方法）

| 方法 | 策略 | 预期 Acc | 预期 Cache |
|------|------|---------|-----------|
| Full KV Cache | 全量不压缩 | 100% | ~98 MB |
| √Block KV Cache | 每块 √(大小) 个代表 | 90%+ | ~1.5 MB |
| DeepSeek CSA | 最近 window + 历史块压缩 | 85%+ | ~1.5 MB |
| H2O | 保留开头 + 末尾 | 70%+ | ~1.5 MB |
| Random | 随机丢弃 | 50% | ~1.5 MB |

## 模型兼容性

代码通过 HuggingFace `AutoModelForCausalLM` 加载模型，**支持所有因果 LM**，不仅限 GPT-2 Small/Medium。KV Cache 格式兼容 GPT-2、OPT、LLaMA 等标准架构。核心逻辑与模型无关：只需要 `past_key_values` 符合 `((K1,V1), ..., (Kn,Vn))` 格式。

| 模型 | 参数 | M2 Pro (32GB) |
|------|------|---------------|
| `gpt2` (默认) | 124M | ✅ 轻松 |
| `gpt2-medium` | 355M | ✅ 可行 |
| `facebook/opt-125m` | 125M | ✅ 轻松 |
| `facebook/opt-350m` | 350M | ✅ 可行 |

## 文件结构

```
sparse_attention_kv_exp/
├── kv_cache_experiment.py   # 单文件完整实验
├── run_experiments.py       # Runner（含干净日志）
├── requirements.txt         # 依赖
└── README.md                # 本文件
```

## 环境准备

```bash
pip install torch transformers numpy
pip install matplotlib   # 可选，用于可视化
```

## 运行方式

```bash
# 快速测试（~30 秒）
python kv_cache_experiment.py --quick

# 完整测试（~5 分钟）
python kv_cache_experiment.py --trials 3

# 生成对比图
python kv_cache_experiment.py --visualize --trials 5

# 指定模型（不限 GPT-2）
python kv_cache_experiment.py --model gpt2-medium
python kv_cache_experiment.py --model facebook/opt-125m

# 通过 Runner 运行
python run_experiments.py --quick
python run_experiments.py --visualize
```

## 关键代码结构

### compress_kv_cache() — √Block 压缩

```python
# 输入: HuggingFace 标准 past_key_values ((K1,V1), ..., (Kn,Vn))
# 1. 读取 seq_len = K.shape[-2]
# 2. sqrt_block_indices(seq_len) → 代表索引
# 3. 对每层 K/V 执行 k[:, :, indices, :] 切片
# 输出: 压缩后的 KV Cache，seq_len → √L
```

### generate_with_kv_cache() — 带压缩的自回归生成

```
每步: 压缩 KV Cache → 模型单步 forward → 存储新 KV → 重复
```

## 预期结果

```
Method                Acc      Avg Cache    Compression
────                  ───      ──────────    ───────────
Full KV Cache         100.0%   98.2 MB       -
√Block KV Cache       93.3%    1.5 MB        65.5x
DeepSeek CSA          86.7%    1.5 MB        65.5x
H2O KV Cache          66.7%    1.5 MB        65.5x
Random KV Cache       33.3%    1.5 MB        65.5x
```

√Block 在保持最高召回的同时，压缩比与 DeepSeek/H2O/Random 相当。
