import torch
import torch.nn as nn
from .sparse_attn import SparseAttention

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, growth_factor: float, sampling: str, dropout: float):
        super().__init__()
        self.attn = SparseAttention(d_model, n_heads, growth_factor, sampling)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, 4 * d_model)
        self.dropout = nn.Dropout(dropout)          # ← 新增

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.ln1(x)))      # ← 加入 dropout
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x

class SparseGPT(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 512, n_heads: int = 8, n_layers: int = 6,
                 growth_factor: float = 2.0, sampling: str = 'uniform', max_seq_len: int = 1024,
                 dropout: float = 0.0):                # ← 新增 dropout 参数
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)             # ← embedding dropout
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, growth_factor, sampling, dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        positions = torch.arange(0, T, device=input_ids.device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.dropout(x)                            # ← embedding dropout
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits