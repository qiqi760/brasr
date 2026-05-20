"""
projections.py
──────────────
Projection heads that map encoder outputs into the shared
embedding space of dimension D (embed_dim).

Paper (Section 2.2):
    Both text encoder (BERT, hidden=768) and audio encoder
    (Data2Vec-large, hidden=1024) produce embeddings of different
    dimensions. A projection layer aligns them to a common D.

    Paper does not explicitly describe the projection architecture;
    we assume a single linear layer (no activation) following the
    convention from CLIP/CLAP literature.

    TODO: Replace with a 2-layer MLP + GeLU if ablation shows benefit.

MODIFIED (2026-05-13):
    1) 仿照 dasheng-glap 的 projector head 结构，将单层线性层替换为
       2-layer MLP + ReLU，以提升投影空间的表达能力。
       dasheng-glap 参考代码: glap_model/models/glap.py:62-63

    2) 进一步将 projector head 替换为两层 Transformer 架构，以显著
       增大参数量并增强特征交互能力。
       每层 TransformerEncoderLayer 包含 Multi-Head Self-Attention +
       Feed-Forward Network (dim_feedforward = 4 × d_model)，使用
       Pre-LN (norm_first=True) 与 GELU 激活，训练更稳定。

    原代码（单层 Linear 与 2-layer MLP）均已注释保留，便于对比与回退。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    """
    Attention Pooling 模块。

    仿照论文 "Efficient and Scalable Bias Retrieval Framework for Contextual
    Biasing ASR in Speech LLM" (arXiv:2505.19179) 中的 Attention Pooling:

        令 d 为注意力计算维度，输出为:
            X̂ = Linear · Σ_{T′}(Attention(H))

    实现方式：使用可学习的查询向量 Q，以 H 作为 key 和 value，通过
    缩放点积注意力计算加权聚合特征，最后经 Linear 层映射。

    参照 PyTorch nn.MultiheadAttention 实现，支持真正的 multi-head
    attention，并安全处理全 padding 序列（避免 softmax 输出 nan 导致
    分布式训练梯度崩溃）。

    Args:
        hidden_size: 输入特征维度（也是输出维度）。
        num_heads:   注意力头数（默认 1，即单头注意力池化）。
        attn_dim:    注意力计算维度 d；若为 None，则等于 hidden_size。
                     必须能被 num_heads 整除。

    Input:
        hidden_states: [B, T, hidden_size] — 序列特征。
        attention_mask: [B, T] — 可选，1=有效位置，0=padding（用于 text）。

    Output:
        pooled: [B, hidden_size] — 注意力池化后的聚合特征。
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 1,
        attn_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.attn_dim = attn_dim or hidden_size
        self.num_heads = num_heads
        self.head_dim = self.attn_dim // num_heads

        if self.attn_dim % num_heads != 0:
            raise ValueError(
                f"attn_dim ({self.attn_dim}) must be divisible by num_heads ({num_heads})"
            )

        # 可学习的查询向量 Q [num_heads, 1, head_dim]
        self.query = nn.Parameter(torch.randn(num_heads, 1, self.head_dim))

        # Key / Value 投影
        self.key_proj = nn.Linear(hidden_size, self.attn_dim, bias=False)
        self.value_proj = nn.Linear(hidden_size, self.attn_dim, bias=False)

        # 输出 Linear（对应公式中的 "Linear"）
        self.out_proj = nn.Linear(self.attn_dim, hidden_size, bias=True)

        self.scale = self.head_dim ** -0.5

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.query)
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states:  [B, T, hidden_size]
            attention_mask: [B, T] — 1=valid, 0=padding (optional)

        Returns:
            pooled: [B, hidden_size]
        """
        B, T, _ = hidden_states.shape

        # Project K, V
        k = self.key_proj(hidden_states)    # [B, T, attn_dim]
        v = self.value_proj(hidden_states)  # [B, T, attn_dim]

        # Reshape for multi-head attention: [B, num_heads, T, head_dim]
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Expand query to batch size: [B, num_heads, 1, head_dim]
        q = self.query.unsqueeze(0).expand(B, -1, -1, -1)

        # Compute attention scores: [B, num_heads, 1, T]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply padding mask if provided
        if attention_mask is not None:
            # attention_mask: [B, T] with 1=valid, 0=pad
            # Expand to [B, 1, 1, T] for broadcasting
            mask_bool = attention_mask.unsqueeze(1).unsqueeze(2).bool()
            scores = scores.masked_fill(~mask_bool, float("-inf"))

        # Softmax over time dimension
        attn_weights = F.softmax(scores, dim=-1)  # [B, num_heads, 1, T]

        # 安全处理：全 padding 序列会导致 softmax 输出 nan
        # 参照 PyTorch 做法，将 nan 替换为 0（即均匀分布的退化情况）
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        # Weighted sum: [B, num_heads, 1, head_dim]
        pooled = torch.matmul(attn_weights, v)

        # Concatenate heads: [B, 1, attn_dim]
        pooled = pooled.transpose(1, 2).contiguous().view(B, 1, self.attn_dim)

        # Linear projection
        pooled = self.out_proj(pooled)  # [B, 1, hidden_size]

        return pooled.squeeze(1)  # [B, hidden_size]


class ProjectionHead(nn.Module):
    """
    Projection head with optional L2-normalisation.

    Used to project:
        - text embeddings:  [B, text_hidden]  → [B, embed_dim]
        - audio embeddings: [B, audio_hidden] → [B, embed_dim]
          and also for local:
        - audio local:      [B, T', audio_hidden] → [B, T', embed_dim]

    Args:
        in_dim:    Input feature dimension (e.g. 768 for BERT, 1024 for D2V).
        out_dim:   Shared embedding dimension D (e.g. 512).
        normalize: If True, L2-normalise the output (recommended for cosine
                   similarity based contrastive loss).

    Shape:
        input:  [..., in_dim]  or  [..., seq_len, in_dim]
        output: [..., out_dim]  or  [..., seq_len, out_dim]
                (2-D input → 2-D output; 3-D input → 3-D output)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        normalize: bool = True,
        num_layers: int = 2,
        num_heads: int = 8,
        dim_feedforward: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.normalize = normalize
        self.out_dim = out_dim

        if dim_feedforward is None:
            dim_feedforward = out_dim * 4

        # ── 原代码 v0：单层线性投影（已注释）────────────────
        # 原实现为单层线性投影（无激活函数，无 bias）
        # self.linear = nn.Linear(in_dim, out_dim, bias=False)
        # ──────────────────────────────────────────────────

        # ── 原代码 v1：2-layer MLP + ReLU─────────
        # 仿照 dasheng-glap 的 projector head 结构：
        #   nn.Sequential(Linear → ReLU → Linear)
        self.projection = nn.Sequential(
            nn.Linear(in_dim, out_dim, bias=True),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim, bias=True),
        )
        # ──────────────────────────────────────────────────
        '''
        # ── 修改处 v2：替换为两层 Transformer 架构 ──────────
        # 1) 先将输入特征维度映射到 Transformer 的 d_model
        self.input_proj = nn.Linear(in_dim, out_dim, bias=True)

        # 2) 可学习的位置编码（支持最长 512 的序列）
        #    audio local 经过下采样后 T'' 通常 < 200，512 足够覆盖
        self.pos_embed = nn.Embedding(512, out_dim)

        # 3) 两层 Transformer Encoder (Pre-LN, GELU)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=out_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN，训练更稳定
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # ──────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., in_dim]  or  [..., seq_len, in_dim]
               - 2-D: [B, in_dim]      (e.g. pooled text / audio global)
               - 3-D: [B, T, in_dim]   (e.g. frame-level audio local)

        Returns:
            projected: [..., out_dim]  or  [..., seq_len, out_dim]
        """
        # 记录输入维度以便后续恢复
        orig_ndim = x.dim()

        if orig_ndim == 2:
            x = x.unsqueeze(1)  # [B, 1, in_dim]

        # x: [B, seq_len, in_dim]
        x = self.input_proj(x)  # [B, seq_len, out_dim]

        # 添加可学习的位置编码
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)  # [seq_len]
        pos_emb = self.pos_embed(positions).unsqueeze(0)     # [1, seq_len, out_dim]
        x = x + pos_emb

        # 两层 Transformer
        x = self.transformer(x)  # [B, seq_len, out_dim]

        # 恢复原始维度：2-D 输入 squeeze 回 [B, out_dim]；3-D 保持序列维度
        if orig_ndim == 2:
            x = x.squeeze(1)  # [B, out_dim]

        if self.normalize:
            x = F.normalize(x, p=2, dim=-1)  # [..., out_dim]
        return x
    '''
    
    def forward(self, x):
        x = self.projection(x)
        if self.normalize:
            x = F.normalize(x, p=2, dim=-1)  # [..., out_dim]
        return x