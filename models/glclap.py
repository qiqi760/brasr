"""
glclap.py
─────────
Top-level GLCLAP model: wires together text encoder, audio encoder,
and projection heads. Produces the four embeddings needed for loss computation.

Paper (Section 2.2) — embedding equations:

    Text:
        Et  = p(ft(Xt))   → global text embedding  [B, D]
        Et' = p(ft(Xt'))  → local  text embedding  [B, D]
        (shared ft weights; p = mean pooling done inside TextEncoder)

    Audio:
        Ea' = fa(Xa)       → local  audio embedding [B, T', D]
        Ea  = p(Ea')       → global audio embedding [B, D]

    Projection:
        All embeddings are projected to D via separate linear heads.

Loss computation is delegated to losses/contrastive.py.

                    ┌────────────────────────┐
        Xt  ──────► │  TextEncoder (BERT)     │──► pool ──► text_proj ──► Et  [B, D]
        Xt' ──────► │  (shared weights)       │──► pool ──► text_proj ──► Et' [B, D]
                    └────────────────────────┘
                    ┌────────────────────────┐
        Xa  ──────► │  AudioEncoder (D2V)    │──► Ea'[B,T',H] ──► audio_proj ──► Ea'_proj [B,T',D]
                    │                        │──► mean ──► Ea [B,H] ──► audio_proj ──► Ea_proj [B,D]
                    └────────────────────────┘
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch
import torch.nn as nn

from .audio_encoder import AudioEncoder
from .projections import ProjectionHead
from .text_encoder import TextEncoder


@dataclass
class GLCLAPOutput:
    """
    Container for all embeddings produced by a forward pass.

    Shapes (standard GLCLAP mode):
        text_global:   [B, embed_dim]     — Et  (projected)
        text_local:    [B, embed_dim]     — Et' (projected)
        audio_global:  [B, embed_dim]     — Ea  (projected, mean-pooled)
        audio_local:   [B, T', embed_dim] — Ea' (projected, frame-level)

    Shapes (local_only=True  simplified mode):
        text_local:    [B, embed_dim]     — pooled subtext embedding
        audio_local:   [B, embed_dim]     — pooled audio embedding
        text_global / audio_global:  None
    """
    text_global:  Optional[torch.Tensor] = None   # [B, D]
    text_local:   Optional[torch.Tensor] = None   # [B, D]
    audio_global: Optional[torch.Tensor] = None   # [B, D]
    audio_local:  Optional[torch.Tensor] = None   # [B, D] or [B, T', D]


class GLCLAP(nn.Module):
    """
    Global-Local Contrastive Language-Audio Pre-training model.

    Args:
        text_model_name:   HuggingFace BERT variant identifier.
        audio_model_name:  HuggingFace Data2Vec audio variant identifier.
        embed_dim:         Shared projection dimension D (default 512).
        text_freeze_layers:  Number of bottom BERT layers to freeze.
        audio_freeze_layers: Number of bottom D2V layers to freeze.
        text_use_attention_pooling: If True, replace mean pooling with
                                    AttentionPooling in text encoder.
        audio_use_attention_pooling: If True, replace mean pooling with
                                     AttentionPooling in audio encoder.

    Forward inputs:
        text_input_ids:           [B, N]           — global text tokens
        text_attention_mask:      [B, N]
        subtext_input_ids:        [B, N']           — local subtext tokens
        subtext_attention_mask:   [B, N']
        waveform:                 [B, T_samples]    — raw PCM float32

    Forward output:
        GLCLAPOutput with four projected, L2-normalised embedding tensors.
    """

    def __init__(
        self,
        text_model_name: str = "bert-base-multilingual-uncased",
        audio_model_name: str = "facebook/data2vec-audio-large-960h",
        embed_dim: int = 512,
        text_freeze_layers: int = 0,
        audio_freeze_layers: int = 0,
        detach_encoders: bool = False,
        text_use_attention_pooling: bool = False,
        audio_use_attention_pooling: bool = False,
    ) -> None:
        super().__init__()

        self.text_encoder = TextEncoder(
            model_name=text_model_name,
            freeze_layers=text_freeze_layers,
            use_attention_pooling=text_use_attention_pooling,
        )
        self.audio_encoder = AudioEncoder(
            model_name=audio_model_name,
            freeze_layers=audio_freeze_layers,
            use_attention_pooling=audio_use_attention_pooling,
        )

        # Projection heads
        # text hidden: 768 (bert-base) → embed_dim
        self.text_proj = ProjectionHead(
            in_dim=self.text_encoder.hidden_size,
            out_dim=embed_dim,
            normalize=True,
        )
        # audio hidden: 1024 (data2vec-large) → embed_dim
        # Same projection is applied to both local [B, T', H] and global [B, H]
        self.audio_proj = ProjectionHead(
            in_dim=self.audio_encoder.hidden_size,
            out_dim=embed_dim,
            normalize=True,
        )

        self.embed_dim = embed_dim
        self.detach_encoders = detach_encoders

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode a tokenised text batch and project to embed_dim.

        Args:
            input_ids:      [B, N]
            attention_mask: [B, N]

        Returns:
            [B, embed_dim]  — L2-normalised
        """
        # detach_body=True runs the frozen BERT inside torch.no_grad() while
        # leaving AttentionPooling (if any) on the computation graph so it
        # still receives gradients.
        pooled = self.text_encoder(
            input_ids, attention_mask, detach_body=self.detach_encoders
        )  # [B, 768]
        return self.text_proj(pooled)                              # [B, D]

    def _encode_with_dedup(
        self,
        sample_ids: Optional[torch.Tensor],
        encode_fn: Callable,
        *tensor_args: torch.Tensor,
        **kwargs: Any,
    ) -> Any:
        """
        对展平输入按 sample_ids 去重后编码，再将 embedding 复制回 B*K。

        当 flatten_subtexts=True 时，同一音频/全局文本会被复制 K 次。
        为避免重复 forward 浪费计算与显存，本方法先提取每个原始样本的
        唯一输入，编码一次后再按 sample_ids 映射回展平维度。

        MODIFIED (2026-05-14): 新增。

        Args:
            sample_ids:  [B*K] 分组标识；None 时退化为逐条编码（兼容旧逻辑）。
            encode_fn:   self.encode_text 或 self.encode_audio 等编码方法。
            *tensor_args: 传给 encode_fn 的张量参数，每个都是 [B*K, ...]。
            **kwargs:    传给 encode_fn 的额外关键字参数。

        Returns:
            与 encode_fn 返回值同结构，但 batch 维度已恢复为 B*K。
        """
        if sample_ids is None:
            return encode_fn(*tensor_args, **kwargs)

        # 计算每个原始样本第一次出现的位置（sample_ids 在 collate_fn 中已排序）
        first_mask = torch.cat([
            sample_ids.new_tensor([True]),
            sample_ids[1:] != sample_ids[:-1]
        ])
        unique_idx = torch.where(first_mask)[0]

        # 取唯一输入，编码一次
        unique_tensors = [t[unique_idx] for t in tensor_args]
        unique_out = encode_fn(*unique_tensors, **kwargs)

        # 构建逆映射：每个展平位置对应 unique 中的哪个索引
        _, inverse = torch.unique(sample_ids, return_inverse=True)

        # 将输出复制回 B*K
        if isinstance(unique_out, tuple):
            return tuple(u[inverse] for u in unique_out)
        return unique_out[inverse]

    # ──────────────────────────────────────────────────────────────────

    def encode_audio(
        self,
        waveform: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pool: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a waveform batch and project to embed_dim.

        Args:
            waveform:       [B, T_samples]
            attention_mask: [B, T_samples]  — optional
            pool:           If True return only global [B, D];
                            if False return (local [B, T', D], global [B, D]).

        Returns:
            When pool=True:  [B, embed_dim]
            When pool=False: ([B, T', embed_dim], [B, embed_dim])
        """
        # detach_body=True runs the frozen Data2Vec body inside torch.no_grad()
        # while leaving AttentionPooling (if any) on the computation graph. 
        local_emb, global_emb = self.audio_encoder(
            waveform, attention_mask, detach_body=self.detach_encoders
        )
        # local_emb:  [B, T', hidden_size]
        # global_emb: [B, hidden_size]

        local_proj  = self.audio_proj(local_emb)   # [B, T', D]
        global_proj = self.audio_proj(global_emb)  # [B, D]

        if pool:
            return global_proj  # [B, D]
        return local_proj, global_proj  # ([B, T', D], [B, D])

    def forward(
        self,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        subtext_input_ids: torch.Tensor,
        subtext_attention_mask: torch.Tensor,
        waveform: torch.Tensor,
        waveform_attention_mask: Optional[torch.Tensor] = None,
        local_only: bool = False,
        sample_ids: Optional[torch.Tensor] = None,
    ) -> GLCLAPOutput:
        """
        Full GLCLAP forward pass.

        Args:
            text_input_ids:           [B, N]
            text_attention_mask:      [B, N]
            subtext_input_ids:        [B, N']
            subtext_attention_mask:   [B, N']
            waveform:                 [B, T_samples]
            waveform_attention_mask:  [B, T_samples]  (optional)
            local_only:               If True, close global branches and return
                                      only pooled subtext [B, D] and pooled
                                      audio [B, D] for simplified contrastive
                                      learning.
            sample_ids:               [B*K] 分组标识（展平模式时提供）。
                                      用于对同一音频/全局文本去重编码，
                                      避免重复 forward。None 时逐条编码。

        Returns:
            GLCLAPOutput — see class docstring for per-mode field shapes.
        """

        if local_only:
            # ── Local-only simplified mode ──
            # Close global branches; keep only subtext (local) and pooled audio.
            text_local = self.encode_text(
                subtext_input_ids, subtext_attention_mask
            )  # [B, D]

            # Audio: pooled global audio embedding [B, D]
            # MODIFIED (2026-05-14): 使用 _encode_with_dedup 避免同一音频重复编码
            if waveform_attention_mask is not None:
                audio_local = self._encode_with_dedup(
                    sample_ids,
                    self.encode_audio,
                    waveform,
                    waveform_attention_mask,
                    pool=True,
                )
            else:
                audio_local = self._encode_with_dedup(
                    sample_ids,
                    self.encode_audio,
                    waveform,
                    pool=True,
                )
            # ──────────────────────────────────────────────────────────

            return GLCLAPOutput(
                text_local=text_local,    # [B, D]
                audio_local=audio_local,  # [B, D]
            )

        # ── Standard GLCLAP mode ──
        # Text branches (shared encoder weights)
        # MODIFIED (2026-05-14): 全局文本去重编码，subtext 保持逐条
        text_global = self._encode_with_dedup(
            sample_ids,
            self.encode_text,
            text_input_ids,
            text_attention_mask,
        )  # [B, D]
        text_local = self.encode_text(
            subtext_input_ids, subtext_attention_mask
        )  # [B, D]
        # ──────────────────────────────────────────────────────────────

        # Audio branch
        # MODIFIED (2026-05-14): 使用 _encode_with_dedup 避免同一音频重复编码
        if waveform_attention_mask is not None:
            audio_local, audio_global = self._encode_with_dedup(
                sample_ids,
                self.encode_audio,
                waveform,
                waveform_attention_mask,
                pool=False,
            )
        else:
            audio_local, audio_global = self._encode_with_dedup(
                sample_ids,
                self.encode_audio,
                waveform,
                pool=False,
            )
        # ──────────────────────────────────────────────────────────────

        return GLCLAPOutput(
            text_global=text_global,    # [B, D]
            text_local=text_local,      # [B, D]
            audio_global=audio_global,  # [B, D]
            audio_local=audio_local,    # [B, T', D]
        )
