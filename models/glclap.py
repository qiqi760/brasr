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
from typing import Optional

import torch
import torch.nn as nn

from .audio_encoder import AudioEncoder
from .projections import ProjectionHead
from .text_encoder import TextEncoder


@dataclass
class GLCLAPOutput:
    """
    Container for all four embeddings produced by a forward pass.

    Shapes:
        text_global:   [B, embed_dim]     — Et  (projected)
        text_local:    [B, embed_dim]     — Et' (projected)
        audio_global:  [B, embed_dim]     — Ea  (projected, mean-pooled)
        audio_local:   [B, T', embed_dim] — Ea' (projected, frame-level)
    """
    text_global:  torch.Tensor   # [B, D]
    text_local:   torch.Tensor   # [B, D]
    audio_global: torch.Tensor   # [B, D]
    audio_local:  torch.Tensor   # [B, T', D]


class GLCLAP(nn.Module):
    """
    Global-Local Contrastive Language-Audio Pre-training model.

    Args:
        text_model_name:   HuggingFace BERT variant identifier.
        audio_model_name:  HuggingFace Data2Vec audio variant identifier.
        embed_dim:         Shared projection dimension D (default 512).
        text_freeze_layers:  Number of bottom BERT layers to freeze.
        audio_freeze_layers: Number of bottom D2V layers to freeze.

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
    ) -> None:
        super().__init__()

        self.text_encoder = TextEncoder(
            model_name=text_model_name,
            freeze_layers=text_freeze_layers,
        )
        self.audio_encoder = AudioEncoder(
            model_name=audio_model_name,
            freeze_layers=audio_freeze_layers,
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
        pooled = self.text_encoder(input_ids, attention_mask)  # [B, 768]
        return self.text_proj(pooled)                          # [B, D]

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
        local_emb, global_emb = self.audio_encoder(waveform, attention_mask)
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

        Returns:
            GLCLAPOutput:
                .text_global:  [B, D]     — Et  projected
                .text_local:   [B, D]     — Et' projected
                .audio_global: [B, D]     — Ea  projected
                .audio_local:  [B, T', D] — Ea' projected
        """
        
        # ── Text branches (shared encoder weights) ──
        text_global  = self.encode_text(text_input_ids,    text_attention_mask)    # [B, D]
        text_local   = self.encode_text(subtext_input_ids, subtext_attention_mask) # [B, D]

        # ── Audio branch ──
        audio_local, audio_global = self.encode_audio(
            waveform,
            attention_mask=waveform_attention_mask,
            pool=False,
        )  # ([B, T', D], [B, D])

        return GLCLAPOutput(
            text_global=text_global,    # [B, D]
            text_local=text_local,      # [B, D]
            audio_global=audio_global,  # [B, D]
            audio_local=audio_local,    # [B, T', D]
        )
