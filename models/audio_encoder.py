"""
audio_encoder.py
────────────────
Data2Vec-based audio encoder for GLCLAP.
(Adds an extra 4× pooling downsampling after HF model output)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import Data2VecAudioModel

from .projections import AttentionPooling


class AudioEncoder(nn.Module):
    """
    Wraps HF Data2VecAudioModel and adds an optional extra pooling downsampling.

    Args:
        model_name:           HuggingFace model ID for Data2Vec audio.
        freeze_layers:        Freeze bottom-N transformer layers (0 = train all).
        post_downsample_rate: Extra pooling factor after HF output (default 4).
                              Set to 1 to disable.
        use_attention_pooling: If True, replace mean pooling with AttentionPooling
                              (arXiv:2505.19179). Default False for backward compat.

    Input:
        waveform:          [B, T_samples]  — raw PCM, float32
        attention_mask:    [B, T_samples]  — optional, 1=valid sample, 0=pad

    Outputs:
        local_emb:  [B, T'', hidden_size]  — frame-level features (Ea')
        global_emb: [B, hidden_size]       — pooled over T'' (Ea)

    where:
        T'   = T_samples // (HF downsampling factor)  (~T/320)
        T''  = T' // post_downsample_rate
    """

    def __init__(
        self,
        model_name: str = "facebook/data2vec-audio-large-960h",
        freeze_layers: int = 0,
        post_downsample_rate: int = 4,
        use_attention_pooling: bool = False,
    ) -> None:
        super().__init__()
        self.model = Data2VecAudioModel.from_pretrained(model_name)
        self.hidden_size: int = self.model.config.hidden_size  # 1024 for large
        self.post_downsample_rate = post_downsample_rate
        self.use_attention_pooling = use_attention_pooling

        if freeze_layers > 0:
            self._freeze_bottom_layers(freeze_layers)

        # Extra pooling layer (applied on time dimension)
        if post_downsample_rate > 1:
            self.pool = nn.AvgPool1d(
                kernel_size=post_downsample_rate,
                stride=post_downsample_rate,
                ceil_mode=False,          # Discard incomplete trailing window
            )
        else:
            self.pool = nn.Identity()

        # Attention Pooling (arXiv:2505.19179) to replace mean pooling
        if use_attention_pooling:
            self.attn_pool = AttentionPooling(
                hidden_size=self.hidden_size,
                num_heads=1,
                attn_dim=self.hidden_size,
            )
        else:
            self.attn_pool = None

    def _freeze_bottom_layers(self, n: int) -> None:
        """Freeze the CNN feature extractor and bottom-n transformer layers."""
        total_layers = len(self.model.encoder.layers)
        if n >= total_layers:
            for param in self.model.parameters():
                param.requires_grad = False
            return
        for param in self.model.feature_extractor.parameters():
            param.requires_grad = False
        for param in self.model.feature_projection.parameters():
            param.requires_grad = False
        for layer in self.model.encoder.layers[:n]:
            for param in layer.parameters():
                param.requires_grad = False

    def forward(
        self,
        waveform: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            waveform:       [B, T_samples]  — raw PCM float32
            attention_mask: [B, T_samples]  — optional padding mask

        Returns:
            local_emb:  [B, T'', hidden_size]  — after extra pooling
            global_emb: [B, hidden_size]
        """
        # Step 1: HF model forward
        outputs = self.model(
            input_values=waveform,
            attention_mask=attention_mask,
        )
        local_emb = outputs.last_hidden_state   # [B, T', hidden_size]

        # Step 2: Extra pooling downsampling (if enabled)
        if self.post_downsample_rate > 1:
            # local_emb: [B, T', D] -> permute to [B, D, T'] for Conv1d pooling
            local_emb = local_emb.permute(0, 2, 1)          # [B, D, T']
            local_emb = self.pool(local_emb)                # [B, D, T'']
            local_emb = local_emb.permute(0, 2, 1)          # [B, T'', D]

        # Step 3: Global embedding via pooling over time
        if self.attn_pool is not None:
            global_emb = self.attn_pool(local_emb)          # [B, D]
        else:
            global_emb = local_emb.mean(dim=1)              # [B, D]

        return local_emb, global_emb
    
#CTC头约束
#解冻