"""
audio_encoder.py
────────────────
Data2Vec-based audio encoder for GLCLAP.

Paper (Section 3.3):
    "We utilize the same structure and pre-training method as
     Data2Vec2.0-large. Specifically, the Data2VecAudioModel is employed,
     which is a transformer-based architecture designed for self-supervised
     learning of speech representations."

    Audio encoder input:  Xa ∈ R^{B × T × F}  (Mel spectrogram, or raw PCM)
    Local audio output:   Ea' = fa(Xa) → [B, T//4, hidden_size]
    Global audio output:  Ea  = mean_pool(Ea') → [B, hidden_size]

    Data2Vec-large hidden_size = 1024.
    The 4× downsampling comes from the CNN feature extractor in Data2Vec.

Notes:
    The paper uses a privately pre-trained Data2Vec2.0-large.
    We default to the public "facebook/data2vec-audio-large-960h" checkpoint
    as the closest publicly available substitute.

    The HuggingFace Data2VecAudioModel accepts raw waveform (float32, [B, T])
    through Wav2Vec2Processor and returns last_hidden_state [B, T', D].
    T' ≈ T // (stride_product_of_CNN_layers) ≈ T // 320 for 16kHz audio.

    TODO: Verify exact downsampling factor with the private checkpoint.
          Paper states T//4 for mel input; HF model may differ.
          If mel features are used as input, a custom CNN frontend must be added.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import Data2VecAudioModel


class AudioEncoder(nn.Module):
    """
    Wraps a HuggingFace Data2VecAudioModel.

    Accepts raw waveform tensors (float32, values in [-1, 1]).
    Returns both local (frame-level) and global (mean-pooled) embeddings.

    Args:
        model_name:    HuggingFace model ID for Data2Vec audio.
        freeze_layers: Freeze bottom-N transformer layers (0 = train all).

    Input:
        waveform:          [B, T_samples]  — raw PCM, float32
        attention_mask:    [B, T_samples]  — optional, 1=valid sample, 0=pad

    Outputs:
        local_emb:  [B, T', hidden_size]  — frame-level features (Ea')
        global_emb: [B, hidden_size]      — mean-pooled over T' (Ea)

    where T' = T_samples // downsampling_factor  (≈T//320 for HF model,
    or T//4 if using a mel-based frontend — see TODO above).
    """

    def __init__(
        self,
        model_name: str = "facebook/data2vec-audio-large-960h",
        freeze_layers: int = 0,
    ) -> None:
        super().__init__()
        self.model = Data2VecAudioModel.from_pretrained(model_name)
        self.hidden_size: int = self.model.config.hidden_size  # 1024 for large

        if freeze_layers > 0:
            self._freeze_bottom_layers(freeze_layers)

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
            local_emb:  [B, T', hidden_size]  — Ea' in the paper
            global_emb: [B, hidden_size]      — Ea in the paper

        Internal shapes:
            outputs.last_hidden_state: [B, T', hidden_size]
            global_emb via mean over dim=1: [B, hidden_size]
        """
        outputs = self.model(
            input_values=waveform,          # [B, T_samples]
            attention_mask=attention_mask,  # [B, T_samples] or None
        )
        local_emb = outputs.last_hidden_state   # [B, T', hidden_size]
        global_emb = local_emb.mean(dim=1)      # [B, hidden_size]

        return local_emb, global_emb            # ([B, T', D], [B, D])
