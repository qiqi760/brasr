"""
text_encoder.py
───────────────
BERT-based text encoder for both global and local (subtext) text branches.

Paper (Section 2.2):
    "The text encoder is initialized with bert-base-multilingual-uncased.
     The local and global branches share the same weights and are followed
     by an average pooling layer p(.) to reduce the word dimension."

    Global branch:  Et  = p(ft(Xt))   → [B, D]
    Local branch:   Et' = p(ft(Xt'))  → [B, D]

    Both branches use the SAME TextEncoder instance (shared weights).

Design:
    - ft(.) = BERT transformer → last hidden states [B, N, H]
    - p(.)  = mean over non-padding token positions → [B, H]
    - The raw H-dimensional output is then projected to embed_dim D
      via ProjectionHead (done in GLCLAP, not here).

Args (constructor):
    model_name:   HuggingFace model ID (default: bert-base-multilingual-uncased).
    freeze_layers: Freeze this many bottom encoder layers (0 = train all).
                   Useful when GPU memory is limited.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel


class TextEncoder(nn.Module):
    """
    Wraps a HuggingFace BERT model and applies masked mean pooling.

    Input tensors come from the BERT tokeniser:
        input_ids:      [B, N]  — token indices (0-padded)
        attention_mask: [B, N]  — 1 for real tokens, 0 for padding

    Output:
        pooled: [B, hidden_size]  — mean of real token embeddings

    hidden_size = 768 for bert-base-multilingual-uncased.
    """

    def __init__(
        self,
        model_name: str = "bert-base-multilingual-uncased",
        freeze_layers: int = 0,
    ) -> None:
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_size: int = self.bert.config.hidden_size  # 768

        if freeze_layers > 0:
            self._freeze_bottom_layers(freeze_layers)

    def _freeze_bottom_layers(self, n: int) -> None:
        """Freeze the embedding layer and bottom-n transformer blocks."""
        # If n is large enough to cover all layers, freeze the entire encoder
        total_layers = len(self.bert.encoder.layer)
        if n >= total_layers:
            for param in self.bert.parameters():
                param.requires_grad = False
            return
        # Freeze embeddings
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        # Freeze bottom-n encoder layers
        for layer in self.bert.encoder.layer[:n]:
            for param in layer.parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            input_ids:      [B, N]  — token indices
            attention_mask: [B, N]  — 1 = real token, 0 = padding

        Returns:
            pooled: [B, hidden_size]  — masked mean over token dimension

        Internal shapes:
            last_hidden: [B, N, hidden_size]
            mask_expanded: [B, N, hidden_size]
            sum_embeddings: [B, hidden_size]
            pooled: [B, hidden_size]
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        last_hidden = outputs.last_hidden_state  # [B, N, hidden_size]

        # Masked mean pooling: average only over real (non-pad) tokens
        mask_expanded = attention_mask.unsqueeze(-1).float()  # [B, N, 1]
        sum_embeddings = (last_hidden * mask_expanded).sum(dim=1)  # [B, hidden_size]
        token_counts   = mask_expanded.sum(dim=1).clamp(min=1e-9)  # [B, 1]
        pooled = sum_embeddings / token_counts                      # [B, hidden_size]

        return pooled  # [B, 768]
