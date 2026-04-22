"""
projections.py
──────────────
Linear projection heads that map encoder outputs into the shared
embedding space of dimension D (embed_dim).

Paper (Section 2.2):
    Both text encoder (BERT, hidden=768) and audio encoder
    (Data2Vec-large, hidden=1024) produce embeddings of different
    dimensions. A projection layer aligns them to a common D.

    Paper does not explicitly describe the projection architecture;
    we assume a single linear layer (no activation) following the
    convention from CLIP/CLAP literature.

    TODO: Replace with a 2-layer MLP + GeLU if ablation shows benefit.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """
    Single linear projection with optional L2-normalisation.

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
        input:  [..., in_dim]
        output: [..., out_dim]   (broadcast-safe for 2-D and 3-D inputs)
    """

    def __init__(self, in_dim: int, out_dim: int, normalize: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.normalize = normalize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., in_dim]

        Returns:
            projected: [..., out_dim], L2-normalised if self.normalize.
        """
        x = self.linear(x)                          # [..., out_dim]
        if self.normalize:
            x = F.normalize(x, p=2, dim=-1)         # [..., out_dim]
        return x
