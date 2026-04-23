"""
contrastive.py
──────────────
Contrastive loss functions for GLCLAP training.

Paper (Section 2.2) — loss definitions:

    l(M) = (1/B) * sum_i log( diag(softmax(M))[i] )
         = mean cross-entropy of a B×B similarity matrix M
           where the diagonal corresponds to matched pairs.

    Global contrastive loss (Eq. 3):
        Lg = l(Et · Ea^T) + l(Ea · Et^T)
             [B,D] × [D,B] → [B,B]   (symmetric, both directions)

    Local max-pooling contrastive loss (Eq. 4):
        Ll = l(max_t(Et' · Ea'^T)) + l(max_t(Ea' · Et'^T))

        Et' shape:  [B, D]      (pooled subtext embeddings)
        Ea' shape:  [B, T', D]  (frame-level audio embeddings)

        Et' · Ea'^T:
            Expand Et' to [B, 1, D] and Ea' to [B, D, T'],
            then bmm → [B, B, T'] ... actually we need cross-batch similarity.

        Cross-batch computation (detailed below):
            For text→audio direction:
                S[i, j, t] = Et'[i] · Ea'[j, t]
                           = einsum('id, jtd -> ijt', Et', Ea')  → [B, B, T']
                max over t → [B, B]   ← this is the similarity matrix

            For audio→text direction:
                Symmetric: S.transpose(0, 1) gives [B, B] with (j, i) indexing,
                which is equivalent to l(max_t(Ea'·Et'^T)).

    Total loss (Eq. 5):
        L = Lg + Ll

    For LCLAP (local-only ablation), only Ll is used.

Temperature:
    Paper does not mention temperature explicitly.
    We scale logits by 1/temperature (learnable or fixed) following CLIP/CLAP.
    Default temperature=0.07 (common choice; treat as hyperparameter).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _info_nce(
    logits: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    InfoNCE loss for a square [B, B] similarity matrix.

    Implements: l(M) = (1/B) sum_i log diag(softmax(M/temp))[i]
                     = mean cross-entropy with target = diagonal indices.

    Args:
        logits:      [B, B]  — raw dot-product similarity matrix
        temperature: Scalar temperature to scale logits.

    Returns:
        Scalar loss (mean over batch).
    """
    B = logits.shape[0]
    logits = logits / temperature                     # [B, B]
    labels = torch.arange(B, device=logits.device)   # [B]  diagonal targets
    return F.cross_entropy(logits, labels)            # scalar


def global_contrastive_loss(
    text_global: torch.Tensor,
    audio_global: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Global contrastive loss Lg (Eq. 3).

    Both directions are computed and averaged:
        l(Et · Ea^T) + l(Ea · Et^T)

    Args:
        text_global:  [B, D]  — L2-normalised global text embeddings (Et)
        audio_global: [B, D]  — L2-normalised global audio embeddings (Ea)
        temperature:  Scalar temperature.

    Returns:
        Scalar loss Lg.

    Intermediate shapes:
        sim_t2a: [B, B]  — text→audio similarity
        sim_a2t: [B, B]  — audio→text similarity (== sim_t2a.T)
    """
    # Assumes embeddings are already L2-normalised (done in ProjectionHead).
    sim_t2a = text_global @ audio_global.T    # [B, B]
    sim_a2t = sim_t2a.T                       # [B, B]

    loss_t2a = _info_nce(sim_t2a, temperature)
    loss_a2t = _info_nce(sim_a2t, temperature)

    return (loss_t2a + loss_a2t) / 2.0  # scalar


def local_contrastive_loss(
    text_local: torch.Tensor,
    audio_local: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Local max-pooling contrastive loss Ll (Eq. 4).

    Computes the cross-batch similarity as:
        S[i, j, t] = Et'[i] · Ea'[j, t]   shape [B, B, T']
    Then takes max over t → [B, B], and applies InfoNCE in both directions.

    Args:
        text_local:  [B, D]    — L2-normalised local text embeddings (Et')
        audio_local: [B, T', D]— L2-normalised local audio embeddings (Ea')
        temperature: Scalar temperature.

    Returns:
        Scalar loss Ll.

    Intermediate shapes:
        text_local:   [B, D]     → unsqueeze → [B, 1, D]
        audio_local:  [B, T', D] → permute  → [B, D, T']
        S (t2a):      [B, B, T'] — cross-batch frame-level similarity
        max_sim_t2a:  [B, B]     — max over T'
        max_sim_a2t:  [B, B]     — audio→text direction (transposed)
    """
    B, T_prime, D = audio_local.shape  # B, T', D

    # text→audio: S[i, j, t] = Et'[i] · Ea'[j, t]
    # Reshape for efficient matrix multiply:
    #   text_local:  [B, D]   → [B, 1, D]
    #   audio_local: [B, T', D] → [B, D, T']  then expand cross-batch
    t = text_local.unsqueeze(1)                  # [B, 1, D]
    a = audio_local.permute(0, 2, 1)             # [B, D, T']

    # Cross-batch: for each (i, j) pair compute t[i] · a[j, :, :]
    # Using einsum:  'bid, jdt -> bijt' is wrong shape; instead:
    # S[i, j, t] = sum_d text_local[i, d] * audio_local[j, t, d]
    # = einsum('id, jtd -> ijt', text_local, audio_local)
    S_t2a = torch.einsum("id, jtd -> ijt", text_local, audio_local)  # [B, B, T']
    max_sim_t2a = S_t2a.max(dim=-1).values                            # [B, B]

    # audio→text direction: S[j, i, t] → max over t → [B, B] indexed by (j, i)
    # = S_t2a.transpose(0,1) followed by max — or equivalently:
    S_a2t = S_t2a.permute(1, 0, 2)                                    # [B, B, T']
    max_sim_a2t = S_a2t.max(dim=-1).values                            # [B, B]

    loss_t2a = _info_nce(max_sim_t2a, temperature)
    loss_a2t = _info_nce(max_sim_a2t, temperature)

    return (loss_t2a + loss_a2t) / 2.0  # scalar


def glclap_loss(
    text_global: torch.Tensor | None,
    text_local: torch.Tensor | None,
    audio_global: torch.Tensor | None,
    audio_local: torch.Tensor | None,
    temperature: float = 0.07,
    local_only: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Combined GLCLAP loss: L = Lg + Ll  (Eq. 5).

    Args:
        text_global:  [B, D]      — Et  (global text, L2-normalised)
        text_local:   [B, D]      — Et' (local text/subtext, L2-normalised)
        audio_global: [B, D]      — Ea  (global audio, L2-normalised)
        audio_local:  [B, T', D]  — Ea' (local audio, L2-normalised)
        temperature:  Scalar temperature.
        local_only:   If True, run simplified local-only contrastive learning:
                      subtext [B, D] vs pooled audio [B, D] using standard
                      InfoNCE (global_contrastive_loss). Global branches are
                      closed and the four standard embeddings may be None.

    Returns:
        dict with keys:
            "loss":         total loss scalar
            "loss_global":  Lg  (0.0 tensor if local_only)
            "loss_local":   Ll  (or the sole contrastive loss when local_only)
    """
    if local_only:
        # New local_only behaviour:
        # text_local  : [B, D]  (subtext embedding)
        # audio_local : [B, D]  (pooled audio embedding)
        assert text_local is not None and audio_local is not None
        loss = global_contrastive_loss(text_local, audio_local, temperature)
        return {
            "loss": loss,
            "loss_global": torch.tensor(0.0, device=loss.device),
            "loss_local": loss,
        }

    # Standard GLCLAP mode
    assert text_local is not None and audio_local is not None
    Ll = local_contrastive_loss(text_local, audio_local, temperature)

    assert text_global is not None and audio_global is not None
    Lg = global_contrastive_loss(text_global, audio_global, temperature)
    total = Lg + Ll

    return {
        "loss": total,
        "loss_global": Lg,
        "loss_local": Ll,
    }
