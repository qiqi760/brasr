"""
retriever.py
────────────
GLCLAP-based bias-word retrieval for contextual biasing ASR (Section 2.3).

Inference pipeline (Figure 3 in the paper):

    1. Encode bias word list Xt_1 … Xt_K → text embeddings [K, D]
    2. Encode audio Xa_i (no average pooling) → local audio embedding [T', D]
    3. Compute similarity matrix: Sim = Et @ Ea'^T  → [K, T']
    4. Max-pool over time: sim_score = Sim.max(dim=1) → [K]
    5. Select words with sim_score > threshold → bias prompt

The retrieved bias words are then passed as prompts to the downstream ASR model
(e.g. Whisper) to improve recognition of rare / named entities.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from models.glclap import GLCLAP

logger = logging.getLogger(__name__)


class BiasWordRetriever:
    """
    Stateful retriever that caches encoded bias-list embeddings for fast lookup.

    Usage:
        retriever = BiasWordRetriever(model, tokenizer, device="cuda")
        retriever.set_bias_list(["Taylor Swift", "Obama", "FIFA"])
        results = retriever.retrieve(waveform)  # → ["Taylor Swift"]

    Args:
        model:       Trained GLCLAP model instance (eval mode).
        tokenizer:   BERT tokenizer matching the text encoder.
        threshold:   Similarity threshold for selection (paper: not specified;
                     treat as tunable hyperparameter, default 0.5).
        top_k:       Maximum number of bias words to return per utterance.
        device:      Compute device.
        max_text_len: Max token length when encoding bias words.
    """

    def __init__(
        self,
        model: GLCLAP,
        tokenizer: AutoTokenizer,
        threshold: float = 0.5,
        top_k: int = 10,
        device: str = "cuda",
        max_text_len: int = 32,
    ) -> None:
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.threshold = threshold
        self.top_k = top_k
        self.device = device
        self.max_text_len = max_text_len

        # Cached bias-list state
        self._bias_words: list[str] = []
        self._bias_embeddings: Optional[torch.Tensor] = None  # [K, D]

    # ──────────────────────────────────────────────────────────────────────
    # Bias list management
    # ──────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def set_bias_list(self, words: list[str]) -> None:
        """
        Encode the user-defined bias word list and cache the embeddings.

        This is called once per context; the embeddings are reused across
        multiple audio inputs.

        Args:
            words: List of K bias phrases/words, e.g. ["Taylor Swift", "FIFA"].

        Side effects:
            Sets self._bias_words and self._bias_embeddings [K, D].
        """
        self._bias_words = words
        if not words:
            self._bias_embeddings = None
            return

        enc = self.tokenizer(
            words,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"].to(self.device)       # [K, N]
        attention_mask = enc["attention_mask"].to(self.device)  # [K, N]

        # Encode: [K, N] → [K, D]  (projected, L2-normalised)
        self._bias_embeddings = self.model.encode_text(input_ids, attention_mask)
        # self._bias_embeddings: [K, D]

        logger.info(f"Encoded {len(words)} bias words → embeddings [{len(words)}, {self._bias_embeddings.shape[-1]}]")

    # ──────────────────────────────────────────────────────────────────────
    # Single-utterance retrieval
    # ──────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def retrieve(
        self,
        waveform: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> list[str]:
        """
        Retrieve matching bias words for a single audio input.

        Args:
            waveform:       [T_samples]  or  [1, T_samples]  — raw PCM float32
            attention_mask: [T_samples]  or  None

        Returns:
            List of selected bias words (subset of self._bias_words).

        Shapes:
            waveform (after squeeze): [1, T_samples]
            audio_local:  [1, T', D]  → squeeze → [T', D]
            Et (bias):    [K, D]
            Sim:          [K, T']
            sim_score:    [K]  (max over T')
            selected:     list of words where sim_score > threshold
        """
        if self._bias_embeddings is None or len(self._bias_words) == 0:
            logger.warning("Bias list is empty. Call set_bias_list() first.")
            return []

        # Ensure [1, T_samples]
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)      # [1, T_samples]
        waveform = waveform.to(self.device)

        if attention_mask is not None:
            if attention_mask.dim() == 1:
                attention_mask = attention_mask.unsqueeze(0)  # [1, T_samples]
            attention_mask = attention_mask.to(self.device)

        # Encode audio — no pooling → local frame embeddings
        audio_local, _ = self.model.encode_audio(
            waveform, attention_mask=attention_mask, pool=False
        )
        # audio_local: [1, T', D]
        audio_local = audio_local.squeeze(0)  # [T', D]

        # Similarity: Et [K, D] @ Ea'^T [D, T'] → [K, T']
        Et = self._bias_embeddings   # [K, D]
        Sim = Et @ audio_local.T     # [K, T']

        # Max-pool over time → [K]
        sim_score, _ = Sim.max(dim=-1)  # [K]

        # Select words above threshold (cap at top_k)
        selected = []
        scores_sorted = sim_score.argsort(descending=True)
        for idx in scores_sorted:
            if len(selected) >= self.top_k:
                break
            if sim_score[idx].item() >= self.threshold:
                selected.append(self._bias_words[idx.item()])

        return selected

    # ──────────────────────────────────────────────────────────────────────
    # Batch retrieval
    # ──────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def retrieve_batch(
        self,
        waveforms: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> list[list[str]]:
        """
        Retrieve bias words for a batch of audio inputs.

        Args:
            waveforms:      [B, T_samples]  — padded batch of PCM
            attention_mask: [B, T_samples]  — optional padding mask

        Returns:
            List of B lists; each inner list contains selected bias words.

        Shapes:
            audio_local_batch: [B, T', D]
            Et:                [K, D]
            Sim:               [B, K, T']  — using einsum
            sim_scores:        [B, K]      — max over T'
        """
        if self._bias_embeddings is None or len(self._bias_words) == 0:
            B = waveforms.shape[0]
            return [[] for _ in range(B)]

        waveforms = waveforms.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        audio_local_batch, _ = self.model.encode_audio(
            waveforms, attention_mask=attention_mask, pool=False
        )
        # audio_local_batch: [B, T', D]

        Et = self._bias_embeddings    # [K, D]

        # Sim[b, k, t] = Et[k] · audio_local[b, t]
        Sim = torch.einsum("kd, btd -> bkt", Et, audio_local_batch)  # [B, K, T']
        sim_scores, _ = Sim.max(dim=-1)  # [B, K]

        results = []
        for b in range(sim_scores.shape[0]):
            scores = sim_scores[b]  # [K]
            selected = []
            for idx in scores.argsort(descending=True):
                if len(selected) >= self.top_k:
                    break
                if scores[idx].item() >= self.threshold:
                    selected.append(self._bias_words[idx.item()])
            results.append(selected)

        return results

    # ──────────────────────────────────────────────────────────────────────
    # Similarity matrix (for visualisation / debugging, Figure 4)
    # ──────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def similarity_matrix(
        self,
        waveform: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the full [K, T'] similarity matrix for inspection / Figure 4.

        Returns:
            Sim: [K, T'] CPU tensor
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        waveform = waveform.to(self.device)

        audio_local, _ = self.model.encode_audio(
            waveform, attention_mask=attention_mask, pool=False
        )
        audio_local = audio_local.squeeze(0)  # [T', D]

        Et = self._bias_embeddings            # [K, D]
        Sim = Et @ audio_local.T              # [K, T']
        return Sim.cpu()
