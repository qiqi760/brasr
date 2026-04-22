# GLCLAP — Reproduction

Faithful PyTorch reproduction of:

> **GLCLAP: A Novel Contrastive Learning Pre-trained Model for Contextual Biasing in ASR**  
> Kong et al., Interspeech 2025

---

## Project Structure

```
GLCLAP/
├── configs/
│   ├── model_config.yaml      # encoder names, projection dim, subtext settings
│   └── train_config.yaml      # datasets, optimizer, scheduler, training hyper-params
│
├── data/
│   ├── dataset.py             # GLCLAPDataset + collate_fn + build_dataloader
│   ├── subtext.py             # random contiguous word-span sampling (Section 2.1)
│   └── audio_utils.py         # waveform loading, Mel spectrogram extraction
│
├── models/
│   ├── text_encoder.py        # BERT with masked mean pooling → [B, 768]
│   ├── audio_encoder.py       # Data2Vec audio → local [B,T',1024] + global [B,1024]
│   ├── projections.py         # Linear projection head → [*, embed_dim]
│   └── glclap.py              # Top-level model: wires encoders + projection heads
│
├── losses/
│   └── contrastive.py         # global_loss (Lg), local_loss (Ll), glclap_loss (L=Lg+Ll)
│
├── training/
│   ├── trainer.py             # Train/val loops, AMP, grad-clip, early stopping
│   └── scheduler.py           # Cosine/linear LR schedule with warmup
│
├── inference/
│   └── retriever.py           # BiasWordRetriever: encodes bias list, retrieves matches
│
├── eval/
│   └── metrics.py             # top1_recall, precision/recall/F1, WER wrapper
│
├── utils/
│   ├── logging.py             # Logging setup
│   └── checkpointing.py       # save/load checkpoint
│
├── scripts/
│   ├── train.py               # Main training entry point
│   ├── evaluate.py            # Batch evaluation (top-1 recall, F1)
│   └── infer.py               # Single-file inference + prompt generation
│
└── requirements.txt
```

---

## System Data Flow

```
                              ┌─────────────────────────────────────────┐
                              │               Training                   │
                              └─────────────────────────────────────────┘

  JSONL manifest
  ─────────────
  {audio_path, text}
        │
        ▼
  GLCLAPDataset.__getitem__
  ├─ load_audio()            → waveform [1, T_samples]
  └─ sample_subtext()        → subtext string
        │
        ▼
  collate_fn (BERT tokenizer)
  ├─ padded waveforms        [B, 1, T_max]
  ├─ text_input_ids          [B, N]       ← global text
  └─ subtext_input_ids       [B, N']      ← local subtext
        │
        ▼
  GLCLAP.forward()
  ├─ TextEncoder(text)       → pool → text_proj   → Et   [B, D]
  ├─ TextEncoder(subtext)    → pool → text_proj   → Et'  [B, D]   (shared weights)
  └─ AudioEncoder(waveform)
       ├─ local  Ea'         → audio_proj          → [B, T', D]
       └─ global Ea (mean)   → audio_proj          → [B, D]
        │
        ▼
  glclap_loss()
  ├─ Lg = InfoNCE(Et·Ea^T) + InfoNCE(Ea·Et^T)        (global branch)
  ├─ Ll = InfoNCE(max_t(Et'·Ea'^T)) + symmetric       (local branch)
  └─ L  = Lg + Ll
        │
        ▼
  AdamW optimizer + cosine LR schedule + AMP

                              ┌─────────────────────────────────────────┐
                              │               Inference                   │
                              └─────────────────────────────────────────┘

  bias_list = ["Taylor Swift", "FIFA", ...]   (K phrases)
        │
        ▼
  BiasWordRetriever.set_bias_list()
  └─ TextEncoder + text_proj → cached Et  [K, D]

  audio file
        │
        ▼
  AudioEncoder (no pooling)   → audio_proj → Ea' [T', D]
        │
        ▼
  Sim = Et @ Ea'^T            → [K, T']
  sim_score = Sim.max(dim=1)  → [K]
  selected = words where sim_score > threshold
        │
        ▼
  Prompt for ASR (e.g. Whisper): "Taylor Swift, ..."
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare manifests

Create a JSONL file for each dataset split:
```json
{"audio_path": "train-clean-100/1234/5678/1234-5678-0001.flac", "text": "have you ever heard taylor swift's songs"}
```

### 3. Train

```bash
python scripts/train.py \
    --model_config configs/model_config.yaml \
    --train_config configs/train_config.yaml
```

Train LCLAP (local-only ablation):
```bash
python scripts/train.py --local_only
```

Resume from checkpoint:
```bash
python scripts/train.py --resume outputs/glclap/checkpoint_epoch010.pt
```

### 4. Evaluate

```bash
python scripts/evaluate.py \
    --checkpoint outputs/glclap/best_model.pt \
    --manifest   data/manifests/aishell1_test_nt.jsonl \
    --bias_list  data/bias_lists/aishell1_bias.txt \
    --threshold  0.5
```

### 5. Infer (single file)

```bash
python scripts/infer.py \
    --checkpoint outputs/glclap/best_model.pt \
    --audio      /path/to/audio.wav \
    --bias_list  data/bias_lists/phonecall.txt
```

---

## Key Tensor Shapes

| Symbol | Meaning | Shape |
|--------|---------|-------|
| `Xt`   | Global text tokens (batch) | `[B, N]` |
| `Xt'`  | Local subtext tokens (batch) | `[B, N']`, N'≤N |
| `Xa`   | Waveform (batch) | `[B, T_samples]` |
| `Et`   | Global text embedding | `[B, D]` |
| `Et'`  | Local text embedding | `[B, D]` |
| `Ea`   | Global audio embedding | `[B, D]` |
| `Ea'`  | Local audio embedding (frame-level) | `[B, T', D]` |
| `Sim`  | Similarity matrix (inference) | `[K, T']` |
| D      | Joint embedding dim | 512 (default) |

---

## Implementation Assumptions / Open Questions

1. **Subtext sampling**: Paper says "randomly extract sub-text". We assume a contiguous word span of length ∈ [1, 5]. Exact algorithm not specified.
2. **Temperature**: Paper does not state temperature for InfoNCE. Default: 0.07 (CLIP convention). Treat as hyperparameter.
3. **Projection**: Paper shows a projection layer but does not specify depth. We use a single linear layer (no activation). A 2-layer MLP with GeLU is a TODO fallback.
4. **Audio encoder**: Paper uses a *privately* pre-trained Data2Vec2.0-large. We use `facebook/data2vec-audio-large-960h` as the closest public substitute.
5. **Downsampling factor**: Paper states T//4 for Mel input; the public HF Data2Vec model uses raw PCM with ~320× downsampling. Verify with your checkpoint.
6. **Similarity threshold**: Paper does not specify the value used for bias-word selection. Tune on a held-out validation set.
7. **Multi-dataset training**: ConcatDataset or round-robin DataLoader is implemented as TODO in `scripts/train.py`.
8. **Validation split**: Paper does not describe a validation procedure. Early stopping uses val loss; supply your own val manifest.

---

## Recommended Implementation Order

1. `data/subtext.py` + `data/audio_utils.py` — verify with unit tests
2. `data/dataset.py` — verify DataLoader output shapes
3. `models/text_encoder.py` — spot-check pooled shape [B, 768]
4. `models/audio_encoder.py` — spot-check local/global shapes
5. `models/projections.py` + `models/glclap.py` — end-to-end forward pass
6. `losses/contrastive.py` — verify Lg and Ll on toy batch
7. `training/trainer.py` — small overfit test on 1 batch
8. `inference/retriever.py` — functional test on dummy audio + bias list
9. `eval/metrics.py` — unit test with known predictions
10. Full training run + evaluation

---

## Citation

```bibtex
@inproceedings{kong2025glclap,
  title     = {GLCLAP: A Novel Contrastive Learning Pre-trained Model for Contextual Biasing in ASR},
  author    = {Kong, Yuxiang and Cui, Fan and Guo, Liyong and Dinkel, Heinrich and Fan, Lichun and Zhang, Junbo and Luan, Jian},
  booktitle = {Interspeech},
  year      = {2025},
  doi       = {10.21437/Interspeech.2025-550}
}
```
