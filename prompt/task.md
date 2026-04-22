I want to reproduce the paper:

"GLCLAP: A Novel Contrastive Learning Pre-trained Model for Contextual Biasing in ASR"

I have already read the paper and extracted the following key components:

1. Model:
- Text encoder: BERT (bert-base-multilingual-uncased)
- Audio encoder: Data2VecAudioModel (or equivalent)
- Dual-branch text:
    - Global: full text
    - Local: subtext (randomly sampled)
- Audio:
    - Local embedding: [B, T', D]
    - Global embedding: mean pooling → [B, D]

2. Loss:
- Global contrastive loss (audio_global vs text_global)
- Local max-pooling contrastive loss (text_local vs audio_local)
- Final loss: L = Lg + Ll

3. Inference:
- Encode bias word list → [K, D]
- Encode audio (no pooling) → [T, D]
- Compute similarity → [K, T]
- Max over time → [K]
- Select words above threshold

Your job is to help me implement this system step by step.