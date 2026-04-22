I want you to help me reproduce a paper in a top-down engineering way.

Paper:
"GLCLAP: A Novel Contrastive Learning Pre-trained Model for Contextual Biasing in ASR"

In this round, do NOT implement the full model or training details yet.
I want you to first construct the overall project framework, so that we can fill in the implementation step by step afterward.
Details of this project can be found in the .pdf file attached.

Your task:
1. Design a complete and modular PyTorch project structure for reproducing this paper.
2. For each file, explain:
   - responsibility
   - main classes/functions
   - expected inputs/outputs
   - dependency relations with other files
3. Describe the full system data flow:
   - dataset preparation
   - subtext sampling
   - tokenization / feature extraction
   - text/audio encoding
   - global/local embedding generation
   - contrastive training
   - bias-word retrieval inference
   - ASR prompt integration
4. For key modules, include tensor shape expectations where possible.
5. If the paper is underspecified, explicitly list assumptions or TODO items.
6. Keep the design practical, debuggable, and faithful to the paper.
7. Do not silently invent details.

Constraints:
- Use PyTorch
- Use Hugging Face for pretrained encoders where appropriate
- Do NOT reimplement BERT or Data2Vec from scratch
- Separate dataset, models, losses, training, inference, utils, and configs
- Do NOT write full implementation yet
- You may generate file skeletons, class/function signatures, docstrings, and TODO placeholders

Output format:
1. Project tree
2. File-by-file explanation
3. System data flow
4. Core interfaces and function signatures
5. Recommended implementation order
6. Open questions / assumptions

Be concrete, technical, and engineering-oriented.
Do not give generic paper summary.