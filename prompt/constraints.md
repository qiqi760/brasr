Constraints:

- Use PyTorch
- Use Hugging Face for pretrained encoders (BERT, Data2Vec)
- Do NOT reimplement encoders from scratch
- Keep modules separate:
    - dataset
    - model
    - loss
    - training loop
    - inference

- Always specify tensor shapes in comments
- Always explain why a layer is added (e.g., projection)
- If dimensions mismatch, explicitly add projection layers

- Do NOT implement everything at once
- Only implement ONE step at a time