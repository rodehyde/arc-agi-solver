---
name: ML architecture (current)
description: Transformer with in-context learning — one model type for all clusters, merge by retraining on union
type: project
originSessionId: fcc9c5ec-164b-406c-9e31-7198f453b7f9
---
## Current approach: decoder-only transformer with in-context learning

**Key design principle:** Same model architecture for every cluster, different trained parameters. Clusters can be merged by retraining on the combined task set. No hardcoded per-cluster solvers.

### Model
- Decoder-only transformer, 5.3M params
- d_model=256, 8 heads, 6 layers, ffn_dim=1024
- 18-token vocabulary (colours 0-9 + 8 special tokens: START, START_IN, END_IN, START_OUT, END_OUT, END, ROW_SEP, PAD)
- 5-feature per-token encoding: [colour, col, row, colour_change, grid_number]
- Embeddings: token + sinusoidal row/col + learned grid-number segment + learned colour-change + learned absolute position; learnable scale balances them (Hodel 2024 approach)
- Files: `src/transformer_model.py`, `src/arc_tokenizer.py`

### Training
- In-context: K=3 training pairs as prefix, predict test output token-by-token
- Loss on test output tokens only
- Training data: RE-ARC (1000 synthetic examples per task) — `data/re_arc/`
- Token-budget batching: pack sequences greedily up to max_tokens per batch
- Augmentation: D4 (4 rotations × 2 flips) + colour permutations
- LR schedule: LinearLR warmup → CosineAnnealingLR (SequentialLR)
- Early stopping on validation loss with patience
- Script: `scripts/train_transformer.py`
- Colab notebook: `notebooks/train_colab.ipynb` (A100, ~6s/epoch)

### Cluster strategy
1. Identify coherent cluster (tasks share the same type of rule)
2. Train transformer on that cluster's RE-ARC data
3. Evaluate exact match on original ARC training tasks
4. Merge clusters: pass `--clusters 9 12` to train on union; same model handles both

### Results so far
- Cluster 18 (n=20, mixed rules): 65% cell accuracy, 0 exact match — model can't determine which rule to apply
- Cluster 9 (n=24, mostly geometric transforms): **not yet trained** — expected to perform much better

### Earlier approach (archived)
A more complex 8-phase pipeline was discussed (CNN grid encoder + CLIP-style alignment loss + TTT). This was set aside in favour of the simpler transformer approach to get results faster. May revisit if transformer ceiling is too low.
