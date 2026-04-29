# NanoVLM Architecture

## Overview

NanoVLM is a CLIP-style dual-encoder vision-language model that maps images and text into a shared 256-dimensional embedding space. The system is organized into:

- `model.py` for core model classes and checkpoint loading
- `nanoVLM.ipynb` for data pipeline, training, checkpointing, and evaluation
- `app.py` for Gradio-based inference demo

The current architecture uses spatial attention in the image encoder and a Transformer encoder for text.

## Core Model (`model.py`)

### Shared Hyperparameters

- Embedding dimension: `256`
- Attention heads: `4`
- Text Transformer layers: `6`
- Dropout: `0.1`
- Context length: `64`
- Vocab size target: `10000`

### Tokenizer: `SimpleTokenizer`

- Word-level regex tokenizer with special tokens:
  - `<pad>`, `<unk>`, `<bos>`, `<eos>`
- Vocabulary is built from training captions using min frequency threshold.
- Inference restores tokenizer with `load_state()` from `tokenizer_vocab.pth`.

### Image Encoder: `ImageEncoder`

Pipeline:

1. MobileNetV2 feature backbone (ImageNet pretrained)
2. Freeze first 14 backbone blocks (partial fine-tuning)
3. Convert feature map to spatial token sequence (`H*W`, typically `7x7=49` tokens at 224 input)
4. Linear projection from 1280 channels to 256 embedding dim (per token)
5. Pre-norm multi-head self-attention over spatial tokens with residual
6. Pre-norm MLP block over tokens with residual
7. Mean pool over spatial tokens
8. Final LayerNorm + L2 normalization

Output: unit-normalized image embedding of shape `[B, 256]`.

### Text Encoder: `TextEncoder`

Pipeline:

1. Token embedding + positional embedding
2. 6-layer Transformer encoder (`batch_first=True`)
3. Attention-mask-aware mean pooling over valid tokens
4. Residual MLP projection head
5. LayerNorm + L2 normalization

Output: unit-normalized text embedding of shape `[B, 256]`.

### Contrastive Objective: `CLIPLoss`

- Learnable temperature parameter (`logit_scale`)
- Symmetric image->text and text->image cross-entropy
- Optional label smoothing (`0.1`)
- Similarity matrix from dot products of normalized embeddings

## Training Architecture (`nanoVLM.ipynb`)

### Data Pipeline

- Loads paired image-caption data (current notebook setup targets MS-COCO train/val paths).
- Custom paired dataset returns `(image_tensor, token_ids, raw_text)`.
- Custom collate function:
  - Pads images to per-batch max HxW
  - Pads token sequences
  - Builds attention masks

### Augmentation and Preprocessing

- Train transform includes resize, random crop, flip, color jitter, grayscale, normalize
- Validation transform includes deterministic resize/center crop and normalize

### Optimization Stack

- Optimizer: `AdamW`
- LR schedule: linear warmup then cosine decay (`LambdaLR`)
- Mixed precision: `torch.amp` + `GradScaler`
- Gradient accumulation (`GRAD_ACCUM_STEPS=2`) for larger effective batch
- Gradient clipping (`max_norm=1.0`)
- Logit scale clamp for stability
- EMA shadow models for smoother evaluation
- Early stopping on validation loss

### Checkpointing

Best checkpoints saved for:

- image encoder
- text encoder
- criterion
- optimizer
- scheduler
- metadata (`epoch`, `best_val_loss`)

Resume logic restores these states when present.

## Inference Architecture (`app.py`)

### Runtime Flow

1. Load trained encoders via `load_models()`
2. Load tokenizer vocabulary from `tokenizer_vocab.pth`
3. Preprocess uploaded image with validation-like transform
4. Encode image once
5. Encode each candidate caption
6. Compute cosine-style similarity via embedding dot product
7. Rank captions and visualize scores with matplotlib

### UI

- Gradio Blocks interface
- Input: 1 image + up to 5 caption candidates
- Output:
  - horizontal score chart
  - best-match summary text

The chart now uses dynamic x-limits instead of fixed `[0, 1]`.

## Current Evaluation State

- Notebook currently includes batch-local Recall@K evaluation across validation batches.
- This is useful for monitoring but not the final retrieval benchmark.
- A global Recall@K pass (full validation embedding matrix) should be run after training for final reporting.

## Design Strengths

- Clean dual-encoder architecture for scalable retrieval
- Spatial attention now operates over actual image regions
- Stable training recipe (AMP, warmup+cosine, EMA, grad accumulation, clipping)
- Clear separation between model definition, training notebook, and demo app

## Current Limitations

- Tokenizer is still word-level (not subword/BPE)
- Evaluation headline metric is currently batch-local Recall@K in notebook
- Some project documentation may need final consistency pass once training metrics are finalized
