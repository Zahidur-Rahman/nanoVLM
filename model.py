"""
NanoVLM - Architecture & Inference
==================================
Vision-Language alignment model with MobileNetV2 vision backbone 
and Transformer text encoder.
"""

import re
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from typing import List
from torchvision import models as tv_models

# ── Hyperparameters (must match training) ─────────────────────────────────────
EMBD_DIM      = 256
ATTEN_HEADS   = 4
NUM_TX_LAYERS = 6
DROPOUT       = 0.1
VOCAB_SIZE    = 10000
CONTEXT_LEN   = 64


# ── Tokenizer ─────────────────────────────────────────────────────────────────
class SimpleTokenizer:
    """
    Lightweight word-level tokenizer. Must be re-built with the same training
    captions to reproduce the same vocabulary as during training.
    For inference, load the saved tokenizer state via `load_state`.
    """
    def __init__(self, min_freq=2, max_vocab_size=10000):
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size
        self.pad_token, self.unk_token = "<pad>", "<unk>"
        self.bos_token, self.eos_token = "<bos>", "<eos>"
        self.specials = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        self.stoi = {tok: i for i, tok in enumerate(self.specials)}
        self.itos = list(self.specials)

    def _tokenize(self, text: str):
        return re.findall(r"\w+|[^\w\s]", text.lower())

    @property
    def pad_id(self): return self.stoi[self.pad_token]
    @property
    def unk_id(self): return self.stoi[self.unk_token]
    @property
    def bos_id(self): return self.stoi[self.bos_token]
    @property
    def eos_id(self): return self.stoi[self.eos_token]

    def build_vocab(self, texts: List[str]):
        counter = Counter()
        for t in texts:
            counter.update(self._tokenize(t))
        max_new = self.max_vocab_size - len(self.specials)
        sorted_tokens = sorted(
            [(tok, freq) for tok, freq in counter.items() if freq >= self.min_freq],
            key=lambda x: (-x[1], x[0])
        )[:max_new]
        for tok, _ in sorted_tokens:
            if tok not in self.stoi:
                self.stoi[tok] = len(self.itos)
                self.itos.append(tok)

    def encode(self, text: str, max_len=64):
        tokens = self._tokenize(text)
        ids = [self.bos_id] + [self.stoi.get(t, self.unk_id) for t in tokens] + [self.eos_id]
        return torch.tensor(ids[:max_len], dtype=torch.long)

    def decode(self, ids: torch.Tensor):
        if ids.ndim > 1:
            return [self.decode(seq) for seq in ids]
        tokens = [self.itos[i] for i in ids.tolist()
                  if i not in [self.pad_id, self.bos_id, self.eos_id]]
        return " ".join(tokens)

    def get_state(self):
        """Returns tokenizer vocabulary for saving."""
        return {"stoi": self.stoi, "itos": self.itos}

    def load_state(self, state: dict):
        """Restores tokenizer vocabulary from a saved state dict."""
        self.stoi = state["stoi"]
        self.itos = state["itos"]


# ── Image Encoder ─────────────────────────────────────────────────────────────
class ImageEncoder(nn.Module):
    """
    Vision encoder: MobileNetV2 backbone → spatial token projection →
    self-attention over image regions → mean pooling → MLP head.

    MobileNetV2 outputs a 7×7 spatial grid (for 224×224 input), giving 49
    spatial tokens.  Self-attention lets the model learn which regions of the
    image carry semantic weight for a given caption (e.g. focus on the dog,
    not the sky).  This replaces the previous design that pooled *before*
    attention, which collapsed spatial info and made the attention a no-op.
    """
    def __init__(self, embd_dim=EMBD_DIM, dropout=DROPOUT):
        super().__init__()

        # Pretrained backbone with partial freezing
        backbone = tv_models.mobilenet_v2(weights=tv_models.MobileNet_V2_Weights.DEFAULT).features

        # Freeze first 14 blocks (low-level texture/edge detectors)
        for i, child in enumerate(backbone.children()):
            if i < 14:
                for p in child.parameters():
                    p.requires_grad = False

        self.backbone   = backbone
        self.dropout    = nn.Dropout(p=dropout)
        self.projection = nn.Linear(1280, embd_dim)      # per-token projection

        # Pre-norm self-attention over spatial tokens (Transformer block pattern)
        self.attn_norm = nn.LayerNorm(embd_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embd_dim, num_heads=ATTEN_HEADS, batch_first=True, dropout=dropout
        )

        # Pre-norm MLP applied per spatial token before pooling
        self.mlp_norm = nn.LayerNorm(embd_dim)
        self.mlp_head = nn.Sequential(
            nn.Linear(embd_dim, embd_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embd_dim * 2, embd_dim),
        )

        self.final_norm = nn.LayerNorm(embd_dim)

    def forward(self, x):
        x = self.backbone(x)                              # [B, 1280, H', W']
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)                  # [B, H'*W', 1280]  (49 spatial tokens)
        x = self.dropout(self.projection(x))               # [B, 49, embd_dim]

        # Self-attention with pre-norm + residual
        x_norm = self.attn_norm(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out                                  # [B, 49, embd_dim]

        # Per-token MLP with pre-norm + residual
        x = x + self.mlp_head(self.mlp_norm(x))           # [B, 49, embd_dim]

        # Global average pool over spatial tokens
        x = x.mean(dim=1)                                 # [B, embd_dim]

        return F.normalize(self.final_norm(x), dim=-1)     # unit-norm embedding


# ── Text Encoder ──────────────────────────────────────────────────────────────
class TextEncoder(nn.Module):
    """
    Textual encoder using a multi-layer Transformer with mean-pooling 
    and residual MLP projection.
    """
    def __init__(self, embd_dim=EMBD_DIM, num_heads=ATTEN_HEADS,
                 vocab_size=VOCAB_SIZE, context_window=CONTEXT_LEN,
                 num_layers=NUM_TX_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.token_embedding    = nn.Embedding(vocab_size, embd_dim)
        self.position_embedding = nn.Embedding(context_window, embd_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embd_dim, nhead=num_heads,
            dim_feedforward=embd_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embd_dim)

        self.mlp_head = nn.Sequential(
            nn.Linear(embd_dim, embd_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embd_dim * 2, embd_dim),
        )

    def forward(self, toks, attention_mask=None):
        N, L = toks.shape
        pos = torch.arange(L, device=toks.device).unsqueeze(0)
        x   = self.token_embedding(toks) + self.position_embedding(pos)

        pad_mask = (attention_mask == 0) if attention_mask is not None else None
        x = self.transformer(x, src_key_padding_mask=pad_mask)   # [B, L, embd_dim]

        # Masked mean pooling — exclude padding positions
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()   # [B, L, 1]
            x = (x * mask).sum(1) / mask.sum(1).clamp(min=1e-9)  # [B, embd_dim]
        else:
            x = x.mean(dim=1)                             # [B, embd_dim]
        x = x + self.mlp_head(x)  # MLP head with residual

        return F.normalize(self.norm(x), dim=-1)   # unit-norm embedding


# ── CLIP Loss ─────────────────────────────────────────────────────────────────
class CLIPLoss(nn.Module):
    """
    Symmetric cross-entropy over the image-text similarity matrix.
    """
    def __init__(self, label_smoothing=0.1):
        super().__init__()
        self.logit_scale     = nn.Parameter(torch.ones([]) * (1 / 0.07))
        self.label_smoothing = label_smoothing

    def forward(self, image_emb, txt_emb):
        scale   = self.logit_scale.exp().clamp(max=100)
        logits  = image_emb @ txt_emb.T * scale
        targets = torch.arange(image_emb.size(0), device=image_emb.device)
        loss_i = F.cross_entropy(logits,   targets, label_smoothing=self.label_smoothing)
        loss_t = F.cross_entropy(logits.T, targets, label_smoothing=self.label_smoothing)
        return (loss_i + loss_t) / 2.0


# ── Convenience: load trained models ─────────────────────────────────────────
def load_models(img_enc_path="best_img_enc.pth",
                txt_enc_path="best_txt_enc.pth",
                vocab_size=VOCAB_SIZE,
                device=None):
    """
    Load trained ImageEncoder and TextEncoder from saved checkpoint files.

    Args:
        img_enc_path: Path to best_img_enc.pth
        txt_enc_path: Path to best_txt_enc.pth
        vocab_size:   Must match the vocabulary size used during training (default 10000)
        device:       torch.device — defaults to CUDA if available, else CPU

    Returns:
        img_enc, txt_enc  (both in eval mode, on device)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_enc = ImageEncoder().to(device)
    txt_enc = TextEncoder(vocab_size=vocab_size).to(device)

    img_enc.load_state_dict(torch.load(img_enc_path, map_location=device))
    txt_enc.load_state_dict(torch.load(txt_enc_path, map_location=device))

    img_enc.eval()
    txt_enc.eval()
    return img_enc, txt_enc
