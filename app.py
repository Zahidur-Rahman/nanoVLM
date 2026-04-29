"""
NanoVLM - Gradio Demo
=====================
Inference interface for image-caption matching.
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import gradio as gr
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io

from model import SimpleTokenizer, load_models, CONTEXT_LEN

# ── Configuration ─────────────────────────────────────────────────────────────
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_ENC_PATH = "best_img_enc.pth"
TXT_ENC_PATH = "best_txt_enc.pth"
VOCAB_PATH   = "tokenizer_vocab.pth"   # saved tokenizer vocabulary

# ── Load models once at startup ───────────────────────────────────────────────
print(f"Loading NanoVLM models on {DEVICE}...")
img_enc, txt_enc = load_models(IMG_ENC_PATH, TXT_ENC_PATH, device=DEVICE)

tokenizer = SimpleTokenizer()
vocab_state = torch.load(VOCAB_PATH, map_location="cpu")
tokenizer.load_state(vocab_state)
print(f"Tokenizer vocab size: {len(tokenizer.itos)}")

# ── Image pre-processing (same as validation pipeline) ────────────────────────
_NORM_MEAN = [0.48145466, 0.4578275, 0.40821073]   # CLIP ImageNet stats
_NORM_STD  = [0.26862954, 0.26130258, 0.27577711]

img_transform = T.Compose([
    T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=_NORM_MEAN, std=_NORM_STD),
])


# ── Core inference function ───────────────────────────────────────────────────
def score_image_vs_captions(image: Image.Image, captions: list[str]) -> dict:
    """
    Given a PIL image and a list of text captions, returns a dict of
    {caption: similarity_score} suitable for gr.Label.
    """
    # --- Image embedding ---
    img_tensor = img_transform(image).unsqueeze(0).to(DEVICE)   # [1, 3, 224, 224]
    with torch.no_grad():
        img_emb = img_enc(img_tensor)   # [1, 256]

    # --- Text embeddings ---
    scores = {}
    for caption in captions:
        if not caption.strip():
            continue
        toks = tokenizer.encode(caption, max_len=CONTEXT_LEN).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            txt_emb = txt_enc(toks)   # [1, 256]
        score = (img_emb @ txt_emb.T).item()   # cosine similarity (both L2-normalised)
        scores[caption.strip()] = round(float(score), 4)

    return scores


# ── Gradio handler ─────────────────────────────────────────────────────────────
def run_demo(image, caption1, caption2, caption3, caption4, caption5):
    """Main Gradio handler: takes image + 5 captions, returns ranked label scores."""
    if image is None:
        return None, "⚠️ Please upload an image first."

    captions = [c for c in [caption1, caption2, caption3, caption4, caption5] if c.strip()]
    if not captions:
        return None, "⚠️ Please enter at least one caption."

    scores = score_image_vs_captions(image, captions)

    # Sort by score descending
    sorted_scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

    # Build a simple matplotlib bar chart
    fig, ax = plt.subplots(figsize=(7, max(2, len(sorted_scores) * 0.8)))
    labels  = list(sorted_scores.keys())
    values  = list(sorted_scores.values())
    colors  = ["#4CAF50" if i == 0 else "#90CAF9" for i in range(len(labels))]

    bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1], edgecolor="white")
    ax.set_xlabel("Cosine Similarity Score", fontsize=11)
    ax.set_title("NanoVLM: Image-Caption Matching", fontsize=13, fontweight="bold")
    lo = min(min(values) - 0.05, 0)
    hi = max(max(values) + 0.08, 0.3)
    ax.set_xlim(lo, hi)

    for bar, val in zip(bars, values[::-1]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=10)

    fig.tight_layout()

    # Convert to PIL image for Gradio
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    chart_img = Image.open(buf).copy()
    buf.close()

    best = max(sorted_scores, key=sorted_scores.get)
    summary = f"✅ Best match: **\"{best}\"** (score: {sorted_scores[best]:.3f})"
    return chart_img, summary


# ── Build Gradio UI ───────────────────────────────────────────────────────────
with gr.Blocks(theme=gr.themes.Soft(), title="NanoVLM Demo") as demo:
    gr.Markdown(
        """
        # 🔭 NanoVLM — Vision-Language Matching Demo
        **NanoVLM** is a CLIP-style contrastive model trained on Flickr30k.
        It aligns images and text in a shared 256-dimensional embedding space.

        **How to use:**
        1. Upload any image on the left.
        2. Enter 2–5 caption candidates.
        3. Click **Match!** — the model scores how well each caption fits the image.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="📷 Upload Image")
            caption1 = gr.Textbox(label="Caption 1", placeholder="A dog running on the beach")
            caption2 = gr.Textbox(label="Caption 2", placeholder="A cat sitting on a chair")
            caption3 = gr.Textbox(label="Caption 3 (optional)", placeholder="")
            caption4 = gr.Textbox(label="Caption 4 (optional)", placeholder="")
            caption5 = gr.Textbox(label="Caption 5 (optional)", placeholder="")
            run_btn  = gr.Button("⚡ Match!", variant="primary")

        with gr.Column(scale=1):
            chart_output   = gr.Image(label="📊 Similarity Scores")
            summary_output = gr.Markdown(label="Result")

    run_btn.click(
        fn=run_demo,
        inputs=[image_input, caption1, caption2, caption3, caption4, caption5],
        outputs=[chart_output, summary_output],
    )

    gr.Markdown(
        """
        ---
        **Architecture:** MobileNetV2 (spatial attention) + 6-layer Transformer | 
        **Training:** MS-COCO, EMA + Gradient Accumulation |
        **Embedding:** 256-dim shared latent space
        """
    )

if __name__ == "__main__":
    demo.launch()
