# NanoVLM: A High-Performance "Nano" Vision-Language Model

NanoVLM is a robust, PyTorch-based implementation of a Vision-Language Model (VLM) inspired by OpenAI's CLIP architecture. While "Nano" in scale, it utilizes professional-grade techniques including pre-trained backbones, EMA weighting, and gradient accumulation to achieve state-of-the-art alignment quality on a single consumer GPU.

This project demonstrates how to bridge visual and linguistic semantics by aligning representations in a shared 256-dimensional latent space using the Flickr30k dataset.

## 🚀 Key Features

- **Advanced Multimodal Architecture**: 
  - **Image Encoder**: Utilizes a **MobileNetV2** backbone (pretrained on ImageNet) for rich feature extraction, enhanced with a **Multi-Head Self-Attention** block and a deep MLP projection head with GELU activations and Residual connections.
  - **Text Encoder**: A deep **6-layer Transformer** (8 attention heads) capable of capturing complex linguistic context, featuring positional embeddings and specialized projection heads.
- **Superior Training Stability**: 
  - **EMA (Exponential Moving Average)**: Maintains a shadow copy of model weights for smoother, more generalized evaluation.
  - **Gradient Accumulation**: Simulates an effective batch size of **512**, providing stable contrastive gradients without exceeding VRAM limits.
  - **Label Smoothing (0.1)**: Integrated into the symmetric cross-entropy loss to prevent overconfidence and improve zero-shot robustness.
- **Professional Pipeline**: 
  - **Mixed Precision (AMP)**: Uses `torch.amp` for accelerated training.
  - **Scheduling**: Linear Warmup followed by **Cosine Decay** for optimal convergence.
  - **Real-time Monitoring**: Integrated with **TensorBoard** for live tracking of loss, temperature, and learning rates.
- **Robust Evaluation**: Full-epoch validation tracking with batch-local **Recall@1, Recall@5, and Recall@10** metrics.

## 📊 Performance Results (Flickr30k)

After training on the Flickr30k validation set, the model achieved the following retrieval metrics:

| Metric | Result |
| :--- | :--- |
| **Image→Text Recall@1** | **13.9%** |
| **Image→Text Recall@5** | **56.6%** |
| **Image→Text Recall@10** | **73.0%** |

## 📁 Project Structure

```text
NanoVLM/
├── nanoVLM.ipynb         # Main Jupyter Notebook (Upgraded Version)
├── nanoVLM_original.ipynb # Original scratch-built version (for reference)
├── data/                 # Flickr30k dataset directory
├── runs/                 # TensorBoard log directory
├── best_img_enc.pth      # Top-performing Image Encoder weights
├── best_txt_enc.pth      # Top-performing Text Encoder weights
└── best_meta.pth         # Training metadata (epoch, loss, etc.)
```

## 🛠️ Prerequisites & Installation

```bash
pip install torch torchvision numpy matplotlib pillow tensorboard
```

## 🧠 Architecture Details

### The Pipeline
1. **Visual Alignment**: Images undergo **Stochastic Augmentation** (RandomResizedCrop, ColorJitter) before being processed by the MobileNetV2 backbone. A Multi-head attention layer provides global spatial context before the final projection.
2. **Textual Alignment**: Captions are tokenized via a custom `SimpleTokenizer`. The 6-layer Transformer processes the sequences, and the `[BOS]` token embedding is projected into the shared space.
3. **Shared Latent Space**: Both modalities are projected into a **256-dimensional** hypersphere. Alignment is enforced by maximizing the cosine similarity between matching pairs in the batch.

## 🏃 How to Run

1. Open `nanoVLM.ipynb`.
2. Run cells top-to-bottom.
3. To view training progress, run:
   ```bash
   tensorboard --logdir=runs/nanoVLM
   ```

## 📈 Evaluation

The notebook automatically calculates **Recall@K** at the end of training, comparing the **Live** training weights against the **EMA** smoothed weights to ensure the highest possible retrieval accuracy.
