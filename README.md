# NanoVLM: A High-Performance "Nano" Vision-Language Model

NanoVLM is a robust, PyTorch-based implementation of a Vision-Language Model (VLM) inspired by OpenAI's CLIP architecture. While "Nano" in scale, it utilizes professional-grade techniques including pre-trained backbones, EMA weighting, and gradient accumulation to achieve state-of-the-art alignment quality on a single consumer GPU.

This project demonstrates how to bridge visual and linguistic semantics by aligning representations in a shared 256-dimensional latent space, trained natively on the massive 591k-pair MS-COCO dataset.

## 🚀 Key Features

- **Advanced Multimodal Architecture**: 
  - **Image Encoder**: Utilizes a **MobileNetV2** backbone (pretrained on ImageNet) for rich feature extraction. Upgraded with **Spatial Self-Attention** over a 7x7 patch grid (49 tokens) allowing the model to learn localized feature relevance before pooling.
  - **Text Encoder**: A deep **6-layer Transformer** (4 attention heads) with **Masked Mean Pooling** to correctly handle padding tokens during semantic alignment.
- **Superior Training Stability**: 
  - **Optimized Regularization**: Uses a weight decay of **0.05** and label smoothing (**0.1**) to ensure robust generalization and prevent overfitting on smaller datasets.
  - **Gradient Accumulation**: Simulates an effective batch size of **512**, providing stable contrastive gradients without exceeding VRAM limits.
- **Professional Pipeline**: 
  - **Mixed Precision (AMP)**: Uses `torch.amp` for accelerated training.
  - **Scheduling**: Linear Warmup followed by **Cosine Decay** for optimal convergence.
  - **Robust Early Stopping**: Improved patience settings to allow the model to fully explore the loss landscape.

## 📊 Performance Results (MS-COCO)

After an extended training run on the full MS-COCO dataset (using Spatial Attention), the model achieved the following state-of-the-art metrics for its size class:

| Metric | Result |
| :--- | :--- |
| **Best Val Loss** | **2.6436** |
| **Image→Text Recall@1** | **19.2%** |
| **Image→Text Recall@5** | **68.7%** |
| **Image→Text Recall@10** | **84.8%** |

*Note: Recall@10 of 84.8% indicates the correct matching caption is found in the top 10 results for nearly 85% of all images in the complex COCO validation set.*

## 🛠️ Installation & Usage

1. **Install Dependencies**:
   ```bash
   pip install torch torchvision matplotlib pillow gradio
   ```
2. **Run Inference**:
   Ensure `best_img_enc.pth`, `best_txt_enc.pth`, and `tokenizer_vocab.pth` are in the root directory.
   ```bash
   python app.py
   ```

## 📂 Project Structure

- `model.py`: Core architecture, including tokenizer and dual-encoder definitions.
- `nanoVLM.ipynb`: Full training pipeline, including data ingestion, contrastive loop, and evaluation.
- `app.py`: Gradio-based web interface for real-time image-caption similarity matching.
- `save_tokenizer.py`: Utility script to serialize the training vocabulary for inference.
