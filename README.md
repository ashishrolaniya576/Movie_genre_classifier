# 📽️ Multimodal Movie Genre Classifier (MM-IMDb)

> A high-performance deep learning pipeline built with **PyTorch** that predicts movie genres by fusing visual aesthetics from posters and semantic themes from plot summaries.

---

## 📖 Overview

This project implements a **Late-Fusion Multimodal Network** that combines two specialized branches to achieve superior multi-label genre classification across **25 target genres**. By jointly analyzing movie poster visuals and plot text, the model captures complementary signals that neither modality alone can provide.

---

## 🏗️ Architecture

```
┌─────────────────────┐       ┌─────────────────────┐
│    Movie Poster     │       │    Plot Summary     │
│   (Image Input)     │       │   (Text Input)      │
└────────┬────────────┘       └──────────┬──────────┘
         │                               │
         ▼                               ▼
┌─────────────────────┐       ┌─────────────────────┐
│  Vision Branch      │       │   Text Branch       │
│  ResNet-50          │       │   DistilBERT        │
│  → 2048-dim vector  │       │  → 768-dim vector   │
└────────┬────────────┘       └──────────┬──────────┘
         │                               │
         └──────────────┬────────────────┘
                        │ Concatenation
                        ▼
              ┌─────────────────────┐
              │   Fusion Layer      │
              │   2816-dim input    │
              │   MLP + Dropout     │
              └─────────┬───────────┘
                        │
                        ▼
              ┌─────────────────────┐
              │  Multi-Label Output │
              │  (25 genres)        │
              └─────────────────────┘
```

| Branch | Backbone | Output Dimensions |
|---|---|---|
| **Vision** | ResNet-50 (pre-trained) | 2048-d spatial features |
| **Text** | DistilBERT (pre-trained) | 768-d contextual embeddings |
| **Fusion** | Concatenation + MLP | 2816-d → 25 labels |

---

## 📁 Project Structure

```
Movie_genre_classifier/
├── src/
│   ├── data_loader.py      # MM-IMDb dataset parser & augmentations
│   ├── model.py            # Dual-tower fusion architecture
│   ├── train.py            # Training loop with GPU acceleration
│   └── predict.py          # Real-time inference script
├── LLM_Prompts/            # Prompt engineering logs & documentation
├── movie_genre.ipynb       # Interactive analysis & visualization notebook
├── .gitignore              # Excludes datasets and model checkpoints
├── requirements.txt        # Python dependencies
└── README.md
```

---

## 📊 Performance

| Metric | Value |
|---|---|
| **Macro-F1 Score** | `0.8410` |
| **Training Epochs** | 10 |
| **Hardware** | NVIDIA GPU (Lightning AI) |
| **Dataset** | MM-IMDb (visual + textual modalities) |

> Macro-F1 of 0.84 reflects strong performance on imbalanced multi-label classification across 25 genres.

---

## ✨ Key Features

**Multi-Label Classification**
Handles overlapping genres (e.g., *Sci-Fi + Action*) for a single movie using sigmoid outputs with configurable thresholds.

**Class Imbalance Handling**
Uses `BCEWithLogitsLoss` with `pos_weight` to counteract the inherent imbalance across rare and common genre labels.

**Real-Time Inference**
A standalone `predict.py` script lets you test any movie by providing a poster image path and plot text — no training setup required.

**Modular Design**
Clean separation of data loading, model definition, training, and inference modules for maintainability and extensibility.

---

## 🛠️ Installation & Usage

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Movie_genre_classifier.git
cd Movie_genre_classifier
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Inference

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
python3 src/predict.py
```

The inference script will prompt you for a **poster image path** and a **plot summary**, then output the predicted genres with confidence scores.

### 4. Train the Model (Optional)

```bash
python3 src/train.py
```

---

## 🔮 Future Roadmap

- [ ] **Cross-Modal Attention Fusion** — Dynamically weigh text vs. image importance using attention mechanisms
- [ ] **Web Demo** — Wrap inference in a Gradio or Streamlit UI for public access
- [ ] **Extended Metadata** — Integrate Director, Cast, Budget, and Release Year as additional input features
- [ ] **Model Compression** — Distillation or quantization for faster inference

---

## 📦 Dependencies

Key libraries used in this project (see `requirements.txt` for full list):

- `torch` / `torchvision` — Model training and image processing
- `transformers` — DistilBERT text encoder (HuggingFace)
- `Pillow` — Image loading and augmentation
- `scikit-learn` — Evaluation metrics
- `numpy` / `pandas` — Data handling
