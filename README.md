# Multimodal Movie Genre Classifier (MM-IMDb)

A high-performance Deep Learning pipeline built with **PyTorch** that predicts movie genres by fusing visual aesthetics from posters and semantic themes from plot summaries.

## 🚀 Project Architecture
This project implements a **Late-Fusion Multimodal Network**:
- **Vision Branch:** Pre-trained **ResNet-50** extracts 2048-dimensional spatial features from movie posters.
- **Text Branch:** Pre-trained **DistilBERT** generates 768-dimensional contextual embeddings from plot descriptions.
- **Fusion Layer:** A feature-concatenation layer followed by a Multi-Layer Perceptron (MLP) with Dropout for multi-label classification.

## 📁 Project Structure
```text
movie_genre_classifier/
├── data/               # Local data (Ignored by Git)
├── src/
│   ├── data_loader.py  # MM-IMDb specific dataset parser
│   ├── model.py        # Dual-tower fusion architecture
│   └── train.py        # Training loop with Macro-F1 evaluation
├── .gitignore          # Rules to prevent uploading large datasets
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation