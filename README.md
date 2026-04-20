Multimodal Movie Genre Classifier

An AI system that predicts movie genres by fusing visual features from posters and semantic embeddings from plot summaries.
🚀 Project Overview

This project implements a Late Fusion Multimodal Architecture. By combining a ResNet-50 (Computer Vision) and DistilBERT (Natural Language Processing), the model achieves a more nuanced understanding of movie genres than a single-modality system could.
Performance Results

    Macro-F1 Score: 0.8410

    Exact Match Ratio: [Your %]

    Training Time: 10 Epochs on NVIDIA GPU (Lightning AI)

🧠 Architecture

The model processes two distinct data streams before fusing them for final classification:

    Vision Tower: ResNet-50 (Pre-trained on ImageNet) extracts spatial features from movie posters.

    Text Tower: DistilBERT extracts contextual embeddings from movie plot summaries.

    Fusion Head: A Multi-Layer Perceptron (MLP) that concatenates both feature vectors (2048+768=2816 dimensions) to predict 25 possible genres.

📂 Project Structure
Plaintext

.
├── src/
│   ├── data_loader.py   # JSON parsing & PyTorch Dataset logic
│   ├── model.py        # MultiModalModel architecture
│   ├── train.py        # Training & Checkpointing script
│   └── predict.py      # Real-time inference script
├── movie_genre.ipynb   # Interactive analysis and visualization
├── checkpoints/        # Saved model weights (.pt files)
└── requirements.txt    # Dependencies

🛠️ How to Run
1. Setup Environment

Install all required libraries using pip:
Bash

pip install -r requirements.txt

2. Real-Time Inference

Test the model with your own images and plot descriptions:
Bash

export PYTHONPATH=$PYTHONPATH:$(pwd)
python3 src/predict.py

✨ Key Features

    Multi-Label Classification: Handles movies belonging to multiple genres simultaneously.

    Weighted Loss: Utilizes BCEWithLogitsLoss to account for class imbalance across 25 genres.

    Inference Tool: Includes a dedicated script for testing "unseen" data from external sources.

💡 Future Improvements

    Implement Attention-based Fusion (Cross-modal attention).

    Deploy a web interface using Gradio or Streamlit.

    Incorporate additional metadata such as Director or Cast as model features.