📽️ Multimodal Movie Genre Classifier

An AI system that predicts movie genres by fusing visual features from posters and semantic embeddings from plot summaries.
🚀 Overview

This project implements a Late Fusion Multimodal Architecture. By combining a ResNet-50 (Computer Vision) and DistilBERT (Natural Language Processing), the model achieves a more nuanced understanding of movie genres than a single-modality system could.
📊 Performance Results

    Macro-F1 Score: 0.8410

    Exact Match Ratio: (Insert your subset accuracy % here)

    Training Time: 10 Epochs on NVIDIA GPU (Lightning AI)

🧠 Architecture

    Vision Tower: ResNet-50 (Pre-trained on ImageNet) to extract spatial features from movie posters.

    Text Tower: DistilBERT to extract contextual embeddings from plot summaries.

    Fusion Head: A multi-layer perceptron (MLP) that concatenates both feature vectors (2816 dimensions) to predict 25 possible genres.

📂 Project Structure
Bash

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
Bash

pip install -r requirements.txt

2. Real-Time Inference

Test the model with your own images and plots:
Bash

export PYTHONPATH=$PYTHONPATH:$(pwd)
python3 src/predict.py

🌟 Key Features

    Multi-Label Classification: Handles movies with multiple genres (e.g., Action + Adventure + Sci-Fi).

    Weighted Loss: Uses BCEWithLogitsLoss to handle label imbalance.

    Interactive Demo: Includes a script for testing "unseen" movies from the web.

💡 Future Improvements

    Implement Attention-based Fusion (Cross-modal attention).

    Deploy as a web application using Gradio or Streamlit.

    Integrate metadata like "Director" or "Cast" as additional features.