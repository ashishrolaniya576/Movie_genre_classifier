Multimodal Movie Genre Classifier

An AI system that predicts movie genres by fusing visual features from posters and semantic embeddings from plot summaries.
🚀 Project Overview

This project implements a Late Fusion Multimodal Architecture. By combining a ResNet-50 (Computer Vision) and DistilBERT (Natural Language Processing), the model achieves a more nuanced understanding of movie genres than a single-modality system could.
Performance Results

    Macro-F1 Score: 0.8410

    Training Time: 10 Epochs on NVIDIA GPU (Lightning AI)

    Dataset: MM-IMDb (Posters + Plot Metadata)

🧠 Model Architecture

The system processes two distinct data streams before fusing them for final classification:

    Vision Tower: ResNet-50 (Pre-trained on ImageNet) extracts spatial features from movie posters.

    Text Tower: DistilBERT extracts contextual embeddings from movie plot summaries.

    Fusion Head: A Multi-Layer Perceptron (MLP) that concatenates both feature vectors (2048+768=2816 dimensions) to predict across 25 genre categories.

📂 Project Structure

    src/data_loader.py: Handles JSON parsing, image preprocessing, and PyTorch Dataset/DataLoader logic.

    src/model.py: Defines the MultiModalModel architecture and the feature fusion layers.

    src/train.py: Contains the training loop, loss functions, and checkpoint saving logic.

    src/predict.py: A standalone tool for real-time inference on new, unseen movie data.

    movie_genre.ipynb: Interactive notebook for data visualization, testing, and result analysis.

    checkpoints/: Directory containing trained model weights (.pt files) for each epoch.

✨ Key Features & Usage

    Multi-Label Classification: Advanced logic to identify multiple overlapping genres for a single movie entry.

    Optimized Learning: Utilizes BCEWithLogitsLoss to effectively manage class imbalance across the 25 target genres.

    Automated Setup: Environment can be fully initialized using the command: pip install -r requirements.txt.

    Real-Time Inference: Supports external testing via terminal using: export PYTHONPATH=$PYTHONPATH:$(pwd) && python3 src/predict.py.

    Checkpointing System: Automatically saves model states, allowing for training resumption and instant demo loading.

💡 Future Improvements

    Attention Fusion: Implementing Cross-modal attention to weigh text vs. image importance dynamically.

    Web Deployment: Wrapping the inference script in a Gradio or Streamlit interface for non-technical users.

    Feature Expansion: Integrating additional metadata such as Director, Cast, and Budget to refine predictions.