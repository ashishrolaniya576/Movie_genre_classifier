#  Multimodal Movie Genre Classifier (MM-IMDb)

A **high-performance Deep Learning pipeline** built with **PyTorch** that predicts movie genres by combining:

-  **Visual aesthetics** from movie posters  
-  **Semantic meaning** from plot summaries  

---

##  Project Architecture

This project implements a **Late-Fusion Multimodal Neural Network**:

### 🔹 Vision Branch
- Uses **ResNet-50 (Pre-trained on ImageNet)**
- Extracts **2048-dimensional spatial features** from posters

### 🔹 Text Branch
- Uses **DistilBERT (Transformer-based model)**
- Generates **768-dimensional contextual embeddings** from plot descriptions

### 🔹 Fusion Layer
- Concatenates features (**2048 + 768 = 2816 dimensions**)
- Passes through a **Multi-Layer Perceptron (MLP)**
- Uses **Dropout** for regularization
- Outputs **multi-label genre predictions**

---

##  Project Structure

```text
movie_genre_classifier/
├── data/                    # Local dataset (ignored by Git)
├── src/
│   ├── data_loader.py      # Dataset parsing (MM-IMDb)
│   ├── model.py            # Multimodal architecture (ResNet + BERT)
│   ├── train.py            # Training pipeline with evaluation
│   ├── predict.py          # Inference script for predictions
├── movie_genre.ipynb       # Notebook for experimentation & testing
├── .gitignore              # Ignore large files (dataset, checkpoints)
├── requirements.txt        # Dependencies
└── README.md               # Project documentation

---

##  Performance Results

-  **Macro-F1 Score:** **0.8410**
-  **Training Setup:** NVIDIA GPU (Lightning AI)
-  **Epochs:** 10
-  **Dataset:** MM-IMDb *(Posters + Plot Metadata)*


##  Key Features

-  **Multimodal Learning** → Combines image + text
-  **Multi-Label Classification** → Multiple genres per movie
-  **Class Imbalance Handling** → BCEWithLogitsLoss
-  **Scalable Architecture** → Easy to extend

---

##  Installation
    pip install -r requirements.txt


### 🔹 Run Inference
    python src/predict.py


### 🔹 Experiment (Notebook)
    movie_genre.ipynb


---

##  Future Improvements

-  **Cross-Modal Attention**
  - Learn importance of text vs image dynamically

- 🌐 **Web Deployment**
  - Build UI using **Gradio / Streamlit**

-  **Feature Expansion**
  - Add **Director, Cast, Budget**

-  **Advanced Models**
  - Use **BERT / RoBERTa**
  - Try **Vision Transformers (ViT)**

---

##  Tech Stack

-  Python  
-  PyTorch  
-  HuggingFace Transformers  
-  Torchvision  
-  Scikit-learn  

---