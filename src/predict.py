import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PIL import Image
from torchvision import transforms
from transformers import DistilBertTokenizer
from src.model import MultiModalModel
from src.data_loader import ALL_GENRES

def run_demo(image_path, plot_text):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Model and Weights
    model = MultiModalModel(num_genres=len(ALL_GENRES)).to(DEVICE)
    checkpoint = "checkpoints/model_epoch_10.pt" # Point to your best result
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    model.eval()

    # 2. Preprocess
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Prepare Image
    image = Image.open(image_path).convert('RGB')
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # Prepare Text
    inputs = tokenizer(plot_text, return_tensors="pt", truncation=True, padding=True, max_length=256).to(DEVICE)

    # 3. Predict
    with torch.no_grad():
        outputs = model(img_tensor, inputs['input_ids'], inputs['attention_mask'])
        probs = torch.sigmoid(outputs).cpu().numpy()[0]

    # 4. Show Results
    print("\n--- Model Prediction Results ---")
    for i, genre in enumerate(ALL_GENRES):
        if probs[i] > 0.5: # 50% confidence threshold
            print(f"✅ {genre}: {probs[i]*100:.2f}%")

if __name__ == "__main__":
    img_file = input("Enter path to movie poster image: ")
    plot = input("Enter movie plot summary: ")
    run_demo(img_file, plot)