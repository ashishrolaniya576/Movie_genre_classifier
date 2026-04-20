import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_loader import parse_json_to_dataframe, MovieGenreDataset, ALL_GENRES
from model import MultiModalModel
from torchvision import transforms
from transformers import DistilBertTokenizer
from sklearn.model_selection import train_test_split

# Config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 10
BATCH_SIZE = 16

def train():
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    df = parse_json_to_dataframe("imdb/json", "imdb/dataset")
    train_df, val_df = train_test_split(df, test_size=0.15)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_loader = DataLoader(MovieGenreDataset(train_df, tokenizer, transform), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(MovieGenreDataset(val_df, tokenizer, transform), batch_size=BATCH_SIZE)

    model = MultiModalModel(num_genres=len(ALL_GENRES)).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    for epoch in range(EPOCHS):
        model.train()
        for images, ids, mask, labels in train_loader:
            images, ids, mask, labels = images.to(DEVICE), ids.to(DEVICE), mask.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images, ids, mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} Complete. Loss: {loss.item():.4f}")
        torch.save(model.state_dict(), f"checkpoints/model_epoch_{epoch+1}.pt")

if __name__ == "__main__":
    train()