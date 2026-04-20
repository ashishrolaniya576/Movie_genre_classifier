import os
import json
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import DistilBertTokenizer
from PIL import Image

ALL_GENRES = [
    "Action", "Adventure", "Animation", "Biography", "Comedy",
    "Crime", "Documentary", "Drama", "Family", "Fantasy",
    "Film-Noir", "History", "Horror", "Music", "Musical",
    "Mystery", "News", "Reality-TV", "Romance", "Sci-Fi", "Short",
    "Sport", "Thriller", "War", "Western",
]
GENRE_TO_IDX = {g: i for i, g in enumerate(ALL_GENRES)}
NUM_GENRES = len(ALL_GENRES)

def parse_json_to_dataframe(json_folder, image_folder):
    records = []
    # Check for all common image extensions
    valid_exts = [".jpg", ".jpeg", ".png", ".webp"]
    
    for fname in os.listdir(json_folder):
        if fname.endswith(".json"):
            movie_id = fname.replace(".json", "")
            json_path = os.path.join(json_folder, fname)
            
            # Find the image with any valid extension
            img_path = None
            for ext in valid_exts:
                test_path = os.path.join(image_folder, f"{movie_id}{ext}")
                if os.path.isfile(test_path):
                    img_path = test_path
                    break
            
            if not img_path: continue

            with open(json_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            
            plot = meta.get("plot", "")
            if isinstance(plot, list): plot = " ".join(plot)
            
            genres = meta.get("genres", [])
            if not genres: continue

            records.append({
                "movie_id": movie_id,
                "plot": str(plot).strip(),
                "genres": genres,
                "image_path": img_path
            })
    return pd.DataFrame(records)

class MovieGenreDataset(Dataset):
    def __init__(self, dataframe, tokenizer, transform=None, max_length=256):
        self.df = dataframe
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_length = max_length

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        if self.transform: image = self.transform(image)

        encoding = self.tokenizer(
            row["plot"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return image, encoding["input_ids"].squeeze(0), encoding["attention_mask"].squeeze(0), \
               torch.tensor([1.0 if g in row["genres"] else 0.0 for g in ALL_GENRES])