import torch
import torch.nn as nn
from torchvision import models
from transformers import DistilBertModel

class MultiModalModel(nn.Module):
    def __init__(self, num_genres=25, freeze_backbones=False):
        super().__init__()
        # Vision Tower
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.vision_tower = nn.Sequential(*list(resnet.children())[:-1])
        
        # Text Tower
        self.text_tower = DistilBertModel.from_pretrained("distilbert-base-uncased")
        
        if freeze_backbones:
            for p in self.vision_tower.parameters(): p.requires_grad = False
            for p in self.text_tower.parameters(): p.requires_grad = False

        # Fusion Head (2048 from ResNet + 768 from BERT = 2816)
        self.fusion = nn.Sequential(
            nn.Linear(2048 + 768, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_genres)
        )

    def forward(self, image, input_ids, attention_mask):
        v_feat = self.vision_tower(image).flatten(1)
        t_feat = self.text_tower(input_ids, attention_mask).last_hidden_state[:, 0, :]
        return self.fusion(torch.cat((v_feat, t_feat), dim=1))