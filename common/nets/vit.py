import json
from PIL import Image

import numpy as np

import torch
from torchvision import transforms

from pytorch_pretrained_vit import ViT

model_name = 'B_16_imagenet1k'
model = ViT(model_name, pretrained=True)
print(model)
print(model.image_size)
# img = Image.open('img.jpg')
# Preprocess image
tfms = transforms.Compose([transforms.ToPILImage(),transforms.Resize(model.image_size), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),])
img = tfms(img).unsqueeze(0)
print(img.size())
model.eval()
with torch.no_grad():
    outputs = model(img).squeeze(0)
print(outputs.size())