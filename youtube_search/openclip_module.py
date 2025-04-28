# openclip_module.py

import torch
import open_clip
from PIL import Image

class OpenClip:
    def __init__(self):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')

    def encode_image(self, pil_image):
        image_tensor = self.preprocess(pil_image).unsqueeze(0)
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features

    def encode_text(self, text):
        tokens = self.tokenizer([text])
        with torch.no_grad():
            text_features = self.model.encode_text(tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features

