#!/usr/bin/env python
# coding: utf-8

# In[44]:


import torch
import open_clip
import numpy as np
import pandas as pd
from PIL import Image

class OpenClip:
    def __init__(self):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        self.model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')

    def displayImg(self, img_path: str):
        """Display an image"""
        Image.open(img_path).show()

    def encodeImage(self, img_path: str):
        """Encode image with OpenClip"""
        img = Image.open(img_path)
        image = self.preprocess(img).unsqueeze(0)
        with torch.no_grad():
            #encode image
            image_features = self.model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)

        return image_features

    def encodeText(self, text_list: List[str]):
        """Encode text with OpenClip"""
        text = self.tokenizer(text_list)
        text_features = self.model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features

    def getCosineSim(self, image_features: torch.tensor, text_features: torch.tensor):
        """calculate cosine similarity"""
        return (image_features @ text_features.T).detach()[0]



# In[ ]:





# In[ ]:





# In[ ]:




