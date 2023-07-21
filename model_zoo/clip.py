import collections
from functools import partial
import math

import torch
import clip
import torch.nn as nn
import torch.nn.functional as F

class CLIPWrapper(nn.Module):
    def __init__(self, name, num_classes) -> None:
        super().__init__()
        self.model, self.preprocess = clip.load(name)

    def prepare_text_enc(self, class_names, device):
        self.promts = [f"This is an image of {c}" for c in class_names]
        class_texts = clip.tokenize(self.promts).to(device)
        all_text_features = self.model.encode_text(class_texts)
        self.register_buffer("text_features", all_text_features)
    
    def forward(self, images, return_feature_list=False):
        feats = self.model.encode_image(images)
        logits = feats @ self.text_features.T
        if return_feature_list:
            return logits.float(), [feats.float()]
        return logits.float()
    
    def load(self, path):
        pass

    @property
    def is_clip(self):
        return True