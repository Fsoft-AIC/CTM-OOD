from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
import copy
from core.detection.methods.utils import CalMeanClass
from  .base_detector import BaseDetector

to_np = lambda x: x.data.cpu().numpy()
concat = lambda x: np.concatenate(x, axis=0)

energy_score = lambda smax, output, T: -T*torch.logsumexp(output / T, dim=1)
msp_score = lambda smax, output=None, T=None: -torch.max(smax, dim=1).values
xent_score = lambda smax, output, T: output.mean(1) - torch.logsumexp(output, dim=1)

class MSPDetector(BaseDetector):
    def __init__(self, use_xent=False, T=None, *args, **kwargs) -> None:
        super().__init__('msp')
        self.use_xent = use_xent
        self.T = T
        if self.use_xent:
            self.score_fn = lambda smax, output: -xent_score(smax, output, self.T)
        else:
            self.score_fn = lambda smax, output=None: torch.max(smax, dim=1).values
    
    @torch.no_grad()
    def score_batch(self, forward_fn, data):
        output = forward_fn(data, return_feature_list=False)
        probs = F.softmax(output, dim=1)
        preds = torch.argmax(output, dim=1)
        return {
            "scores": self.score_fn(probs, output), 
            "preds": preds
        }

    def __str__(self) -> str:
        return f"{self.name}_T={self.T}_use_xent={self.use_xent}"

class MLSDetector(BaseDetector):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__('mls')
    
    @torch.no_grad()
    def score_batch(self, forward_fn, data):
        logits = forward_fn(data, return_feature_list=False)
        preds = torch.argmax(logits, dim=1)
        return {
            "scores": logits.max(dim=-1).values, 
            "preds": preds
        }

    def __str__(self) -> str:
        return f"{self.name}"

class EnergyBasedDetector(MSPDetector):
    def __init__(self, T, *args, **kwargs) -> None:
        super().__init__()
        self.name = 'energy'
        self.T = T
        self.score_fn = lambda smax, output: -energy_score(smax, output, self.T)

    def __str__(self) -> str:
        return f"{self.name}_T={self.T}"
