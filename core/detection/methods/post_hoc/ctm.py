from pathlib import Path
import torch
import torch.nn.functional as F
from core.detection.methods.utils import CalMeanClass
from  .base_detector import BaseDetector

class CTMDetector(BaseDetector):
    def __init__(self, *, cache_class_means=None, cache_feats=None, layer_index=0, **kwargs) -> None:
        super().__init__('ctm')
        self.cache_class_means = cache_class_means
        self.cache_feats = cache_feats
        self.layer_index = layer_index
    
    def adapt(self, forward_fn, ref_id_dataloader, *, net, **kwargs):
        if self.cache_class_means is not None and Path(self.cache_class_means).exists():
            class_means = torch.load(self.cache_class_means).to(net.weight.data.device)
            class_means = F.normalize(class_means, dim=-1)
        else:
            calmean = CalMeanClass(layer_index = self.layer_index)
            class_means, global_mean = calmean.adapt(forward_fn, ref_id_dataloader, cache_feats_path=self.cache_feats, cache_mean_path=self.cache_class_means, normalize=False)
            class_means = F.normalize(class_means, dim=-1)
        class_means.to(net.fc.weight.data.device)
        self._class_means = class_means
    
    @torch.no_grad()
    def score_batch(self, forward_fn, data):
        logits, feat_list = forward_fn(data, return_feature_list=True)
        preds = torch.argmax(logits, dim=1)
        
        feats = feat_list[self.layer_index]
        if self.normalize_feature:
            feats = F.normalize(feats, dim=-1)
        custom_logits = feats @ self._class_means.T
        scores = custom_logits.max(dim=-1).values
        return {
            "scores": scores, 
            "preds": preds
        }

    def __str__(self) -> str:
        return f"{self.name}_layer={self.layer_index}" 
