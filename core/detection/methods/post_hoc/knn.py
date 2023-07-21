from typing import Any

import faiss
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .base_detector import BaseDetector

normalizer = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10


class KNNPostprocessor(BaseDetector):
    def __init__(self, *, K, layer_index=0,**kwargs):
        super().__init__('knn')
        self.K = K
        self.activation_log = None
        self.layer_index = layer_index

    @torch.no_grad()
    def adapt(self, forward_fn, ref_id_dataloader, **kwargs):
        activation_log = []
        for batch in tqdm(ref_id_dataloader,
                            desc='Building KNN index table: ',
                            position=0,
                            leave=True):
            _, features = forward_fn(batch, return_feature_list=True)
            feature = features[self.layer_index]
            activation_log.append(normalizer(feature.data.cpu().numpy()))

        self.activation_log = np.concatenate(activation_log, axis=0)
        self.index = faiss.IndexFlatL2(feature.shape[1])
        res = faiss.StandardGpuResources() 
        self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        self.index.add(self.activation_log)

    
    @property
    def require_adapt(self):
        return True
    
    @torch.no_grad()    
    def score_batch(self, forward_fn, data: Any):
        output, features = forward_fn(data, return_feature_list=True)
        feature = features[self.layer_index]
        dim = feature.shape[1]
        feature_normed = normalizer(feature.data.cpu().numpy())
        D, _ = self.index.search(
            feature_normed,
            self.K,
        )
        kth_dist = D[:, -1]
        _, pred = torch.max(torch.softmax(output, dim=1), dim=1)
        return {
            "scores": -torch.from_numpy(kth_dist),
            "preds": pred
        }

    def set_hyperparam(self, hyperparam: list):
        self.K = hyperparam[0]

    def get_hyperparam(self):
        return self.K
    
    def __str__(self) -> str:
        return f"{self.name}_K={self.K}_layer-index={self.layer_index}"