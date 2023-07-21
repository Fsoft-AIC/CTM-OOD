from pathlib import Path
from typing import Callable
import torch
import torch.nn.functional as F
from tqdm import tqdm

def list_feature_by_class(features, labels):
    num_class = labels.max().item() + 1
    features_by_class = []
    for c in range(num_class):
        inds = (labels == c).squeeze()
        features_by_class.append(features[inds])
    return features_by_class

def mean_and_normalize_list_features(features_by_class):
    class_means = torch.cat([f.mean(dim=0, keepdim=True) for f in features_by_class])
    class_dir = F.normalize(class_means, dim=-1)
    return class_means, class_dir

class ForwardPass(Callable):
    def forward(self, batch, return_feature_list=False, penultimate_feature=False):
        '''
        Params:
            batch: (X, y)
            return_feature_list: default=False
            penultimate_feature: default=False
        '''
        raise NotImplementedError()
    
    def __call__(self, batch, return_feature_list=False, penultimate_feature=False):
        return self.forward(batch, return_feature_list=False, penultimate_feature=False)

class PreProcessFeature(Callable):
    def __init__(self, layer_index=0) -> None:
        super().__init__()
        self.layer_index = layer_index

    def _forward_collect(self, forward_fn, data):
        logits, features_list = forward_fn(data, return_feature_list=True)
        probs = F.softmax(logits, dim=1)
        features = features_list[self.layer_index]
        preds = probs.argmax(-1).squeeze()
        return features, probs, preds

    def adapt(self, forward_fn, train_loader, **kwargs):
        with torch.no_grad():
            outputs = []
            for batch in tqdm(train_loader, desc="Collect ref feats"):
                b_features, probs, preds = self._forward_collect(forward_fn, batch)
                labels = batch[1].to(b_features.device)
                outputs.append((probs, b_features, labels))
            probs, b_features, labels = zip(*outputs)
            probs = torch.vstack(probs)
            features = torch.vstack(b_features)
            labels = torch.cat(labels)

        Z = features
        Cov = torch.matmul(Z.t(), Z) * (1 / Z.shape[0])
        r = torch.linalg.matrix_rank(Cov)
        U, S, Vh = torch.linalg.svd(Cov)

        self._proj_matrix = U[:, :-120] @ Vh[:-120, :]

    def __call__(self, feats):
        return feats @ self._proj_matrix.T

class CalMeanClass:
    def __init__(self, layer_index=0) -> None:
        super().__init__()
        self.layer_index = layer_index

    def _forward_collect(self, forward_fn, data):
        logits, features_list = forward_fn(data, return_feature_list=True)
        probs = F.softmax(logits, dim=1)
        features = features_list[self.layer_index]
        preds = probs.argmax(-1).squeeze()
        return features, probs, preds

    def adapt(self, forward_fn, train_loader, *,cache_feats_path=None, cache_mean_path=None, normalize=False, **kwargs):
        if cache_feats_path is not None and Path(cache_feats_path).exists():
            print("Loading cached feats")
            probs, features, labels = torch.load(cache_feats_path, map_location="cpu")
        else:
            with torch.no_grad():
                outputs = []
                for batch in tqdm(train_loader, desc="Collect ref feats"):
                    b_features, probs, preds = self._forward_collect(forward_fn, batch)
                    labels = batch[1].to(b_features.device)
                    outputs.append((probs, b_features, labels))
                probs, b_features, labels = zip(*outputs)
                probs = torch.vstack(probs)
                features = torch.vstack(b_features)
                labels = torch.cat(labels)
        
        if normalize:
            # denom = features.norm(2, dim=1, keepdim=True).clamp_min(1e-12)
            # features = features / (denom**2)
            features_by_class = list_feature_by_class(F.normalize(features, dim=-1), labels)
            means, mean_dirs = mean_and_normalize_list_features(features_by_class)
        else:
            features_by_class = list_feature_by_class(features, labels)
            means, mean_dirs = mean_and_normalize_list_features(features_by_class)
        global_mean = features.mean(dim=0, keepdim=True)
        # mean_list = []
        # for c in labels.unique():
        #     c_inds = (labels == c)
        #     mean_list.append(features[c_inds].mean(dim=0, keepdim=True))
        # mean_list = torch.cat(mean_list)
        if cache_mean_path is not None:
            torch.save((means, global_mean), cache_mean_path)
        return means, global_mean