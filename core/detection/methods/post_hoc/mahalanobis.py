from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import numpy as np
from  .base_detector import BaseDetector

import logging

logger = logging.getLogger(__name__)

class MahalanobisDetector(BaseDetector):
    def __init__(self, *, noise_magnitude, layer_index=0, DATASET_MEAN, DATASET_STD, **kwargs) -> None:
        super().__init__('mahalanobis')
        self.magnitude = noise_magnitude
        self.layer_index = layer_index
        self.sample_mean = None
        self.precision = None
        self.DATASET_MEAN = DATASET_MEAN
        self.DATASET_STD = DATASET_STD

    @torch.no_grad()
    def adapt(self, forward_fn, ref_id_dataloader, num_classes):
        '''
        Compute the mean and correlation matrix for ref id data 
        '''
        self.num_classes=num_classes
        self.sample_mean, self.precision = sample_estimator(forward_fn, 
                                                            self.num_classes, 
                                                            ref_id_dataloader)
                                                
    @property
    def require_adapt(self):
        return True
    
    def score_batch(self, forward_fn, batch):
        data: torch.Tensor = batch[0].cuda()
        data = data.requires_grad_(True) # Set requires_grad to True for input data
        batch[0] = data
        layer_index = self.layer_index
        out_features = forward_fn(batch, return_feature_list=True)[1][layer_index] # Expect features to be vectors
        # compute Mahalanobis score
        with torch.no_grad():
            gaussian_scores = self._gaussian_score(out_features, layer_index)
            md_preds = gaussian_scores.argmax(1)

        batch_sample_mean = self.sample_mean[layer_index][md_preds] # Get the mean of the predicted class
        centered_feats = out_features - batch_sample_mean
        md = 0.5*((centered_feats @ self.precision[layer_index]) @ centered_feats.T).diag()
        # print(md.shape)
        loss = torch.mean(md)
        loss.backward()
        
        gradient = torch.ge(data.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2

        gradient.index_copy_(1, torch.LongTensor([0]).cuda(), gradient.index_select(1, torch.LongTensor([0]).cuda()) / self.DATASET_STD[0])
        gradient.index_copy_(1, torch.LongTensor([1]).cuda(), gradient.index_select(1, torch.LongTensor([1]).cuda()) / self.DATASET_STD[1])
        gradient.index_copy_(1, torch.LongTensor([2]).cuda(), gradient.index_select(1, torch.LongTensor([2]).cuda()) / self.DATASET_STD[2])
        
        tempInputs = torch.add(data.data, gradient, alpha=-self.magnitude)
        with torch.no_grad():
            noise_out_features = forward_fn((tempInputs,), return_feature_list=True)[1][layer_index]
            noise_gaussian_scores = self._gaussian_score(noise_out_features, layer_index)
            noise_gaussian_scores = torch.max(noise_gaussian_scores, dim=1).values
        return {
            "scores": noise_gaussian_scores,
            "preds": md_preds
        }
    
    def _gaussian_score(self, out_features, layer_index):
        gaussian_scores = []
        for i in range(self.num_classes):
            zero_f = out_features.data - self.sample_mean[layer_index][i]
            term_gau = -0.5*torch.mm(torch.mm(zero_f, self.precision[layer_index]), zero_f.t()).diag()
            gaussian_scores.append(term_gau.view(-1,1))
        return torch.cat(gaussian_scores, 1)
    
    def __str__(self) -> str:
        return f"{self.name}_noise={self.magnitude}_layerindex={self.layer_index}"


@torch.no_grad()
def sample_estimator(forward_fn, num_classes, train_loader):
    """
    compute sample mean and precision (inverse of covariance)
    return: sample_class_mean: list of class mean
             precision: list of precisions
    """
    sample_input = next(iter(train_loader))
    feature_list = forward_fn(sample_input, return_feature_list=True)[1]
    feature_size_list = np.array([out.size(1) for out in feature_list])
    device = feature_list[0].device
    logger.info(f'Get sample mean and covariance. Num feature layers: {len(feature_size_list)}')
    list_features = []
    labels = []

    correct, total = 0, 0
    for batch in train_loader:
        target = batch[1]
        total += target.size(0)
        logits, out_features_list = forward_fn(batch, return_feature_list=True)
        pred = logits.cpu().argmax(1)
        correct += (pred == target).sum()

        list_features.append(out_features_list)
        labels.append(target)

    list_features = [torch.cat(feats) for feats in zip(*list_features)]
    labels = torch.cat(labels)

    sample_class_mean_list = []
    precision_list = []
    for k in range(len(list_features)):
        class_means = torch.Tensor(num_classes, feature_size_list[k]).to(device)
        for c in range(num_classes):
            c_inds = (labels == c)
            if c_inds.sum() > 0:
                class_means[c] = torch.mean(list_features[k][c_inds], dim=0)
                list_features[k][c_inds] -= class_means[c] # center the data by subtracting class mean

        sample_class_mean_list.append(class_means)
        # X = list_features[k].cpu().numpy()
        # group_lasso.fit(X)
        # temp_precision = group_lasso.precision_
        # temp_precision = torch.from_numpy(temp_precision).float().to(device)
        cov_k = (list_features[k].T @ list_features[k]) / list_features[k].shape[0] # get the covariance matrix
        precision_k = torch.linalg.pinv(cov_k, hermitian=True) # get the precision (inverse covariance) matrix
        precision_list.append(precision_k)

    logger.info('\n Training Accuracy:({:.2f}%)\n'.format(100. * correct / total))
    return sample_class_mean_list, precision_list
