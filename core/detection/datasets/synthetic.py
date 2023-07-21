from pathlib import Path
import torch
import numpy as np
import torchvision.transforms as trn
import torchvision.datasets as dset
from ood_detection.datasets import ImageFolderOOD
import copy
from glob import glob
from os import path as osp
from torchvision.datasets.folder import ImageFolder, default_loader
from PIL import Image

jigsaw = lambda x: torch.cat((
    torch.cat((torch.cat((x[:, 8:16, :16], x[:, :8, :16]), 1),
               x[:, 16:, :16]), 2),
    torch.cat((x[:, 16:, 16:],
               torch.cat((x[:, :16, 24:], x[:, :16, 16:24]), 2)), 2),
), 1)

speckle = lambda x: torch.clamp(x + x * torch.randn_like(x), 0, 1)
rgb_shift = lambda x: torch.cat((x[1:2].index_select(2, torch.LongTensor([i for i in range(32 - 1, -1, -1)])),
                                 x[2:, :, :], x[0:1, :, :]), 0)
invert = lambda x: torch.cat((x[0:1, :, :], 1 - x[1:2, :, ], 1 - x[2:, :, :],), 0)
pixelate = lambda x: x.resize((int(32 * 0.2), int(32 * 0.2)), Image.Resampling.BOX).resize((32, 32), Image.Resampling.BOX)

class AvgOfPair(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.shuffle_indices = np.arange(len(dataset))
        np.random.shuffle(self.shuffle_indices)

    def __getitem__(self, i):
        random_idx = np.random.choice(len(self.dataset))
        while random_idx == i:
            random_idx = np.random.choice(len(self.dataset))

        return self.dataset[i][0] / 2. + self.dataset[random_idx][0] / 2., 0

    def __len__(self):
        return len(self.dataset)

class GeomMeanOfPair(torch.utils.data.Dataset):
    def __init__(self, dataset, mean, std):
        self.dataset = dataset
        self.shuffle_indices = np.arange(len(dataset))
        np.random.shuffle(self.shuffle_indices)
        self.mean = mean
        self.std = std

    def __getitem__(self, i):
        random_idx = np.random.choice(len(self.dataset))
        while random_idx == i:
            random_idx = np.random.choice(len(self.dataset))

        return trn.Normalize(self.mean, self.std)(torch.sqrt(self.dataset[i][0] * self.dataset[random_idx][0])), 0

    def __len__(self):
        return len(self.dataset)

def get_uniform_dataset(num_examples, image_shape=(3, 32, 32)):
    dummy_targets = torch.ones(num_examples)
    ood_data = torch.from_numpy(
        np.random.uniform(size=(num_examples, *image_shape),
                        low=-1.0, high=1.0).astype(np.float32))
    return torch.utils.data.TensorDataset(ood_data, dummy_targets)

def get_avgofpair_dataset(base_ood_dataset):
    return AvgOfPair(base_ood_dataset)

def get_geomeanofpair_dataset(base_ood_dataset):
    return AvgOfPair(base_ood_dataset)

def get_jigsave_dataset(base_ood_dataset, mean, std):
    return _copy_and_replace_transform(base_ood_dataset, jigsaw, mean, std)

def get_speckle_dataset(base_ood_dataset, mean, std):
    return _copy_and_replace_transform(base_ood_dataset, speckle, mean, std)

def get_pixelate_dataset(base_ood_dataset, mean, std):
    return _copy_and_replace_transform(base_ood_dataset, pixelate, mean, std, is_PIL_op=True)

def get_rgbshift_dataset(base_ood_dataset, mean, std):
    return _copy_and_replace_transform(base_ood_dataset, rgb_shift, mean, std)

def get_inverted_dataset(base_ood_dataset, mean, std):
    return _copy_and_replace_transform(base_ood_dataset, invert, mean, std)

def _copy_and_replace_transform(base_ood_dataset, op, mean, std, is_PIL_op=False):
    base_ood_dataset = copy.copy(base_ood_dataset)
    if is_PIL_op:
        new_transform = trn.Compose([op, trn.ToTensor(),trn.Normalize(mean, std)])
    else:
        new_transform = trn.Compose([trn.ToTensor(), op, trn.Normalize(mean, std)])
    base_ood_dataset.transform = new_transform
    return base_ood_dataset