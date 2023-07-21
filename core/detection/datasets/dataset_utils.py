import logging
from glob import glob
from os import path as osp

import numpy as np
import torch
from PIL import Image
from torchvision import datasets
from torchvision.datasets.folder import default_loader

logger = logging.getLogger(__name__)
testsets_names = [
    "iSUN",  # iSUN
    "LSUN_resize",  # LSUN (resize)
    "LSUN",  # LSUN (crop)
    "Imagenet_resize",  # Tiny-ImageNet (resize)
    "Imagenet",  # Tiny - ImageNet(crop)
    "Uniform",  # Uniform noise
    "Gaussian",  # Gaussian noise
]


class ImageFolderOOD(datasets.VisionDataset):
    def __init__(self, root, *args, **kwargs):
        super().__init__(root, *args, **kwargs)
        img_path_list = (
            glob(osp.join(root, "*", "*.jpeg"))
            + glob(osp.join(root, "*", "*.png"))
            + glob(osp.join(root, "*", "*.jpg"))
            + glob(osp.join(root, "*", "*", "*.JPEG"))
            + glob(osp.join(root, "*", "*.JPEG"))
        )
        if len(img_path_list) == 0:
            logger.error("Dataset was not downloaded {}".format(root))

        self.data_paths = img_path_list
        self.targets = [-1] * len(img_path_list)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img_path, target = self.data_paths[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = np.array(default_loader(img_path))
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data_paths)
        


class UniformNoiseDataset(datasets.VisionDataset):
    """
    Create dataset with random noise images in the same structure of CIFAR10
    """

    def __init__(self, root, *args, **kwargs):
        super().__init__(root, *args, **kwargs)
        # Create random data and labels

        self.data = np.random.randint(0, 255, (10000, 32, 32, 3)).astype("uint8")
        self.targets = [-1] * 10000

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


class GaussianNoiseDataset(datasets.VisionDataset):
    """
    Create dataset with random noise images in the same structure of CIFAR10
    """

    def __init__(self, root, *args, **kwargs):
        super().__init__(root, *args, **kwargs)
        # Create random data and labels
        self.targets = [-1] * 10000

        self.data = 255 * np.random.randn(10000, 32, 32, 3) + 255 / 2
        self.data = np.clip(self.data, 0, 255).astype("uint8")

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


class FeaturesDataset(datasets.VisionDataset):
    def __init__(
        self,
        features_list: list,
        labels_list: list,
        outputs_list: list,
        prob_list: list,
        *args,
        **kwargs,
    ):
        super().__init__("", *args, **kwargs)

        self.data = torch.cat(features_list).cpu().numpy()
        self.outputs = torch.cat(outputs_list).cpu().numpy()
        self.targets = torch.cat(labels_list).cpu().numpy()
        self.probs = torch.cat(prob_list).cpu().numpy()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)