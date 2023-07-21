from pathlib import Path
from functools import partial
import numpy as np
from torch.utils.data import Subset
import torchvision.transforms as trn
import torchvision.transforms.functional as TF
import torchvision.datasets as dset
from core.detection.datasets import ImageFolderOOD
from PIL import Image
from typing import Callable, Optional

from PIL import Image
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder

import logging
logger = logging.getLogger(__name__)

# Old norm params, still need for evaluate standart ckpt
CIFAR10_MEAN = [x / 255 for x in [125.3, 123.0, 113.9]]
CIFAR10_STD = [x / 255 for x in [63.0, 62.1, 66.7]]

NORM_PARAMS = {
    'cifar10': {
        # 'mean': (0.4914, 0.4822, 0.4465),
        # 'std': (0.2023, 0.1994, 0.2010)
        'mean': CIFAR10_MEAN,
        'std': CIFAR10_STD
    },
    'cifar100': {
        # 'mean': (0.5071, 0.4867, 0.4408),
        # 'std': (0.2675, 0.2565, 0.2761)
        'mean': CIFAR10_MEAN,
        'std': CIFAR10_STD
    },
    'imagenet': {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    }
}

DATASETS = {
    'cifar10': {
        'far': ['svhn', 'lsun-c', 'lsun-r', 'isun', 'textures', 'places365'],
        'near': ['cifar100']
    },
    'cifar100': {
        'far': ['svhn', 'lsun-c', 'lsun-r', 'isun', 'textures', 'places365'],
        'near': ['cifar10']
    },
    'imagenet': {
        'far': ['inaturalist', 'sun', 'places', 'textures'],
        'near': []
    },
}

ID2PRINTNAME = {
    'mnist': 'MNIST',
    'svhn': "SVHN",
    'lsun-c': "LSUN-C",
    'lsun-r': "LSUN-R",
    'isun': "iSUN",
    'imagenet-r': "Imagenet-R",
    'imagenet-c': "Imagenet-C",
    'cifar10': "CIFAR-10",
    'cifar100': "CIFAR-100",
    'textures': "Textures",
    'places365': "Places365",
    'inaturalist': "iNaturalist",
    'sun': "SUN",
    'places': "Places",
    'imagenet': "ImageNet",
}

def get_id_transform(in_dataset_name,aug=False):
    '''Returns a transform for the in-distribution dataset'''
    # remove_background = trn.Compose([trn.CenterCrop((30,30)), trn.Pad(32 - 30, fill=CIFAR10_MEAN)])
    if in_dataset_name == 'cifar10' or in_dataset_name == 'cifar100':
        id_transform = trn.Compose([
                trn.Resize(32),
                trn.CenterCrop(32),
                trn.ToTensor(), 
                trn.Normalize(NORM_PARAMS[in_dataset_name]['mean'], 
                                NORM_PARAMS[in_dataset_name]['std'])])
        if aug:
            id_transform = trn.Compose([
                    trn.RandomHorizontalFlip(),
                    trn.Resize(32),
                    trn.RandomCrop(32, padding=2),
                    # trn.CenterCrop(32),
                    trn.ToTensor(), 
                    trn.Normalize(NORM_PARAMS[in_dataset_name]['mean'], 
                                    NORM_PARAMS[in_dataset_name]['std'])])
    elif in_dataset_name == 'imagenet':
        id_transform = trn.Compose([
            trn.Resize(256),
            trn.CenterCrop(224),
            trn.ToTensor(),
            trn.Normalize(mean=NORM_PARAMS[in_dataset_name]['mean'], std=NORM_PARAMS[in_dataset_name]['std']),
        ])
    return id_transform
        
def get_ood_dataset_builder(DATA_ROOT):
    '''Returns a function that builds a dataset given a name'''
    return {
        'mnist': partial(dset.MNIST, root= DATA_ROOT, train=False),
        'svhn': partial(dset.SVHN, root= DATA_ROOT / 'SVHN', split="test"),
        'lsun-c': partial(ImageFolderOOD, root=DATA_ROOT / 'LSUN'),
        'lsun-r': partial(ImageFolderOOD, root=DATA_ROOT / 'LSUN_resize'),
        'isun': partial(ImageFolderOOD, root=DATA_ROOT / 'iSUN'),
        'imagenet-r': partial(ImageFolderOOD, root=DATA_ROOT / 'Imagenet_resize'),
        'imagenet-c': partial(ImageFolderOOD, root=DATA_ROOT / 'Imagenet'),
        'textures': partial(ImageFolderOOD, root=DATA_ROOT / 'dtd' / 'images'),
        'places365': partial(ImageFolderOOD, root=DATA_ROOT / 'places365'),
        'places': partial(ImageFolderOOD, root=DATA_ROOT / 'Places'),
        'inaturalist': partial(ImageFolderOOD, root=DATA_ROOT / 'iNaturalist'),
        'sun': partial(ImageFolderOOD, root=DATA_ROOT / 'SUN'),
        'cifar10': partial(dset.CIFAR10, root= DATA_ROOT, train=False),
        'cifar100': partial(dset.CIFAR100, root= DATA_ROOT, train=False),
    }

def get_id_datasets_dict(DATA_ROOT: Path, in_dataset: str, test_transform=None):
    '''Returns a dictionary of in-distribution datasets and metadata e.g. number of classes, etc.'''
    ds = {}
    logger.info(f"Loading ID dataset [{in_dataset}] from {DATA_ROOT}")
    if test_transform is None:
        test_transform = get_id_transform(in_dataset)
    if in_dataset == 'cifar10':
        ds["train"] = dset.CIFAR10(DATA_ROOT, train=True, transform=test_transform) # type: ignore
        ds["test"] = dset.CIFAR10(DATA_ROOT, train=False, transform=test_transform) # type: ignore
        classes = ds['train'].classes
        NUM_CLASSES = 10
    elif in_dataset == 'cifar100':
        ds["train"] = dset.CIFAR100(DATA_ROOT, train=True, transform=test_transform) # type: ignore
        ds["test"] = dset.CIFAR100(DATA_ROOT, train=False, transform=test_transform) # type: ignore
        classes = ds['train'].classes
        NUM_CLASSES = 100
    elif in_dataset == 'imagenet':
        IMAGENET_PATH = Path('host_data/imagenet_full')
        ds["train"] = ImageFolder(IMAGENET_PATH / 'train', transform=test_transform) # type: ignore
        classes = ds['train'].classes
        # _, labels = zip(*ds['train'].samples)
        # sort_idx = np.argsort(labels)
        # ds["train"] = Subset(ds["train"], sort_idx)
        # ds['train'] = Subset(ds['train'], np.random.choice(len(ds['train']), int(0.5*len(ds['train'])), replace=False))
        ds["test"] = ImageFolder(IMAGENET_PATH / 'val', transform=test_transform) # type: ignore
        NUM_CLASSES = 1000
    else:
        raise ValueError
    return {
        "ds": ds,
        "meta": {"num_classes": NUM_CLASSES, "class_names": classes}
    }

def get_ood_datasets_dict(DATA_ROOT: Path, in_dataset: str, id_transform=None):
    '''Returns a dictionary of OOD datasets and metadata e.g. in-distribution dataset name, etc.'''
    if id_transform is None:
        id_transform = get_id_transform(in_dataset)
    dataset_builder = get_ood_dataset_builder(DATA_ROOT)
    OOD_DATASETS = {'far': {}, 'near': {}}
    for dataset_id in DATASETS[in_dataset]['far']:
        OOD_DATASETS['far'][ID2PRINTNAME[dataset_id]] = dataset_builder[dataset_id](transform=id_transform)
    for dataset_id in DATASETS[in_dataset]['near']:
        OOD_DATASETS['near'][ID2PRINTNAME[dataset_id]] = dataset_builder[dataset_id](transform=id_transform)
    return {
        "ds": OOD_DATASETS,
        "meta": {
            "in_dataset": in_dataset
        }
    }
