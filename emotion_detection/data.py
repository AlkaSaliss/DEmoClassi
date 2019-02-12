# Imports
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import os
import pandas as pd
import numpy as np
from skimage import exposure


# Define the transforms for the training and validation sets
BATCH_SIZE = 128
DATA_DIR = "../../fer2013/processed_images"


def get_dataloaders(batch_size=BATCH_SIZE, data_dir=DATA_DIR,
                    chunksize=10000, resize=None, hist_eq=False):

    chunksize = None
    resize = None
    hist_eq = None

    data_transforms = {
        'Training': transforms.Compose([
            transforms.ToTensor(),
        ]),
        'PublicTest': transforms.Compose([
            transforms.ToTensor(),
        ])
    }

    # Load the datasets with ImageFolder
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ['Training', 'PublicTest']
    }

    # Using the image datasets and the transforms, define the dataloaders
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=batch_size, num_workers=4, shuffle=True)
        for x in ['Training', 'PublicTest']
    }

    return dataloaders


def get_dataloaders_fer48(path_to_csv, batch_size=BATCH_SIZE, chunksize=10000, resize=None, hist_eq=False):

    """

    :param path_to_csv:
    :param batch_size:
    :param flag:
    :param chunksize:
    :param transform:
    :return:
    """

    class HistEq(object):
        def __call__(self, im):
            return exposure.equalize_hist(im)

    data_transforms = [transforms.ToTensor()]
    if resize:
        data_transforms.insert(0, transforms.Resize(resize))
    if hist_eq:
        data_transforms.insert(1, HistEq)

    # Load the datasets with ImageFolder
    image_datasets = {
        x: FerDataset48(path_to_csv, x, chunksize, data_transforms)
        for x in ['Training', 'PublicTest']
    }

    # Using the image datasets and the transforms, define the dataloaders
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=batch_size, num_workers=8, shuffle=True)
        for x in ['Training', 'PublicTest']
    }

    return dataloaders


class FerDataset48(Dataset):
    """
    Custom pytorch dataset class implementation to load utk_face images
    """

    def __init__(self, data_dir, flag, chunksize=10000, transform=None):
        """

        :param root_dir:
        :param transform:
        """
        self.data = self._read_csv(data_dir, chunksize, flag)
        self.transform = transform

    def __len__(self):
        """

        :return:
        """
        return self.data.shape[0]

    def __getitem__(self, idx):
        """

        :param idx:
        :return:
        """

        im = np.array([
            int(i) for i in self.data['pixels'][idx].split(' ')
        ]).reshape((1, 48, 48))
        lab = int(self.data['emotion'][idx])

        return self.transform(im), torch.tensor(lab, dtype=torch.int)

    def _read_csv(self, path_to_csv, chunksize, flag='Train'):
        chunks = pd.read_csv(path_to_csv, sep=',', chunksize=chunksize)
        list_chunks = []
        for chunk in chunks:
            mask = chunk['Flag'] == flag
            list_chunks.append(chunk.loc[mask])
        return pd.concat(list_chunks)

data_loader_lambda = {
    'get_dataloaders_fer48': get_dataloaders_fer48,
    'get_dataloaders': get_dataloaders
}