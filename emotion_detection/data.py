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
DATA_DIR = "../../fer2013/fer2013.csv"


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


def get_dataloaders_fer48(data_dir, batch_size=BATCH_SIZE, chunksize=10000, resize=None, hist_eq=False):

    """

    :param data_dir:
    :param batch_size:
    :param chunksize:
    :param transform:
    :return:
    """

    class HistEq(object):
        def __call__(self, im):
            res = np.expand_dims(exposure.equalize_hist(im), 2)
            # print(res.dtype)
            return res

    data_transforms = [transforms.ToTensor()]
    if resize:
        data_transforms = [transforms.Resize(resize)] + data_transforms
        if hist_eq:
            data_transforms.insert(1, HistEq())
    elif hist_eq:
        data_transforms = [HistEq()] + data_transforms

    data_transforms = transforms.Compose(data_transforms)
    # Load the datasets with ImageFolder
    image_datasets = {
        x: FerDataset48(data_dir, x, chunksize, data_transforms)
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

    def __init__(self, data_dir, flag, chunksize=20000, transform=None):
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
            int(i) for i in self.data['pixels'].iloc[idx].split(' ')
        ]).reshape((48, 48))
        # lab = np.array(self.data['emotion'].iloc[idx]).reshape((1, 1)).astype(np.uint8)
        lab = np.array(self.data['emotion'].iloc[idx]).astype(np.uint8)

        im, lab = self.transform(im).to(torch.float32), torch.from_numpy(lab).long()  # torch.from_numpy(lab).unsqueeze_(0)
        # print(im.dtype, im.size())
        # print(lab.dtype, lab.size())

        return im, lab

    def _read_csv(self, path_to_csv, chunksize, flag='Training'):
        chunks = pd.read_csv(path_to_csv, sep=',', chunksize=chunksize)
        list_chunks = []
        for chunk in chunks:
            mask = chunk['Usage'] == flag
            list_chunks.append(chunk.loc[mask])
        return pd.concat(list_chunks)

data_loader_lambda = {
    'get_dataloaders_fer48': get_dataloaders_fer48,
    'get_dataloaders': get_dataloaders
}