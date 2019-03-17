# Imports
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import os
import pandas as pd
import numpy as np
from skimage import exposure
from skimage.transform import resize as sk_resize
import warnings
warnings.filterwarnings('ignore', category=UserWarning)


# Define the transforms for the training and validation sets
BATCH_SIZE = 256
DATA_DIR = "../../fer2013/fer2013.csv"


def get_dataloaders(batch_size=BATCH_SIZE, data_dir=DATA_DIR,
                    chunksize=10000, resize=None, to_rgb=True, hist_eq=False, normalize=False):

    """
    This function creates pytorch train and validation data loaders from a directory containing
    preprocessed images.
    The directory contains 2 subdirectories  named `Training` and `PublicTest`
    that contain resp. train and validation images. Note that in each of both dorectories there's also subdirectories,
     each one containing images for a given target class.

    :param batch_size: positive int for batch size
    :param data_dir: root folder containing `Training` and `PublicTest` subdirectories
    :param chunksize: This argument is not used, placed here for API consistency to have a common interface
                        with other data loaders functions
    :param resize: This argument is not used, placed here for API consistency to have a common interface
                        with other data loaders functions
    :param to_rgb: This argument is not used, placed here for API consistency to have a common interface
                        with other data loaders functions
    :param hist_eq: This argument is not used, placed here for API consistency to have a common interface
                        with other data loaders functions
    :param normalize: Boolean indicating whether to normalize the input images, useful when using pretrained models.
    :return: a dictionary with `Training` and `PublicTest` as keys, and  the corresponding values are pytorch
            `Dataloader` objects, that yields train and validation batches resp.
    """

    to_rgb = None
    chunksize = None
    resize = None
    hist_eq = None

    list_transforms = [transforms.ToTensor()]
    if normalize:
        list_transforms = list_transforms + [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    data_transforms = {
        'Training': transforms.Compose(list_transforms),
        'PublicTest': transforms.Compose(list_transforms)
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


def get_dataloaders_fer48(data_dir, batch_size=BATCH_SIZE, chunksize=10000,
                          resize=None, add_channel_dim=False, to_rgb=True, hist_eq=False, normalize=False):

    """
    Function to create pytorch data loaders from a csv contraining Fer2013 dataset

    :param data_dir: full path to the csv file
    :param batch_size: positive int for batch size
    :param chunksize: positive int, for loading the csv in chunks of this size. useful when we have a big file
                    that can not fit into memory
    :param resize: tuple of int (height, width), resize input image using this shape
    :param add_channel_dim: the fer images are (48, 48), but when using CNN we need three dimensions: height, width,
            and channel. This argument is a boolean telling whether to add this third dimension.
    :param to_rgb: fer images are gray scale, so when using transfer learning we need to convert them into rgb images.
            this argument is a boolean telling whether to repeat the grayscale image into 3 channels to mimic a colored
            image
    :param hist_eq: boolean, whether to apply histogram equalization to the grayscale image
    :param normalize: whether to normalize the image, using imagenet parameters
    :return: a dictionary with `Training` and `PublicTest` as keys, and  the corresponding values are pytorch
            `Dataloader` objects, that yields train and validation batches resp.
    """

    class AddChannel(object):
        def __call__(self, im):
            return np.expand_dims(im, 2)

    class HistEq(object):
        def __call__(self, im):
            # res = AddChannel()(exposure.equalize_hist(im))
            return exposure.equalize_hist(im)

    class ToRGB(object):
        def __call__(self, im):
            if len(im.shape) < 3:
                im = np.expand_dims(im, 2)
            return np.repeat(im, 3, axis=2)

    class SkResize(object):
        def __init__(self, size):
            self.size = size

        def __call__(self, im, size=None):
            return sk_resize(im, self.size)

    data_transforms = [transforms.ToTensor()]
    if resize:
        # data_transforms = [transforms.Resize(resize)] + data_transforms
        data_transforms = [SkResize(resize)] + data_transforms
        if hist_eq:
            data_transforms.insert(1, HistEq())
            if to_rgb:
                data_transforms.insert(2, ToRGB())
            elif add_channel_dim:
                data_transforms.insert(2, AddChannel())
        elif to_rgb:
            data_transforms.insert(1, ToRGB())

    elif hist_eq:
        data_transforms = [HistEq()] + data_transforms
        if to_rgb:
            data_transforms.insert(1, ToRGB())
        elif add_channel_dim:
            data_transforms.insert(1, AddChannel())
    elif to_rgb:
        data_transforms = [ToRGB()] + data_transforms
    elif add_channel_dim:
        data_transforms = [AddChannel()] + data_transforms

    if normalize:
        data_transforms = data_transforms + [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]

    data_transforms = transforms.Compose(data_transforms)

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
    Custom pytorch `Dataset` class implementation to load fer images dataset.
    This class loads the images from the original fer raw csv file
    """

    def __init__(self, data_dir, flag, chunksize=20000, transform=None):

        """

        :param data_dir: path to the csv file containing the data
        :param flag: string indicating which slice of data to load. IN the fer csv file, there are 3 sets of data:
                `Training`, `PublicTest`, `PrivateTest`
        :param chunksize: positive int, for loading the csv in chunks of this size. useful when we have a big file
                    that can not fit into memory
        :param transform: pytorch transform to apply to the image
        """

        self.data = self._read_csv(data_dir, chunksize, flag)
        self.transform = transform

    def __len__(self):
        """

        :return: the size of the dataset
        """
        return self.data.shape[0]

    def __getitem__(self, idx):
        """

        :param idx: retrieve the image and label at position `idx`
        :return: 2 pytorch tensors representing th eimage and the label resp.
        """

        im = np.array([
            int(i) for i in self.data['pixels'].iloc[idx].split(' ')
        ]).reshape((48, 48))

        lab = np.array(self.data['emotion'].iloc[idx]).astype(np.uint8)

        im, lab = self.transform(im).to(torch.float32), torch.from_numpy(lab).long()

        return im, lab

    def _read_csv(self, path_to_csv, chunksize, flag='Training'):
        chunks = pd.read_csv(path_to_csv, sep=',', chunksize=chunksize)
        list_chunks = []
        for chunk in chunks:
            mask = chunk['Usage'] == flag
            list_chunks.append(chunk.loc[mask])
        return pd.concat(list_chunks)


# a dictionary mapping the two dataloaders functions defined above with their names
data_loader_lambda = {
    'get_dataloaders_fer48': get_dataloaders_fer48,
    'get_dataloaders': get_dataloaders
}
