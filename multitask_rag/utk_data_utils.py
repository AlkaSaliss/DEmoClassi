# Imports
from torch.utils.data import DataLoader, Dataset
import os
import glob
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from shutil import copy2
import tqdm
import matplotlib.pyplot as plt
import random
import numpy as np


class RagDataset(Dataset):
    """
    Custom pytorch dataset class implementation to load utk_face images
    """

    def __init__(self, root_dir, transform=None, n_samples=None):

        """
        Custom pytorch dataset class for loding utk face images along with labels : age, gender  and race

        :param root_dir: root folder containing the raw images
        :param transform: torchvision `transforms.Compose` object containing transformation to be applied to images
        :param n_samples: int, number of images samples to consider, useful for taking a small subset for debugging
        """

        self.root_dir = root_dir
        self.list_images = glob.glob(os.path.join(self.root_dir, '*[jJ][pP]*'))
        self.transform = transform
        self.n_samples = n_samples

    def __len__(self):
        """
        :return: The total number of samples in the dataset
        """
        if self.n_samples is None:
            return len(self.list_images)
        return self.n_samples

    def __getitem__(self, idx):
        """
        selects a sample and returns it in the right format as model input

        :param idx: int representing a sample indew in the whole dataset
        :return: the sample image at position idx, as pytorch tensor, and corresponding labels
        """
        image = Image.open(self.list_images[idx])
        image = self.transform(image).float()

        # In the utk face dataset, labels are contained in image names, for instance an image of age 23, black man
        # is typically named as `root_dir/23_0_1_158478565845.jpg`
        labels = self.list_images[idx].split('/')[-1].split('_')
        age, gender, race = torch.tensor(float(labels[0])).float(),\
                            torch.tensor(int(labels[1])).long(),\
                            torch.tensor(int(labels[2])).long()

        return image, age, gender, race


def split_utk(src_dir, dest_dir, train_split=0.7):
    """
    Utility function for spliting the dataset into train validation and test set.

    :param src_dir: directory containnig the images
    :param dest_dir: directory where to save the images in 3 subdirectories : `train`, `valid` and `test`
    :param train_split: a float between 0 and 1, representing the percentage of the training set, the rest will be
            equally distributed into valid and test sets
    :return:
    """

    os.makedirs(os.path.join(dest_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(dest_dir, 'valid'), exist_ok=True)
    os.makedirs(os.path.join(dest_dir, 'test'), exist_ok=True)
    train_path = os.path.join(dest_dir, 'train')
    val_path = os.path.join(dest_dir, 'valid')
    test_path = os.path.join(dest_dir, 'test')

    list_images = glob.glob(os.path.join(src_dir, '*.jp*'))
    list_images = [im for im in list_images if len(im.split('/')[-1].split('_')) == 4]
    list_labels = [item.split('/')[-1].split('_') for item in list_images]
    # age = [item[0] for item in list_labels]
    gender = [item[1] for item in list_labels]
    race = [item[2] for item in list_labels]
    labels = [j+k for j, k in zip(gender, race)]
    # labels = [i + j + k for i, j, k in zip(age, gender, race)]

    train_images, val_images, train_labels, val_labels = train_test_split(list_images, labels,
                                                                          test_size=1.0-train_split,
                                                                          stratify=labels,
                                                                          random_state=123)
    val_images, test_images = train_test_split(val_images, test_size=0.5, stratify=val_labels, random_state=234)

    def copy_images(dest_folder, list_images_):
        for f in tqdm.tqdm(list_images_):
            copy2(f, dest_folder)

    print('------------Copying train images-------------')
    copy_images(train_path, train_images)
    print('------------Copying valid images-------------')
    copy_images(val_path, val_images)
    print('------------Copying test images-------------')
    copy_images(test_path, test_images)


def get_utk_dataloader(batch_size=256, data_dir=None, n_samples=None, data_transforms=None, flag='train', **split_args):

    """
    Utility function for creating train, validation and test data loaders

    :param batch_size: int, size of batches
    :param data_dir: directory containing subdirectories `train` and `valid` and `test`
    :param n_samples: number of sample images to consider, useful for debugging with small subset
    :param data_transforms: torchvision `transforms.Compose` object containing transformation to be applied to images
    :param flag: string representing the set to be loaded among : `train`, `valid` and `test`
    :param split_args: kwargs as take by the function `split_utk()` defined above. This can be used to split the data
            into train, validation and test sets if this is not already done. In thsi case the argument `data_dir` is
            no longer needed as the data directory will be the dest_dir from `split_args`
    :return: pytorch `DataLoader` object to iterate through and get batches
    """

    # if relevant split the image data into train, validation and test set
    if split_args.get('src_dir') and split_args.get('dest_dir') and split_args.get('train_split'):
        split_utk(split_args['src_dir'], split_args['dest_dir'], train_split=split_args['train_split'])
        data_dir = split_args['dest_dir']

    image_dataset = RagDataset(os.path.join(data_dir, flag), data_transforms, n_samples)
    shuffle = True if flag == 'train' else False
    dataloaders = DataLoader(image_dataset, batch_size=batch_size, num_workers=8, shuffle=shuffle)

    return dataloaders


def display_examples_utk(root_path, label_type, label_value):

    list_files = [item for item in glob.glob(os.path.join(root_path, '*.jpg'))
                  if len(item.split('/')[-1].split('_')[:-1]) == 3]

    labels = [item.split('/')[-1].split('_')[:-1] for item in list_files]

    labels = {
        'age': [int(item[0]) for item in labels],
        'gender': [int(item[1]) for item in labels],
        'race': [int(item[2]) for item in labels]
    }

    label_names = {
        "race": {0: 'White', 1: 'Black', 2: 'Asian', 3: 'Indian', 4: 'Other'},
        "gender": {0: 'Male', 1: 'Female'},
    }

    print('Sample images for {} : {}'.format(
        label_type, label_names[label_type][label_value] if label_type != 'age' else label_value
    ))
    inds = [ind for ind, lab in enumerate(labels[label_type]) if lab == label_value]

    samples = random.sample(inds, k=4)

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 5))
    ax = ax.ravel()
    for idx, im_ind in enumerate(samples):
        im = np.asarray(Image.open(list_files[im_ind]))
        ax[idx].imshow(im)
