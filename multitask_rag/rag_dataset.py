# Imports
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import numpy as np
import os
import glob
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from shutil import copy2
import tqdm


class RagDataset(Dataset):
    """
    Custom pytorch dataset class implementation to load utk_face images
    """

    def __init__(self, root_dir, transform=None):
        """

        :param root_dir:
        :param transform:
        """
        self.root_dir = root_dir
        self.list_images = glob.glob(os.path.join(self.root_dir, '*[jJ][pP]*'))
        self.transform = transform

    def __len__(self):
        """

        :return:
        """
        return len(self.list_images)

    def __getitem__(self, idx):
        """

        :param idx:
        :return:
        """
        # image = np.array(Image.open(self.list_images[idx]))
        image = Image.open(self.list_images[idx])
        image = self.transform(image).float()
        labels = self.list_images[idx].split('/')[-1].split('_')
        age, gender, race = torch.tensor(float(labels[0])).float(),\
                            torch.tensor(int(labels[1])).long(),\
                            torch.tensor(int(labels[2])).long()

        return image, age, gender, race


def split_utk(src_dir, dest_dir, train_split=0.7):
    """
    :param src_dir:
    :param dest_dir:
    :param train:
    :param val:
    :param test:
    :return:
    """

    os.makedirs(os.path.join(dest_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(dest_dir, 'valid'), exist_ok=True)
    os.makedirs(os.path.join(dest_dir, 'test'), exist_ok=True)
    train_path = os.path.join(dest_dir, 'train')
    val_path = os.path.join(dest_dir, 'valid')
    test_path = os.path.join(dest_dir, 'test')

    list_images = glob.glob(os.path.join(src_dir, '*jp*'))
    list_labels = [item.split('_') for item in list_images]
    age = [item[0] for item in list_labels]
    gender = [item[1] for item in list_labels]
    race = [item[2] for item in list_labels]
    labels = [i+j+k for i, j, k in zip(age, gender, race)]

    train_images, val_images, train_labels, val_labels = train_test_split(list_images, labels,
                                                                          test_size=1.0-train_split, stratify=labels)
    val_images, test_images = train_test_split(val_images, test_size=0.5, stratify=val_labels)

    def copy_images(dest_folder, list_images):
        for f in tqdm.tqdm(list_images):
            copy2(f, dest_folder)

    print('------------Copying train images-------------')
    copy_images(train_path, train_images)
    print('------------Copying valid images-------------')
    copy_images(val_path, val_images)
    print('------------Copying test images-------------')
    copy_images(test_path, test_images)


# Define the transforms for the training and validation sets
BATCH_SIZE = 32
# DATA_DIR = "..\\..\\UTKface_Aligned_cropped\\utk_data"
DATA_DIR = "/media/sf_Documents/COMPUTER_VISION/UTKface_Aligned_cropped/utk_face_split"


def get_dataloaders(batch_size=BATCH_SIZE, data_dir=DATA_DIR,
                    resize=(224, 224), normalize=False, **split_args):
    """

    :param batch_size:
    :param data_dir:
    :param split_args:
    :return:
    """

    class Identity(object):
        def __call__(self, obj):
            return obj

    if split_args.get('src_dir') and split_args.get('dest_dir') and split_args.get('train_split'):
        split_utk(split_args['src_dir'], split_args['dest_dir'], train_split=split_args['train_split'])
        data_dir = split_args['dest_dir']

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(15),
            transforms.Resize(resize),
            transforms.ToTensor(),

        ] + [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) if normalize else Identity()]),
        'valid': transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor()
        ] + [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) if normalize else Identity()]),
        'test': transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor()
        ] + [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) if normalize else Identity()])
    }

    # Load the datasets with ImageFolder
    image_datasets = {
        x: RagDataset(os.path.join(data_dir, x), data_transforms[x])
        for x in ['train', 'valid', 'test']
    }

    # Using the image datasets and the transforms, define the dataloaders
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=batch_size, num_workers=8, shuffle=True)
        for x in ['train', 'valid', 'test']
    }

    return dataloaders
