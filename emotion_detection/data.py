# Imports
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os


# Define the transforms for the training and validation sets
BATCH_SIZE = 128
DATA_DIR = "../../fer2013/processed_images"


def get_dataloaders(batch_size=BATCH_SIZE, data_dir=DATA_DIR, preprocess=False):
    if preprocess:
        pass

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
