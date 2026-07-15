import torch

from torchvision.transforms import v2
from torchvision.datasets import ImageFolder

import numpy as np

def create_dataloader(path, 
                      augment=False, 
                      valid_size=0.25, 
                      seed=42, 
                      batch_size=64):

    train_transform = _build_transform(augment)
    valid_transform  = _build_transform(False)

    train_dataset = ImageFolder(root=path, 
                                transform=train_transform)
    val_dataset = ImageFolder(root=path, 
                              transform=valid_transform)

    dataset_length = len(train_dataset)
    indicies = np.arange(dataset_length)
    split = int(np.floor(dataset_length*valid_size))

    #randomly shuffle the indicies
    np.random.seed(seed)
    np.random.shuffle(indicies)

    # train, validation split
    train_idx, val_idx = indicies[split:], indicies[:split]

    train_set = torch.utils.data.Subset(train_dataset, train_idx)
    val_set = torch.utils.data.Subset(val_dataset, val_idx)

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               shuffle=True)
    valid_loader = torch.utils.data.DataLoader(val_set, 
                                               batch_size=batch_size,
                                               shuffle=False)
    
    return train_loader, valid_loader, train_dataset.classes


def _build_transform(augment: bool):
    """Build the image preprocessing pipeline (paper Sections 3.2 and 3.3.1).

    All images are resized from 64x64 to 224x224 (the input size the
    ImageNet-pretrained VGG16 / Wide ResNet-50 backbones expect), converted
    to tensors, and normalised with the ImageNet channel mean/std that the
    pretrained weights were trained with.

    Args:
        augment: When True, prepend the augmentation steps used in the
            authors' reference notebooks (random horizontal flip, rotation,
            random vertical flip) — matches the pipeline that produced the
            paper's reported 98.55% / 99.17% with-augmentation results.
            Validation loaders must always pass False.

    Returns:
        A torchvision ``transforms.v2.Compose`` pipeline.
    """
    augmentations = [
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(20),
        v2.RandomVerticalFlip(p=0.5),
    ] if augment else []

    return v2.Compose([
        v2.Resize((224, 224)),
        *augmentations,
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])