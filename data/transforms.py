import random
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import *
from data.segmentation import statistics


class ImgRotation:
    """Produce 4 rotated versions of the input image."""

    def __init__(self):
        self.angles = [0, 90, 180, 270]

    def __call__(self, img):
        """
        Produce 4 rotated versions of the input image.
        Args:
            img: the input PIL image to be rotated.
        Returns:
            rotated_imgs: a list containing all the rotated versions of img.
            labels: a list containing the corresponding label for each rotated image in rotated_imgs.
        """
        rotated_imgs = []
        labels = []
        for idx, angle in enumerate(self.angles):
            rotated_imgs.append(TF.rotate(img, angle))
            labels.append(idx)
        assert len(rotated_imgs) == len(labels)
        return rotated_imgs, labels


class ApplyAfterRotations:
    """Apply a transformation to a list of images (e.g. after applying ImgRotation)"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        images, labels = x
        images = [self.transform(i) for i in images]
        return images, labels


class ToTensorAfterRotations:
    """Transform a list of images to a pytorch tensor (e.g. after applying ImgRotation)"""

    def __call__(self, x):
        images, labels = x
        return [TF.to_tensor(i) for i in images], [torch.tensor(l) for l in labels]


def get_transforms_pretraining(args):
    """Returns the transformations for the pretraining task."""
    size = [args.size] * 2
    train_transform = Compose(
        [
            Resize(size),
            RandomCrop(size, pad_if_needed=True),
            ImgRotation(),
            ApplyAfterRotations(RandomApply([ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8)),
            ToTensorAfterRotations(),
            ApplyAfterRotations(Normalize(statistics["mean"], statistics["std"])),
        ]
    )
    val_transform = Compose(
        [
            Resize(size),
            RandomCrop(size, pad_if_needed=True),
            ImgRotation(),
            ToTensorAfterRotations(),
            ApplyAfterRotations(Normalize(statistics["mean"], statistics["std"])),
        ]
    )
    return train_transform, val_transform


def get_transforms_binary_segmentation(args):
    """Returns the transformations for the binary segmentation task."""
    from PIL import Image

    size = [args.size] * 2
    train_transform = Compose(
        [
            Resize(size),
            RandomCrop(size, pad_if_needed=True),
            RandomHorizontalFlip(),
            RandomApply([ColorJitter(0.2, 0.2, 0.2, 0.1)], p=0.8),
            ToTensor(),
            Normalize(statistics["mean"], statistics["std"]),
        ]
    )
    train_transform_mask = Compose(
        [
            Resize(size, interpolation=Image.NEAREST),
            RandomCrop(size, pad_if_needed=True),
            RandomHorizontalFlip(),
            ToTensor(),
        ]
    )
    val_transform = Compose(
        [
            Resize(size),
            ToTensor(),
            Normalize(statistics["mean"], statistics["std"]),
        ]
    )
    val_transform_mask = Compose(
        [Resize(size, interpolation=Image.NEAREST), ToTensor()]
    )
    return train_transform, val_transform, train_transform_mask, val_transform_mask
