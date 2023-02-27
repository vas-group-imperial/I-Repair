
"""
A Cifar10 model used for testing the verification VeriNet.

Author: Patrick Henriksen <patrick@henriksen.as>
"""

import os

import torch
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

from torch.utils.data import Dataset
import torchvision.transforms as trans


# noinspection PyPep8Naming,PyShadowingNames,PyTypeChecker
class TinyImageNetDSet(Dataset):

    """
    A dataset class for the ImageNet dataset.
    """

    def __init__(self, root: str, train: bool = True, transform: trans = None):

        file_dir = os.path.join(root, "train.h5") if train else os.path.join(root, "test.h5")

        if not os.path.isfile(file_dir):
            print("h5py file not found, creating..")
            create_h5py(root, train)

        self.h5py_file = h5py.File(file_dir, 'r')

        self.transform = transform

    def __len__(self):
        return len(self.h5py_file['labels'])

    def __getitem__(self, idx):

        image = torch.from_numpy(self.h5py_file['images'][idx].astype(np.float32))/256
        label = int(self.h5py_file['labels'][idx])

        if self.transform:
            image = self.transform(image)

        return image, label


def create_h5py(root_dir: str, train: bool, num_classes: int = 200):

    """
    Converts from the ImageNet folder/image structure to h5py.

    Args:
        root_dir:
            The root directory of the ImageNet images.
        train:
            If true, the training set is created, otherwise the test set.
        num_classes:
            The number of classes to extract. 
    """

    images = []
    labels = []

    labels_dict = create_label_dict(os.path.join(root_dir, "wnids.txt"))

    if train:
        data_dir = os.path.join(root_dir, "train")
    else:
        data_dir = os.path.join(root_dir, "val")

    for identifier in tqdm(os.listdir(data_dir), desc="Reading images"):

        class_dir = os.path.join(data_dir, identifier)

        if not os.path.isdir(class_dir):
            continue

        filenames = os.listdir(class_dir)
        label = labels_dict[identifier]

        if label >= num_classes:
            continue

        for filename in filenames:

            if filename[-5:] != ".JPEG":
                continue

            image = Image.open(os.path.join(class_dir, filename)).resize((64, 64), Image.BILINEAR)
            image = image.convert('RGB')
            image = np.frombuffer(image.tobytes(), dtype=np.uint8)

            if len(image) == 3*64*64:
                images.append(image.reshape(64, 64, 3).transpose(2, 0, 1))
                labels.append(label)
            else:
                raise TypeError(f"Expected image of lenght: {3*64*64}, got: {len(image)}")

    rand_idx = np.arange(len(labels))
    np.random.shuffle(rand_idx)
    images = list(np.array(images)[rand_idx])
    labels = list(np.array(labels)[rand_idx])

    if train:
        hf = h5py.File(os.path.join(root_dir, "train.h5"), 'w')
    else:
        hf = h5py.File(os.path.join(root_dir, "test.h5"), 'w')

    hf.create_dataset('images', data=images)
    hf.create_dataset('labels', data=labels)

    hf.close()


def create_label_dict(file_dir: str):

    """
    Creates the label dict on form: {id(str): label(int)}

    Args:
        file_dir:
            The directory of the words file.
    """

    label_dict = {}
    with open(file_dir, "r") as file:

        for i, line in enumerate(file):

            identifier = line.strip()
            label_dict[identifier] = i

    return label_dict


if __name__ == '__main__':

    TinyImageNetDSet("../../../resources/tiny-imagenet/tiny-imagenet-200/", train=True)
    TinyImageNetDSet("../../../resources/tiny-imagenet/tiny-imagenet-200/", train=False)
