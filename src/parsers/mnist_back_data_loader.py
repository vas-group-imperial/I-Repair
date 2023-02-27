import requests
import zipfile
import numpy as np
from torch.utils.data import TensorDataset
import torch


def download_url(url: str, save_path: str, chunk_size: int = 128):
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            if chunk:
                fd.write(chunk)


def unzip_dir(data_path: str, save_path: str):

    with zipfile.ZipFile(data_path, 'r') as zip_ref:
        zip_ref.extractall(save_path)


def create_dataset(data_dir: str, data_path: str, file_name: str):

    # load data from file
    data = np.loadtxt(data_path)

    # get images and convert to tensor
    imgs = data[:, :-1] / 1.0
    imgs_tensor = torch.from_numpy(imgs).float()

    # get labels and convert to tensor
    labels = data[:, -1:]
    labels_tensor = torch.flatten(torch.from_numpy(labels)).float()

    torch.save(TensorDataset(imgs_tensor, labels_tensor), data_dir + f"/{file_name}.pt")


if __name__ == "__main__":

    url = "http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_background_images.zip"
    local_filename = url.split("/")[len(url.split("/")) - 1]
    data_dir = "../../resources/"
    save_path_zip = data_dir + local_filename
    save_path = data_dir + "MNIST-BACK-IMAGE"

    # download_url(url, save_path_zip)

    # unzip_dir(save_path_zip, save_path)

    train_dset = create_dataset(save_path+"/mnist_background_images_train.amat")