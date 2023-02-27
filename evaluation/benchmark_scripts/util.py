
"""
A few util scripts for running benchmarks
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.datasets as dset

# noinspection PyPep8Naming
import torchvision.transforms as transforms

from src.models.ffnn import FFNN
from src.models.util.tiny_imgnet_dset import TinyImageNetDSet


# noinspection PyArgumentList
def create_sets(model: FFNN, dataloader: DataLoader, samples: int, classes: list,
                correctly_classified: bool, balanced: bool = False, total_num_class: int = 10,
                max_margin: float = None):

    """
    Naive method to create specific sets by running inputs through the net

    Args:
        model:
            neural network to be repaired
        dataloader:
            The dataloader
        samples:
            number of specific inputs to be selected
        classes:
            A list of the classes to pick from.
        correctly_classified:
            If true, only correctly classified are picked, otherwise only misclassified.
        balanced:
            If true, the same amount of samples are selected from all classes. Note
            that this option expects samples to be divisible by len(classes).
        total_num_class:
            The number of classes in the dataset.
        max_margin:
            If not none and correctly_classified = False, only datapoints with a margin
            smaller than max_margin are picked.
    """

    inputs, labels = [], []

    class_size = samples//len(classes) + (samples % len(classes) > 0)
    picked_per_class = np.zeros(total_num_class)

    for data_org, labels_org in dataloader:
        for image, label in zip(data_org, labels_org):

            if label not in classes:
                continue

            scores = model(torch.Tensor(image))
            predicted = scores.argmax(dim=1)

            if balanced and picked_per_class[predicted] == class_size:
                continue

            if correctly_classified and predicted == label:
                inputs.append(image)
                labels.append(label)
                picked_per_class[predicted] += 1

            elif not correctly_classified and predicted != label:

                if max_margin is None or (scores[0, predicted[0]] - scores[0, int(label)]) < max_margin:
                    inputs.append(image)
                    labels.append(label)
                    picked_per_class[predicted] += 1

            if len(inputs) == samples:
                break

        if len(inputs) == samples:
            break

    if len(inputs) == 0:
        return None, None

    return torch.stack(inputs), torch.stack(labels)


def target_tensor_to_array_sip(target_tensor: torch.Tensor) -> np.array:

    """
    Converts the target tensor to a numpy array for SIP.

    Args:
        target_tensor:
            The target tensor
    Returns:
        The target tensor as a numpy array.
    """

    converted = np.zeros((target_tensor.shape[0]))
    converted[:] = target_tensor

    return converted.astype(int)


def mnist_data_loader(data_dir: str, num_train: int = 49000, normalize: bool = True, batch_size: int = 100):

    """
    MNIST data loader. Creates train, val sets and two test sets: one to be used for repair and one for testing.

    Args:
        data_dir    :
            Data to be loaded - downloaded if not found
        num_train   :
            The number of training examples
        normalize   :
            Indicates whether normalization is used
        batch_size  :
            The batch size of the datasets
    """

    torch.manual_seed(0)

    if normalize:
        mean = (0.1307,)
        std = (0.3081,)
        trns_norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    else:
        trns_norm = transforms.Compose([transforms.ToTensor()])

    mnist_train = dset.MNIST(data_dir, train=True, download=True, transform=trns_norm)
    train_loader = DataLoader(mnist_train, batch_size=batch_size, sampler=sampler.SubsetRandomSampler(range(num_train)))
    val_loader = DataLoader(mnist_train, batch_size=batch_size,
                            sampler=sampler.SubsetRandomSampler(range(num_train, 50000)))

    mnist_test = dset.MNIST(data_dir, train=False, download=True, transform=trns_norm)
    test_loader = DataLoader(mnist_test, batch_size=batch_size, sampler=sampler.SubsetRandomSampler(range(0, 5000)))
    test_loader_repair = DataLoader(mnist_test, batch_size=batch_size,
                                    sampler=sampler.SubsetRandomSampler(range(5000, 10000)))

    return train_loader, val_loader, test_loader, test_loader_repair


def cifar_data_loader(data_dir: str = '../../resources/cifar/', num_train: int = 49000, batch_size: int = 100):

    """
    Cifar data loader. Creates train, val sets and two test sets: one to be used for repair and one for testing.

    Args:
        data_dir:
            Data to be loaded - downloaded if not found
        num_train:
            The number of training examples
        batch_size  :
        The batch size of the datasets
    """

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    trans = transforms.Compose([transforms.ToTensor(), normalize])

    train_dset = dset.CIFAR10(root=data_dir, train=True, transform=trans, download=False)
    train_loader = torch.utils.data.DataLoader(train_dset, batch_size=batch_size,
                                               sampler=sampler.SubsetRandomSampler(range(num_train)))
    val_loader = torch.utils.data.DataLoader(train_dset, batch_size=batch_size,
                                             sampler=sampler.SubsetRandomSampler(range(num_train, 50000)))

    val_dset = dset.CIFAR10(root=data_dir, train=False, transform=trans, download=False)
    test_loader = DataLoader(val_dset, batch_size=1000, sampler=sampler.SubsetRandomSampler(range(0, 5000)))
    test_loader_repair = DataLoader(val_dset, batch_size=1000,
                                    sampler=sampler.SubsetRandomSampler(range(5000, 10000)))

    return train_loader, val_loader, test_loader, test_loader_repair


def tiny_imagenet_data_loader(data_dir: str = "../../resources/tiny-imagenet/tiny-imagenet-200",
                              num_train: int = 90000, batch_size: int = 100):

    """
    Cifar data loader. Creates train, val sets and two test sets: one to be used for repair and one for testing.

    Args:
        data_dir:
            Data to be loaded - downloaded if not found
        num_train:
            The number of training examples
        batch_size  :
        The batch size of the datasets
    """

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    trans = transforms.Compose([normalize])

    train_dset = TinyImageNetDSet(root=data_dir, train=True, transform=trans)
    train_loader = torch.utils.data.DataLoader(train_dset, batch_size=batch_size,
                                               sampler=sampler.SubsetRandomSampler(range(num_train)))
    val_loader = torch.utils.data.DataLoader(train_dset, batch_size=batch_size,
                                             sampler=sampler.SubsetRandomSampler(range(num_train, 100000)))

    val_dset = TinyImageNetDSet(root=data_dir, train=False, transform=trans)
    test_loader = DataLoader(val_dset, batch_size=100, sampler=sampler.SubsetRandomSampler(range(0, 1000)))
    test_loader_repair = DataLoader(val_dset, batch_size=100,
                                    sampler=sampler.SubsetRandomSampler(range(1000, 10000)))

    return train_loader, val_loader, test_loader, test_loader_repair


def mnist_back_image_data_loader(data_dir: str, num_train: int = 7000, batch_size: int = 100):

    """
    MNIST data loader. Creates train, val sets and two test sets: one to be used for repair and one for testing.
    Args:
        data_dir    : Data to be loaded - downloaded if not found
        num_train   : The number of training examples
        batch_size  : The batch size of the datasets
    """

    mnist_bck_train = torch.load(data_dir + "/train.pt")
    train_loader = DataLoader(mnist_bck_train, batch_size=batch_size,
                              sampler=sampler.SubsetRandomSampler(range(num_train)))
    val_loader = DataLoader(mnist_bck_train, batch_size=batch_size,
                            sampler=sampler.SubsetRandomSampler(range(num_train, 12000)))

    mnist_bck_test = torch.load(data_dir + "/test.pt")
    test_loader = DataLoader(mnist_bck_test, batch_size=batch_size,
                             sampler=sampler.SubsetRandomSampler(range(0, 25000)))
    test_loader_repair = DataLoader(mnist_bck_test, batch_size=batch_size,
                                    sampler=sampler.SubsetRandomSampler(range(25000, 50000)))

    return train_loader, val_loader, test_loader, test_loader_repair


# noinspection PyArgumentList
def write_stats(original_model: FFNN,
                repaired_model: FFNN,
                num_repaired: int,
                specific_set: np.array,
                specific_labels: np.array,
                general_set: np.array,
                general_labels: np.array,
                repair_set: np.array,
                repair_labels: np.array,
                test_loader: DataLoader,
                repair_class: int,
                file,
                additional_header: str = None):

    """
    Writes the statistics for all sets.

    Args;
        original_model:
            The model before repairing.
        repaired_model:
            The model after repair.
        num_repaired:
            The number of successfully repaired inputs.
        specific_set:
            The specific inputs.
        specific_labels:
            The labels corresponding to the specific set.
        general_set:
            The general inputs.
        general_labels:
            The labels corresponding to the general set.
        test_set:
            The test inputs.
        test_labels:
            The labels corresponding to the test set.
        repair_class:
            The class of the repair set.
        file:
            The output file.
        additional_header:
            This optional string is written to a seperate line below the header.
    """

    # Get the pre-repair predictions
    specific_pre_preds = original_model(torch.Tensor(specific_set)).cpu().detach().numpy()
    general_pre_preds = original_model(torch.Tensor(general_set)).cpu().detach().numpy()
    repair_pre_preds = original_model(torch.Tensor(repair_set)).cpu().detach().numpy()

    # Get the post-repair predictions
    specific_post_preds = repaired_model(torch.Tensor(specific_set)).cpu().detach().numpy()
    general_post_preds = repaired_model(torch.Tensor(general_set)).cpu().detach().numpy()
    repair_post_preds = repaired_model(torch.Tensor(repair_set)).cpu().detach().numpy()

    test_pre_preds = []
    test_post_preds = []
    test_labels = []

    if test_loader is not None:
        for batch_ndx, sample in enumerate(test_loader):

            test_post_preds.append(repaired_model(sample[0]).cpu().detach())
            test_pre_preds.append(original_model(sample[0]).cpu().detach())
            test_labels.append(sample[1])

    test_post_preds = np.concatenate(test_post_preds)
    test_pre_preds = np.concatenate(test_pre_preds)
    test_labels = np.concatenate(test_labels)

    # Write header
    file.write(f"Repair set from class {repair_class}\n")
    if additional_header is not None:
        file.write(additional_header + "\n")
    file.write("\n")

    file.write(f"Successfully repaired: {num_repaired}/{specific_set.shape[0]} inputs\n\n")

    # Write stats
    write_stats_set("Specific", specific_pre_preds, specific_post_preds, specific_labels, file)
    write_stats_set("General", general_pre_preds, general_post_preds, general_labels, file)
    write_stats_set("Repair", repair_pre_preds, repair_post_preds, repair_labels, file)
    write_stats_set("Test", test_pre_preds, test_post_preds, test_labels, file)


def write_stats_set(name: str,
                    pre_preds: np.array,
                    post_preds: np.array,
                    labels: np.array,
                    file):

    """
    Writes stats corresponding to the provided result arrays to file.

    Args:
        name:
            The name of the set.
        pre_preds:
            Predictions before retraining.
        post_preds:
            Predictions after retraining.
        labels:
            The labels corresponding to the previous sets.
        file:
            The output file.
    """

    num = len(labels)

    misclassifications_pre = (np.argmax(pre_preds, axis=1) != labels).sum()
    misclassifications_post = (np.argmax(post_preds, axis=1) != labels).sum()

    margin_pre = (pre_preds[np.arange(num), labels] - pre_preds.max(axis=1)).sum()
    margin_post = (post_preds[np.arange(num), labels] - post_preds.max(axis=1)).sum()

    acc_pre = 100 * (num - misclassifications_pre)/num
    acc_post = 100 * (num - misclassifications_post)/num

    diff = np.abs(pre_preds - post_preds)
    max_diff = diff.max()
    avg_diff = diff.sum()/np.prod(diff.shape)
    median_diff = np.median(diff.reshape(-1))

    margin_nodes = pre_preds.argmax(axis=1)
    margin_diff = np.abs(pre_preds[np.arange(num), margin_nodes] - post_preds[np.arange(num), margin_nodes])
    margin_max = margin_diff.max()
    margin_avg = margin_diff.sum()/np.prod(margin_diff.shape)
    margin_median = np.median(margin_diff.reshape(-1))

    file.write(f"{name} set (size: {num}):\n")
    file.write(f"Accuracy pre-repair: {acc_pre}%, post-repair: {acc_post}%\n")
    file.write(f"Margins pre-repair: {margin_pre:.2f}, post-repair: {margin_post:.2f}\n")
    file.write(f"Output change: largest: {max_diff:.4f}, average: {avg_diff:.4f}, median: {median_diff:.4f} \n")
    file.write(f"Margin node change: largest: {margin_max:.4f}, average: {margin_avg:.4f}, "
               f"median: {margin_median:.4f} \n\n")
