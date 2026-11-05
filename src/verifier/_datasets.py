__all__ = [
    "load_torch_dataset",
    "get_num_classes",
]

import os

import torch
from torch import Tensor


def load_torch_dataset(
    dataset: str,
    saved_dir_path: str,
    normalize: bool = False,
    means: Tensor | None = None,
    stds: Tensor | None = None,
    verbose: bool = False,
) -> torch.utils.data.DataLoader:
    """Load a dataset from torchvision."""

    os.makedirs(saved_dir_path, exist_ok=True)

    from torchvision import transforms

    transform = transforms.Compose([transforms.ToTensor()])

    if normalize:
        if means is not None and stds is not None:
            means = tuple(means.tolist())
            stds = tuple(stds.tolist())
        elif dataset == "cifar10":
            means = (0.4914, 0.4822, 0.4465)
            stds = (0.2023, 0.1994, 0.2010)
        elif dataset == "mnist":
            means = (0.0,)
            stds = (1.0,)
        else:
            raise ValueError(
                f"Normalization parameters are not provided for dataset {dataset}."
            )
        transform.transforms.append(transforms.Normalize(means, stds))
        if verbose:
            print(f"Set normalization with means: {means} and stds: {stds}")
    else:
        if verbose:
            print("Normalization is not applied to the dataset.")

    kwargs = {"train": False, "download": True, "transform": transform}

    if verbose:
        print(f"Loading dataset {dataset} from {saved_dir_path}...")

    if dataset == "mnist":
        from torchvision.datasets import MNIST

        test_set = MNIST(root=saved_dir_path, **kwargs)
    elif dataset == "cifar10":
        from torchvision.datasets import CIFAR10

        test_set = CIFAR10(root=saved_dir_path + "/CIFAR10", **kwargs)
    elif dataset == "fashion_mnist":
        from torchvision.datasets import FashionMNIST

        test_set = FashionMNIST(root=saved_dir_path, **kwargs)
    elif dataset == "emnist":
        from torchvision.datasets import EMNIST

        test_set = EMNIST(root=saved_dir_path, split="letters", **kwargs)
    else:
        raise NotImplementedError(f"Dataset {dataset} is not supported.")

    from torch.utils.data import DataLoader

    data_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    return data_loader


def get_num_classes(
    dataset: str,
) -> int:
    if dataset in {"mnist", "cifar10", "fashion_mnist"}:
        return 10
    elif dataset == "emnist":
        return 26

    raise NotImplementedError(f"Dataset {dataset} is not supported.")
