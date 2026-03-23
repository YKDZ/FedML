from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


@dataclass
class ShieldFLDataAssets:
    trainset: torch.utils.data.Dataset
    testset: torch.utils.data.Dataset
    client_indices: List[List[int]]
    val_loader: DataLoader
    trust_loader: DataLoader
    test_loader: DataLoader
    val_images: torch.Tensor
    val_labels: torch.Tensor


def _seeded_dataloader(dataset, batch_size, shuffle, seed, num_workers=0):
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
    )


def _load_cifar10(data_path: str):
    root_path = Path(data_path).expanduser().resolve()
    extracted_dir = root_path / "cifar-10-batches-py"
    archive_path = root_path / "cifar-10-python.tar.gz"
    should_download = not extracted_dir.exists() and not archive_path.exists()
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    transform_train = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    server_val_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    trainset = datasets.CIFAR10(
        root=str(root_path), train=True, download=should_download, transform=transform_train
    )
    testset = datasets.CIFAR10(
        root=str(root_path), train=False, download=should_download, transform=transform_test
    )
    server_val_base_dataset = datasets.CIFAR10(
        root=str(root_path), train=True, download=False, transform=server_val_transform
    )
    return trainset, testset, server_val_base_dataset


def _to_abs_size(value, total_size):
    value = float(value)
    if value <= 0:
        return 0
    if value < 1:
        return int(total_size * value)
    return int(value)


def _split_noniid(dataset, num_clients, alpha, seed):
    targets = np.array(dataset.targets)
    num_classes = len(np.unique(targets))
    client_idcs = [[] for _ in range(num_clients)]
    rng = np.random.default_rng(int(seed))

    for class_id in range(num_classes):
        class_indices = np.where(targets == class_id)[0]
        rng.shuffle(class_indices)
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        proportions = np.array(
            [
                p * (len(idx_j) < len(dataset) / num_clients)
                for p, idx_j in zip(proportions, client_idcs)
            ]
        )
        proportions = proportions / proportions.sum()
        split_points = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
        split_indices = np.split(class_indices, split_points)
        for client_id in range(num_clients):
            client_idcs[client_id].extend(split_indices[client_id].tolist())
    return client_idcs


def _loader_to_tensors(loader):
    image_batches = []
    label_batches = []
    for images, labels in loader:
        image_batches.append(images)
        label_batches.append(labels)
    return torch.cat(image_batches, dim=0), torch.cat(label_batches, dim=0)


class _IndexedDatasetView:
    def __init__(self, base_dataset, indices):
        self.base_dataset = base_dataset
        self.base_indices = list(indices)
        self.targets = [base_dataset.targets[i] for i in self.base_indices]

    def __len__(self):
        return len(self.base_indices)


def load_shieldfl_data(args) -> Tuple[list, ShieldFLDataAssets]:
    dataset_name = str(getattr(args, "dataset", "cifar10")).upper()
    if dataset_name != "CIFAR10":
        raise ValueError(f"Phase 1 currently only supports CIFAR10, got {dataset_name}")

    data_path = getattr(args, "data_cache_dir", "./data")
    seed = int(getattr(args, "random_seed", 0))
    batch_size = int(getattr(args, "batch_size", 32))
    num_workers = int(getattr(args, "num_workers", 0))
    client_num = int(getattr(args, "client_num_in_total", 3))
    alpha = float(getattr(args, "partition_alpha", 0.5))
    val_size = _to_abs_size(getattr(args, "server_val_size", 100), 50000)
    trust_size = _to_abs_size(getattr(args, "server_trust_size", 100), 50000)
    client_pool_max_size = int(getattr(args, "client_pool_max_size", 0) or 0)
    max_samples_per_client = int(getattr(args, "max_samples_per_client", 0) or 0)
    test_subset_size = int(getattr(args, "test_subset_size", 0) or 0)

    trainset, testset, server_val_base_dataset = _load_cifar10(data_path)
    shuffled_train_indices = np.random.default_rng(seed).permutation(len(trainset)).tolist()

    server_val_indices = shuffled_train_indices[:val_size]
    server_trust_indices = shuffled_train_indices[val_size : val_size + trust_size]
    client_pool_indices = shuffled_train_indices[val_size + trust_size :]

    if client_pool_max_size > 0:
        client_pool_indices = client_pool_indices[:client_pool_max_size]

    val_subset = Subset(server_val_base_dataset, server_val_indices)
    trust_subset = Subset(trainset, server_trust_indices)
    if test_subset_size > 0:
        test_indices = list(range(min(test_subset_size, len(testset))))
        test_subset = Subset(testset, test_indices)
    else:
        test_subset = testset

    val_loader = _seeded_dataloader(val_subset, batch_size, True, seed, num_workers)
    trust_loader = _seeded_dataloader(trust_subset, batch_size, True, seed + 1, num_workers)
    test_loader = _seeded_dataloader(test_subset, batch_size, False, seed + 2, num_workers)
    val_images, val_labels = _loader_to_tensors(val_loader)

    pool_view = _IndexedDatasetView(trainset, client_pool_indices)
    client_indices_local = _split_noniid(pool_view, client_num, alpha, seed)
    client_indices = [
        [client_pool_indices[j] for j in local_indices]
        for local_indices in client_indices_local
    ]
    if max_samples_per_client > 0:
        client_indices = [indices[:max_samples_per_client] for indices in client_indices]

    server_val_set = set(server_val_indices)
    server_trust_set = set(server_trust_indices)
    client_union = set(idx for indices in client_indices for idx in indices)
    assert server_val_set.isdisjoint(client_union), "server_val_set overlaps with client_pool"
    assert server_trust_set.isdisjoint(client_union), "server_trust_set overlaps with client_pool"
    assert server_val_set.isdisjoint(server_trust_set), "server_val_set overlaps with server_trust_set"

    train_data_local_dict: Dict[int, DataLoader] = {}
    test_data_local_dict: Dict[int, DataLoader] = {}
    train_data_local_num_dict: Dict[int, int] = {}
    for client_id, indices in enumerate(client_indices):
        subset = Subset(trainset, indices)
        train_data_local_dict[client_id] = _seeded_dataloader(
            subset, batch_size, True, seed + client_id + 10, num_workers
        )
        test_data_local_dict[client_id] = test_loader
        train_data_local_num_dict[client_id] = len(indices)

    train_global = _seeded_dataloader(
        Subset(trainset, client_pool_indices), batch_size, False, seed + 3, num_workers
    )
    dataset = [
        len(client_pool_indices),
        len(test_subset),
        train_global,
        test_loader,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        10,
    ]
    assets = ShieldFLDataAssets(
        trainset=trainset,
        testset=testset,
        client_indices=client_indices,
        val_loader=val_loader,
        trust_loader=trust_loader,
        test_loader=test_loader,
        val_images=val_images,
        val_labels=val_labels,
    )
    return dataset, assets
