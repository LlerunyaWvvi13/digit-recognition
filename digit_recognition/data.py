"""Загрузка MNIST и разделение на обучающую и валидационную части."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def _eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))]
    )


def _train_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=8),
            transforms.ToTensor(),
            transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
        ]
    )


def build_loaders(
    data_dir: str | Path,
    batch_size: int = 128,
    val_size: int = 5000,
    seed: int = 42,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Возвращает загрузчики train, val и test.

    Аугментация применяется только к обучающей части. Тестовая часть не участвует
    в выборе модели: по ней считается итоговая метрика лучшей эпохи.
    """
    root = str(data_dir)
    train_full = datasets.MNIST(root, train=True, download=True, transform=_train_transform())
    val_full = datasets.MNIST(root, train=True, download=True, transform=_eval_transform())
    test_set = datasets.MNIST(root, train=False, download=True, transform=_eval_transform())

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(train_full), generator=generator).tolist()
    val_indices, train_indices = indices[:val_size], indices[val_size:]

    options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(
        Subset(train_full, train_indices), shuffle=True, drop_last=True, **options
    )
    val_loader = DataLoader(Subset(val_full, val_indices), shuffle=False, **options)
    test_loader = DataLoader(test_set, shuffle=False, **options)
    return train_loader, val_loader, test_loader
