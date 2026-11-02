"""Обучение модели: python -m digit_recognition.train --epochs 12"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import MNIST_MEAN, MNIST_STD, build_loaders
from .model import DigitCNN


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.inference_mode()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[float, np.ndarray]:
    """Считает точность и матрицу ошибок 10×10 (строки — истинный класс)."""
    model.eval()
    confusion = np.zeros((10, 10), dtype=np.int64)
    for images, labels in loader:
        predictions = model(images.to(device)).argmax(dim=1).cpu().numpy()
        np.add.at(confusion, (labels.numpy(), predictions), 1)
    return float(np.trace(confusion) / confusion.sum()), confusion


def save_checkpoint(path: Path, model: nn.Module, epoch: int, val_accuracy: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "epoch": epoch,
            "val_accuracy": val_accuracy,
            "mean": MNIST_MEAN,
            "std": MNIST_STD,
        },
        path,
    )


def train(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = build_loaders(
        args.data_dir, args.batch_size, seed=args.seed, num_workers=args.workers
    )

    model = DigitCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_loader)
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    checkpoint = Path(args.checkpoint)
    best_accuracy = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()

        val_accuracy, _ = evaluate(model, val_loader, device)
        record = {
            "epoch": epoch,
            "train_loss": round(running_loss / len(train_loader), 4),
            "val_accuracy": round(val_accuracy, 5),
            "seconds": round(time.perf_counter() - started, 1),
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            save_checkpoint(checkpoint, model, epoch, val_accuracy)

    best = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(best["state_dict"])
    test_accuracy, confusion = evaluate(model, test_loader, device)
    per_class = confusion.diagonal() / confusion.sum(axis=1)
    metrics = {
        "best_epoch": best["epoch"],
        "val_accuracy": round(best_accuracy, 5),
        "test_accuracy": round(test_accuracy, 5),
        "test_errors": int(confusion.sum() - np.trace(confusion)),
        "test_per_class_accuracy": {str(k): round(float(v), 4) for k, v in enumerate(per_class)},
        "confusion_matrix": confusion.tolist(),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "epochs": args.epochs,
        "seed": args.seed,
        "history": history,
    }
    metrics_path = Path(args.metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Точность на тесте: {test_accuracy:.4%}; модель: {checkpoint}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Обучение распознаванию цифр на MNIST")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--checkpoint", default="checkpoints/digit_cnn.pt")
    parser.add_argument("--metrics", default="reports/metrics.json")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
