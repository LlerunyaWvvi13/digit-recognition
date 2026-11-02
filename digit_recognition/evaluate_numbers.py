"""Проверка распознавания чисел на строках, собранных из тестовой части MNIST."""

from __future__ import annotations

import argparse
import json

import numpy as np
from torchvision import datasets

from .predict import DigitRecognizer


def positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("значение должно быть не меньше 1")
    return value


def compose_number(digits: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    """Склеивает цифры в строку со случайным зазором, не допуская касаний."""
    gap_range = (4, 12)
    parts = [np.zeros((48, 10), dtype=np.uint8)]
    for digit in digits:
        cell = np.zeros((48, 28), dtype=np.uint8)
        top = int(rng.integers(6, 14))
        cell[top : top + 28] = digit
        parts.append(cell)
        parts.append(np.zeros((48, int(rng.integers(*gap_range))), dtype=np.uint8))
    parts.append(np.zeros((48, 10), dtype=np.uint8))
    return np.hstack(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--checkpoint", default="checkpoints/digit_cnn.pt")
    parser.add_argument("--samples", type=positive_int, default=2000)
    parser.add_argument("--length", type=positive_int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    test_set = datasets.MNIST(args.data_dir, train=False, download=True)
    images = test_set.data.numpy()
    labels = test_set.targets.numpy()
    recognizer = DigitRecognizer(args.checkpoint)
    rng = np.random.default_rng(args.seed)

    correct = 0
    for _ in range(args.samples):
        indices = rng.integers(0, len(images), size=args.length)
        line = compose_number([images[index] for index in indices], rng)
        expected = "".join(str(labels[index]) for index in indices)
        correct += recognizer.predict_number(line).text == expected

    print(
        json.dumps(
            {
                "samples": args.samples,
                "length": args.length,
                "seed": args.seed,
                "number_accuracy": round(correct / args.samples, 4),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
