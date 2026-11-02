"""Распознавание файла: python -m digit_recognition.predict photo.jpg"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from .model import DigitCNN
from .preprocess import extract_digits


@dataclass(frozen=True)
class NumberPrediction:
    text: str
    digits: list[int]
    confidences: list[float]

    @property
    def min_confidence(self) -> float:
        return min(self.confidences, default=0.0)


class DigitRecognizer:
    """Загружает обученную модель и распознает цифры и числа."""

    def __init__(self, checkpoint: str | Path, device: str = "cpu") -> None:
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(
                f"Нет файла модели {path}. Обучите ее: python -m digit_recognition.train"
            )
        self.device = torch.device(device)
        state = torch.load(path, map_location=self.device, weights_only=True)
        self.mean = float(state["mean"])
        self.std = float(state["std"])
        self.model = DigitCNN().to(self.device)
        self.model.load_state_dict(state["state_dict"])
        self.model.eval()

    @torch.inference_mode()
    def classify(self, glyphs: list[np.ndarray]) -> tuple[list[int], list[float]]:
        """Классифицирует символы формата MNIST: uint8, 28×28, белое на черном."""
        if not glyphs:
            return [], []
        batch = torch.from_numpy(np.stack(glyphs)).float().div(255).unsqueeze(1)
        batch = batch.sub(self.mean).div(self.std).to(self.device)
        probabilities = self.model(batch).softmax(dim=1)
        confidences, digits = probabilities.max(dim=1)
        return digits.tolist(), [round(value, 4) for value in confidences.tolist()]

    def predict_number(self, image: np.ndarray) -> NumberPrediction:
        """Распознает одну цифру или строку из нескольких цифр."""
        glyphs, _ = extract_digits(image)
        digits, confidences = self.classify(glyphs)
        return NumberPrediction("".join(map(str, digits)), digits, confidences)


def main() -> int:
    parser = argparse.ArgumentParser(description="Распознавание цифр и чисел на изображении")
    parser.add_argument("image", type=Path, help="путь к изображению")
    parser.add_argument("--checkpoint", default="checkpoints/digit_cnn.pt")
    parser.add_argument("--json", action="store_true", help="вывести результат в JSON")
    args = parser.parse_args()

    image = cv2.imread(str(args.image), cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"Не удалось прочитать изображение: {args.image}", file=sys.stderr)
        return 2
    try:
        result = DigitRecognizer(args.checkpoint).predict_number(image)
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 2
    if not result.digits:
        print("Цифры на изображении не найдены", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(asdict(result), ensure_ascii=False))
    else:
        print(f"{result.text} (минимальная уверенность {result.min_confidence:.1%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
