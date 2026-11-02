"""Распознавание рукописных цифр и чисел на PyTorch."""

from .model import DigitCNN
from .predict import DigitRecognizer

__all__ = ["DigitCNN", "DigitRecognizer"]
