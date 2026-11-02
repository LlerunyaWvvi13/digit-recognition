from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from digit_recognition.model import DigitCNN
from digit_recognition.predict import DigitRecognizer
from digit_recognition.preprocess import binarize, extract_digits, normalize_glyph
from digit_recognition.train import save_checkpoint

CHECKPOINT = Path("checkpoints/digit_cnn.pt")


def draw_number(text: str, inverted: bool = False) -> np.ndarray:
    image = np.full((80, 40 + 45 * len(text)), 255, dtype=np.uint8)
    cv2.putText(image, text, (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 2, 0, 5, cv2.LINE_AA)
    return cv2.bitwise_not(image) if inverted else image


def test_model_returns_ten_logits() -> None:
    logits = DigitCNN()(torch.zeros(4, 1, 28, 28))
    assert logits.shape == (4, 10)


@pytest.mark.parametrize("inverted", [False, True])
def test_binarize_makes_digits_white_on_any_background(inverted: bool) -> None:
    mask = binarize(draw_number("7", inverted))
    assert mask[0, 0] == 0
    assert mask.max() == 255


@pytest.mark.parametrize("inverted", [False, True])
def test_extract_digits_splits_line_left_to_right(inverted: bool) -> None:
    glyphs, boxes = extract_digits(draw_number("2024", inverted))
    assert len(glyphs) == 4
    assert [box.x for box in boxes] == sorted(box.x for box in boxes)
    assert all(glyph.shape == (28, 28) for glyph in glyphs)


def test_small_noise_is_ignored() -> None:
    image = draw_number("31")
    image[3, 3] = 0
    image[75, 120] = 0
    glyphs, _ = extract_digits(image)
    assert len(glyphs) == 2


def test_blank_image_has_no_digits() -> None:
    glyphs, boxes = extract_digits(np.full((50, 50), 255, dtype=np.uint8))
    assert glyphs == []
    assert boxes == []


def test_normalize_glyph_centers_mass() -> None:
    mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(mask, (5, 5), (25, 60), 255, -1)
    canvas = normalize_glyph(mask)
    moments = cv2.moments(canvas)
    assert moments["m10"] / moments["m00"] == pytest.approx(14, abs=1)
    assert moments["m01"] / moments["m00"] == pytest.approx(14, abs=1)


def test_recognizer_restores_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "model.pt"
    model = DigitCNN()
    save_checkpoint(path, model, epoch=1, val_accuracy=0.5)
    recognizer = DigitRecognizer(path)
    digits, confidences = recognizer.classify([np.zeros((28, 28), dtype=np.uint8)])
    assert len(digits) == 1
    assert 0 <= confidences[0] <= 1


def test_missing_checkpoint_has_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Обучите"):
        DigitRecognizer(tmp_path / "missing.pt")


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="нет обученной модели")
def test_trained_model_reads_mnist_digits() -> None:
    from torchvision import datasets

    test_set = datasets.MNIST("data", train=False, download=True)
    recognizer = DigitRecognizer(CHECKPOINT)
    glyphs = [normalize_glyph(image.numpy()) for image in test_set.data[:500]]
    predictions, _ = recognizer.classify(glyphs)
    accuracy = np.mean(np.array(predictions) == test_set.targets[:500].numpy())
    assert accuracy > 0.97
