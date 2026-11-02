"""Приведение обычных изображений к формату MNIST.

Сеть обучена на белых цифрах на черном фоне, вписанных в квадрат 20×20 и
отцентрованных по центру масс внутри 28×28. Здесь то же самое делается для фото
и сканов, а строка цифр делится на отдельные символы.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

GLYPH_BOX = 20
CANVAS = 28
MIN_HEIGHT_RATIO = 0.3
MIN_AREA_RATIO = 0.0002


@dataclass(frozen=True)
class Box:
    x: int
    y: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    def union(self, other: "Box") -> "Box":
        x = min(self.x, other.x)
        y = min(self.y, other.y)
        return Box(
            x,
            y,
            max(self.right, other.right) - x,
            max(self.bottom, other.bottom) - y,
        )


def _flatten_transparency(image: np.ndarray) -> np.ndarray:
    """Накладывает BGRA на белый фон: цвет прозрачных пикселей ничего не значит."""
    if image.dtype.kind in "iu":
        depth = float(np.iinfo(image.dtype).max)
    else:
        depth = 255.0 if float(image.max()) > 1 else 1.0
    alpha = image[:, :, 3:4].astype(np.float32) / depth
    color = image[:, :, :3].astype(np.float32) / depth
    flattened = color * alpha + (1 - alpha)
    return cv2.cvtColor((flattened * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Переводит BGR/BGRA/серое изображение в uint8 с одним каналом."""
    if image is None or image.size == 0:
        raise ValueError("Пустое изображение")
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 3 and image.shape[2] == 4:
        gray = _flatten_transparency(image)
    else:
        raise ValueError(f"Неподдерживаемая форма изображения: {image.shape}")
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return gray


def binarize(image: np.ndarray) -> np.ndarray:
    """Возвращает маску, где цифры белые, а фон черный.

    Полярность определяется по рамке кадра: край изображения почти всегда фон.
    """
    gray = to_grayscale(image)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    border = np.concatenate([mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1]])
    if np.mean(border) > 127:
        mask = cv2.bitwise_not(mask)
    return mask


def foreground(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Возвращает цифры в оттенках серого: белые на черном, как в MNIST.

    Маска нужна только для поиска символов. Сеть точнее работает с мягкими краями,
    поэтому для нее сохраняются исходные полутона в окрестности маски.
    """
    gray = to_grayscale(image).astype(np.float32)
    background_pixels = gray[mask == 0]
    if background_pixels.size == 0 or not np.any(mask):
        return mask
    background = float(np.median(background_pixels))
    ink = np.abs(gray - background)
    peak = float(np.percentile(ink[mask > 0], 95))
    if peak <= 0:
        return mask
    soft = np.clip(ink / peak, 0, 1) * 255
    nearby = cv2.dilate(mask, np.ones((3, 3), np.uint8))
    soft[nearby == 0] = 0
    return soft.astype(np.uint8)


def _merge_overlapping(boxes: list[Box]) -> list[Box]:
    """Склеивает части одного символа, например разорванную верхнюю черту пятерки."""
    merged: list[Box] = []
    for box in sorted(boxes, key=lambda item: item.x):
        if merged:
            previous = merged[-1]
            overlap = min(previous.right, box.right) - max(previous.x, box.x)
            if overlap > 0.5 * min(previous.width, box.width):
                merged[-1] = previous.union(box)
                continue
        merged.append(box)
    return merged


def find_digit_boxes(mask: np.ndarray) -> list[Box]:
    """Находит символы на маске и возвращает их слева направо."""
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_area = max(4.0, mask.size * MIN_AREA_RATIO)
    boxes = [
        Box(
            int(stats[label, cv2.CC_STAT_LEFT]),
            int(stats[label, cv2.CC_STAT_TOP]),
            int(stats[label, cv2.CC_STAT_WIDTH]),
            int(stats[label, cv2.CC_STAT_HEIGHT]),
        )
        for label in range(1, count)
        if stats[label, cv2.CC_STAT_AREA] >= min_area
    ]
    boxes = _merge_overlapping(boxes)
    if not boxes:
        return []
    tallest = max(box.height for box in boxes)
    return [box for box in boxes if box.height >= tallest * MIN_HEIGHT_RATIO]


def normalize_glyph(mask: np.ndarray, box: Box | None = None) -> np.ndarray:
    """Вписывает символ в 20×20 и центрирует по массе на холсте 28×28."""
    glyph = mask if box is None else mask[box.y : box.bottom, box.x : box.right]
    points = cv2.findNonZero(glyph)
    if points is None:
        return np.zeros((CANVAS, CANVAS), dtype=np.uint8)
    x, y, width, height = cv2.boundingRect(points)
    glyph = glyph[y : y + height, x : x + width]

    scale = GLYPH_BOX / max(width, height)
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    resized = cv2.resize(glyph, new_size, interpolation=cv2.INTER_AREA)

    canvas = np.zeros((CANVAS, CANVAS), dtype=np.uint8)
    left = (CANVAS - new_size[0]) // 2
    top = (CANVAS - new_size[1]) // 2
    canvas[top : top + new_size[1], left : left + new_size[0]] = resized

    moments = cv2.moments(canvas)
    if moments["m00"] > 0:
        shift_x = CANVAS / 2 - moments["m10"] / moments["m00"]
        shift_y = CANVAS / 2 - moments["m01"] / moments["m00"]
        matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        canvas = cv2.warpAffine(canvas, matrix, (CANVAS, CANVAS), flags=cv2.INTER_LINEAR)
    return canvas


def extract_digits(image: np.ndarray) -> tuple[list[np.ndarray], list[Box]]:
    """Делит изображение на символы и приводит каждый к формату MNIST."""
    mask = binarize(image)
    boxes = find_digit_boxes(mask)
    ink = foreground(image, mask)
    return [normalize_glyph(ink, box) for box in boxes], boxes
