"""Архитектура сверточной сети для изображений 1×28×28."""

from __future__ import annotations

from torch import Tensor, nn


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class DigitCNN(nn.Module):
    """Компактная CNN: возвращает логиты для классов 0–9."""

    def __init__(self, dropout: float = 0.25) -> None:
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(1, 32),
            _conv_block(32, 32),
            nn.MaxPool2d(2),  # 28×28 → 14×14
            _conv_block(32, 64),
            _conv_block(64, 64),
            nn.MaxPool2d(2),  # 14×14 → 7×7
            _conv_block(64, 96),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(96, 10),
        )

    def forward(self, batch: Tensor) -> Tensor:
        return self.classifier(self.features(batch))
