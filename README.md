# Распознавание цифр и чисел

Сверточная нейросеть на PyTorch: читает рукописные цифры и числа из нескольких цифр. Обучена на MNIST. Промт, по которому написан проект, лежит в [PROMPT.md](PROMPT.md).

## Результаты

Измерено на наборе, который не участвовал в обучении и выборе модели.

| Показатель | Значение |
|---|---|
| Точность на тестовой части MNIST | 99,63% (37 ошибок на 10 000) |
| Весь путь от файла до цифры (предобработка + сеть), MNIST | 99,54% |
| Числа из 4 цифр, собранные из тестовых цифр MNIST | 97,45% (2 000 строк) |
| Размер сети | 121 642 параметра, файл 0,5 МБ |
| Обучение | 12 эпох, около 10 минут на 2 ядрах CPU |

Подробности по эпохам, классам и матрица ошибок: [reports/metrics.json](reports/metrics.json).

## Установка

Нужен Python 3.10 или новее.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Распознавание

Обученная модель уже лежит в `checkpoints/digit_cnn.pt`.

```bash
python -m digit_recognition.predict photo.jpg
python -m digit_recognition.predict photo.jpg --json
```

Из своего кода:

```python
import cv2
from digit_recognition import DigitRecognizer

recognizer = DigitRecognizer("checkpoints/digit_cnn.pt")
result = recognizer.predict_number(cv2.imread("photo.jpg"))
print(result.text, result.confidences)
```

## Обучение и проверка

```bash
python -m digit_recognition.train --epochs 12
python -m digit_recognition.evaluate_numbers --samples 2000
python -m pytest
```

Команда обучения скачает MNIST в папку `data/` и перезапишет `checkpoints/digit_cnn.pt` и `reports/metrics.json`.

## Ограничения

- Модель обучена на отдельных рукописных цифрах. На печатных шрифтах, фото с бликами и сложном фоне точность будет ниже указанной: такие данные в тестах не измерялись.
- Цифры в числе не должны касаться друг друга: строка делится по разрывам между символами.
- Строка должна быть горизонтальной и на однородном фоне.
- Точка, запятая и знак минус не распознаются. Сеть знает только цифры 0–9.
