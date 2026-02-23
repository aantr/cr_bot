"""
training_dataset/
    ├── class1/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── class2/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    └── ...
"""

import os
import shutil
import random
from PIL import Image
import numpy as np
from pathlib import Path
import argparse
from torchvision import transforms
import torch
from torchvision.utils import save_image
import math


def create_directory_structure(base_path):
    """Создает базовую структуру директорий"""
    Path(base_path).mkdir(parents=True, exist_ok=True)

    # Основные папки
    folders = ["train", "val", "search_dataset"]
    for folder in folders:
        Path(os.path.join(base_path, folder)).mkdir(exist_ok=True)

    return True


def get_image_transforms():
    """Возвращает трансформации для аугментации"""
    return [
        # Оригинальное изображение
        transforms.Compose([]),
        # Повороты (исправленные параметры)
        transforms.Compose([transforms.RandomRotation(degrees=(0, 30))]),
        transforms.Compose([transforms.RandomRotation(degrees=(0, 15))]),
        transforms.Compose([transforms.RandomRotation(degrees=(0, 45))]),
        # Горизонтальное отражение
        transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=1.0),
            ]
        ),
        # Комбинации поворотов и отражений
        transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.RandomRotation(degrees=(0, 20))
            ]
        ),
        transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=(0, 10))
            ]
        ),
        # Цветовые аугментации
        transforms.Compose([transforms.ColorJitter(brightness=0.2, contrast=0.2)]),
        transforms.Compose([transforms.ColorJitter(brightness=0.3, saturation=0.3)]),
        transforms.Compose([transforms.ColorJitter(hue=0.25)]),
    ]


import random
from PIL import Image

def apply_augmentation(image_path, output_path, transform_func, index):
    """Применяет аугментацию к изображению и сохраняет его"""
    try:
        # Открываем изображение
        img = Image.open(image_path)
        
        # Сохраняем информацию о наличии альфа-канала
        original_mode = img.mode
        
        # Конвертируем в подходящий режим для трансформации
        if img.mode in ('RGBA', 'LA', 'P'):
            # Если есть альфа-канал или прозрачность, работаем с RGBA
            img = img.convert('RGBA')
        else:
            # Иначе работаем с RGB
            img = img.convert('RGB')

        # Применяем трансформацию
        transformed_img = transform_func(img)

        # Если изображение имеет альфа-канал, накладываем его на случайную заливку
        if hasattr(transformed_img, 'mode') and transformed_img.mode == 'RGBA':
            # Генерируем случайный цвет для фона
            random_color = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255)
            )
            
            # Создаем фон с случайным цветом
            background = Image.new('RGB', transformed_img.size, random_color)
            
            # Накладываем изображение с альфа-каналом на фон
            background.paste(transformed_img, mask=transformed_img.split()[-1])  # Используем альфа-канал как маску
            transformed_img = background
        elif isinstance(transformed_img, torch.Tensor) and transformed_img.shape[0] == 4:
            # Если это тензор с 4 каналами (включая альфа)
            # Здесь можно добавить обработку тензора, но для простоты оставим как есть
            pass

        # Формируем имя файла
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        ext = os.path.splitext(image_path)[1]
        new_filename = f"{base_name}_aug_{index}{ext}"
        new_path = os.path.join(output_path, new_filename)

        # Сохраняем изображение
        if isinstance(transformed_img, torch.Tensor):
            save_image(transformed_img, new_path)
        else:
            # Если изображение RGBA и формат не поддерживает прозрачность, сохраняем как PNG
            if (hasattr(transformed_img, 'mode') and 
                transformed_img.mode == 'RGBA' and 
                ext.lower() not in ['.png', '.tiff', '.tif', '.webp']):
                new_filename = f"{base_name}_aug_{index}.png"
                new_path = os.path.join(output_path, new_filename)
            
            transformed_img.save(new_path)

        return new_path
    except Exception as e:
        print(f"Ошибка при аугментации {image_path}: {e}")
        return None




def process_single_image_as_class(
    image_path,
    class_name,
    output_train_path,
    output_val_path,
    augment_count=8,
    val_ratio=0.2,
):
    """
    Обрабатывает одно изображение как отдельный класс с аугментациями

    Args:
        image_path: путь к исходному изображению
        class_name: имя класса (обычно имя файла без расширения)
        output_train_path: путь для тренировочных данных
        output_val_path: путь для валидационных данных
        augment_count: количество аугментаций
        val_ratio: доля валидационных данных
    """

    # Создаем папки для класса
    train_class_path = os.path.join(output_train_path, class_name)
    val_class_path = os.path.join(output_val_path, class_name)

    Path(train_class_path).mkdir(parents=True, exist_ok=True)
    Path(val_class_path).mkdir(parents=True, exist_ok=True)

    # Получаем трансформации
    transforms_list = get_image_transforms()

    # Ограничиваем количество аугментаций
    transforms_list = transforms_list[: min(len(transforms_list), augment_count)]

    # Применяем аугментации
    augmented_images = []

    # Копируем оригинальное изображение
    original_filename = os.path.basename(image_path)
    original_dst = os.path.join(train_class_path, original_filename)
    shutil.copy2(image_path, original_dst)
    augmented_images.append(original_dst)

    # Применяем дополнительные аугментации
    for i, transform_func in enumerate(transforms_list[1:], 1):
        result_path = apply_augmentation(
            image_path, train_class_path, transform_func, i
        )
        if result_path:
            augmented_images.append(result_path)

    # Разделяем на train и val
    total_images = len(augmented_images)
    val_count = max(1, int(total_images * val_ratio)) if total_images > 1 else 0
    train_count = total_images - val_count

    # Перемещаем часть изображений в val
    if val_count > 0:
        val_images = random.sample(augmented_images, val_count)
        for img_path in val_images:
            filename = os.path.basename(img_path)
            new_path = os.path.join(val_class_path, filename)
            try:
                shutil.move(img_path, new_path)
            except Exception as e:
                print(f"Ошибка перемещения {img_path} в val: {e}")

    return len(augmented_images)


def prepare_search_dataset(source_folder, search_output_path, max_images=None):
    """
    Подготавливает датасет для поиска (простая папка с изображениями)

    Args:
        source_folder: путь к исходным изображениям
        search_output_path: путь для сохранения датасета поиска
        max_images: максимальное количество изображений (None для всех)
    """

    Path(search_output_path).mkdir(parents=True, exist_ok=True)

    # Получаем все изображения
    image_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".gif")
    all_images = []

    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if file.lower().endswith(image_extensions):
                all_images.append(os.path.join(root, file))

    if max_images and len(all_images) > max_images:
        all_images = random.sample(all_images, max_images)

    print(f"Копирование {len(all_images)} изображений для датасета поиска...")

    # Копируем изображения в папку поиска
    copied_count = 0
    for img_path in all_images:
        filename = os.path.basename(img_path)
        dst_path = os.path.join(search_output_path, filename)
        try:
            shutil.copy2(img_path, dst_path)
            copied_count += 1
        except Exception as e:
            print(f"Ошибка копирования {img_path}: {e}")

    print(f"Скопировано {copied_count} изображений для поиска")
    return copied_count


def validate_images(folder_path):
    """Проверяет изображения на корректность"""
    valid_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
    corrupted_files = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(valid_extensions):
                file_path = os.path.join(root, file)
                try:
                    with Image.open(file_path) as img:
                        img.verify()
                except Exception:
                    corrupted_files.append(file_path)
                    # Удаляем поврежденный файл
                    try:
                        os.remove(file_path)
                    except:
                        pass

    if corrupted_files:
        print(f"Найдено и удалено {len(corrupted_files)} поврежденных файлов")

    return len(corrupted_files)


def main():
    parser = argparse.ArgumentParser(
        description="Генерация датасета: каждый файл как отдельный класс с аугментациями"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Путь к исходной папке с изображениями",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./dataset",
        help="Путь для сохранения сгенерированного датасета",
    )
    parser.add_argument(
        "--augment-count",
        type=int,
        default=10,
        help="Количество аугментаций на изображение (по умолчанию 10)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Доля валидационных данных (0.0 - 1.0)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Максимальное количество изображений для обработки",
    )
    parser.add_argument(
        "--max-search-images",
        type=int,
        default=None,
        help="Максимальное количество изображений для поиска",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("ГЕНЕРАЦИЯ ДАТАСЕТА: КАЖДОЕ ИЗОБРАЖЕНИЕ КАК ОТДЕЛЬНЫЙ КЛАСС")
    print("=" * 70)

    source_folder = args.source
    output_base_path = args.output

    if not os.path.exists(source_folder):
        raise ValueError(f"Исходная папка не существует: {source_folder}")

    # Создаем структуру директорий
    print("Создание структуры директорий...")
    create_directory_structure(output_base_path)

    # Получаем все изображения
    image_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".gif")
    all_images = []

    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if file.lower().endswith(image_extensions):
                all_images.append(os.path.join(root, file))

    if args.max_images and len(all_images) > args.max_images:
        all_images = random.sample(all_images, args.max_images)

    print(f"Найдено {len(all_images)} изображений для обработки")

    if len(all_images) == 0:
        raise ValueError("Не найдено изображений в исходной папке")

    # Пути для train и val
    train_output_path = os.path.join(output_base_path, "train")
    val_output_path = os.path.join(output_base_path, "val")

    # Обрабатываем каждое изображение как отдельный класс
    processed_count = 0
    total_augmented = 0

    print(f"\nОбработка изображений как отдельных классов...")
    print(f"Аугментаций на изображение: {args.augment_count}")
    print(f"Доля валидационных данных: {args.val_ratio}")

    for img_path in all_images:
        # Создаем имя класса из имени файла
        filename_without_ext = os.path.splitext(os.path.basename(img_path))[0]
        # Очищаем имя от недопустимых символов
        class_name = "".join(
            c for c in filename_without_ext if c.isalnum() or c in (" ", "-", "_")
        ).strip()
        class_name = class_name.replace(" ", "_")

        if not class_name:
            class_name = f"class_{processed_count}"

        print(f"Обработка {os.path.basename(img_path)} -> класс '{class_name}'")

        try:
            augmented_count = process_single_image_as_class(
                image_path=img_path,
                class_name=class_name,
                output_train_path=train_output_path,
                output_val_path=val_output_path,
                augment_count=args.augment_count,
                val_ratio=args.val_ratio,
            )
            processed_count += 1
            total_augmented += augmented_count

        except Exception as e:
            print(f"Ошибка при обработке {img_path}: {e}")
            continue

    print(f"\nОбработано {processed_count} изображений")
    print(f"Создано {total_augmented} аугментированных изображений")

    # Подготавливаем датасет для поиска
    search_dataset_path = os.path.join(output_base_path, "search_dataset")
    print(f"\nСоздание датасета для поиска в: {search_dataset_path}")

    try:
        num_search_images = prepare_search_dataset(
            source_folder=source_folder,
            search_output_path=search_dataset_path,
            max_images=args.max_search_images,
        )
        print(f"Создан датасет для поиска с {num_search_images} изображениями")
    except Exception as e:
        print(f"Ошибка при создании датасета для поиска: {e}")

    # Проверяем изображения на корректность
    print("\nПроверка изображений на корректность...")
    validate_images(train_output_path)
    validate_images(val_output_path)
    validate_images(search_dataset_path)

    # Подсчитываем количество классов
    train_classes = (
        len(
            [
                d
                for d in os.listdir(train_output_path)
                if os.path.isdir(os.path.join(train_output_path, d))
            ]
        )
        if os.path.exists(train_output_path)
        else 0
    )
    val_classes = (
        len(
            [
                d
                for d in os.listdir(val_output_path)
                if os.path.isdir(os.path.join(val_output_path, d))
            ]
        )
        if os.path.exists(val_output_path)
        else 0
    )

    # Вывод информации о датасете
    print("\n" + "=" * 70)
    print("ИНФОРМАЦИЯ О СГЕНЕРИРОВАННОМ ДАТАСЕТЕ")
    print("=" * 70)
    print(f"Базовая папка: {output_base_path}")
    print(f"Структура:")
    print(f"  ├── train/           # Тренировочные данные ({train_classes} классов)")
    print(f"  ├── val/             # Валидационные данные ({val_classes} классов)")
    print(
        f"  └── search_dataset/  # Датасет для поиска ({num_search_images} изображений)"
    )
    print(f"\nОбработано изображений: {processed_count}")
    print(f"Аугментированных изображений: {total_augmented}")
    print(f"\nДатасет готов к использованию!")
    print(f"Используйте путь к search_dataset для переменной dataset_folder")


if __name__ == "__main__":
    main()
