import os
import cv2
import numpy as np
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt

def analyze_dataset(dataset_path):
    """Анализ структуры датасета"""
    images_dir = Path(dataset_path) / 'images'
    
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    
    # Получение списка файлов
    image_files = list(images_dir.glob('*.png'))
    
    # Извлечение имен классов из имен файлов
    class_names = []
    file_mapping = {}
    
    for file_path in image_files:
        filename = file_path.stem  # без расширения
        # Формат: CardName_card-name.png
        if '_' in filename:
            card_name = filename.split('_')[0]
            class_names.append(card_name)
            if card_name not in file_mapping:
                file_mapping[card_name] = []
            file_mapping[card_name].append(file_path)
    
    print(f"Total images: {len(image_files)}")
    print(f"Number of classes: {len(set(class_names))}")
    print("\nClass distribution:")
    class_counts = Counter(class_names)
    for class_name, count in class_counts.most_common():
        print(f"  {class_name}: {count} images")
    
    return class_counts, file_mapping

# Анализ датасета
dataset_path = 'cards_dataset'
class_counts, file_mapping = analyze_dataset(dataset_path)
