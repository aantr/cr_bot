import numpy as np
from pathlib import Path
import yaml
import json
import cv2
import os
from typing import List, Dict, Tuple, Optional
import argparse


class YOLOAnnotationGenerator:
    """Генератор YOLO аннотаций из списка координат"""

    def __init__(self, classes: List[str] = None):
        """
        Инициализация генератора

        Args:
            classes: список названий классов
        """
        self.classes = classes or []
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

    def parse_coordinates(
        self,
        coord_list: List[List[float]],
        image_width: int = 0,
        image_height: int = 0,
        format_type: str = "auto",
    ) -> List[Dict]:
        """
        Парсинг списка координат в формат YOLO

        Args:
            coord_list: список координат в различных форматах
            image_width: ширина изображения
            image_height: высота изображения
            format_type: тип формата входных данных

        Returns:
            Список аннотаций в формате словаря
        """
        annotations = []

        for i, coords in enumerate(coord_list):
            # Преобразуем в numpy array если нужно
            if isinstance(coords, list):
                coords = np.array(coords)

            # Определяем формат автоматически
            if format_type == "auto":
                format_type = self._detect_format(coords, image_width, image_height)

            # Парсим в зависимости от формата
            if format_type == "yolo_normalized":
                annotation = self._parse_yolo_normalized(coords)
            elif format_type == "yolo_pixel":
                annotation = self._parse_yolo_pixel(coords, image_width, image_height)
            elif format_type == "coco":
                annotation = self._parse_coco(coords, image_width, image_height)
            elif format_type == "pascal_voc":
                annotation = self._parse_pascal_voc(coords, image_width, image_height)
            elif format_type == "xywh_normalized":
                annotation = self._parse_xywh_normalized(coords)
            elif format_type == "xyxy_normalized":
                annotation = self._parse_xyxy_normalized(coords)
            else:
                raise ValueError(f"Неизвестный формат: {format_type}")

            if annotation:
                annotations.append(annotation)

        return annotations

    def _detect_format(self, coords: np.ndarray, img_w: int, img_h: int) -> str:
        """
        Автоматическое определение формата координат
        """
        if len(coords) == 6:
            # Ваш формат: [x_center, y_center, width, height, confidence?, class?]
            return "yolo_normalized"
        elif len(coords) == 4:
            if np.all(coords <= 1.0):
                return "xywh_normalized"
            elif np.all(coords <= max(img_w, img_h)):
                return "coco"  # или yolo_pixel
        return "yolo_normalized"

    def _parse_yolo_normalized(self, coords: np.ndarray) -> Dict:
        """
        Парсинг YOLO формата (нормализованные координаты)
        Формат: [class_id, x_center, y_center, width, height, confidence]
        или: [x_center, y_center, width, height, confidence, class_id]
        """

        # Формат: [x_center, y_center, width, height, confidence, class_id]
        x_center = float(coords[0])
        y_center = float(coords[1])
        width = float(coords[2])
        height = float(coords[3])
        states = coords[4:-1]
        class_id = int(coords[-1])

        return {
            "class_id": class_id,
            "x_center": x_center,
            "y_center": y_center,
            "width": width,
            "height": height,
            "states": states,
        }

    def _parse_yolo_pixel(self, coords: np.ndarray, img_w: int, img_h: int) -> Dict:
        """Парсинг YOLO формата (пиксельные координаты)"""
        if len(coords) >= 5:
            class_id = int(coords[0])
            x_center_px = float(coords[1])
            y_center_px = float(coords[2])
            width_px = float(coords[3])
            height_px = float(coords[4])

            # Нормализуем
            x_center = x_center_px / img_w
            y_center = y_center_px / img_h
            width = width_px / img_w
            height = height_px / img_h

            return {
                "class_id": class_id,
                "x_center": x_center,
                "y_center": y_center,
                "width": width,
                "height": height,
                "confidence": 1.0,
            }
        return None

    def _parse_coco(self, coords: np.ndarray, img_w: int, img_h: int) -> Dict:
        """Парсинг COCO формата [x, y, width, height]"""
        if len(coords) >= 4:
            x = float(coords[0])
            y = float(coords[1])
            width = float(coords[2])
            height = float(coords[3])

            # Конвертируем в YOLO формат
            x_center = (x + width / 2) / img_w
            y_center = (y + height / 2) / img_h
            width_norm = width / img_w
            height_norm = height / img_h

            class_id = int(coords[4]) if len(coords) > 4 else 0

            return {
                "class_id": class_id,
                "x_center": x_center,
                "y_center": y_center,
                "width": width_norm,
                "height": height_norm,
                "confidence": float(coords[5]) if len(coords) > 5 else 1.0,
            }
        return None

    def _parse_pascal_voc(self, coords: np.ndarray, img_w: int, img_h: int) -> Dict:
        """Парсинг Pascal VOC формата [xmin, ymin, xmax, ymax]"""
        if len(coords) >= 4:
            xmin = float(coords[0])
            ymin = float(coords[1])
            xmax = float(coords[2])
            ymax = float(coords[3])

            # Конвертируем в YOLO формат
            x_center = (xmin + xmax) / 2 / img_w
            y_center = (ymin + ymax) / 2 / img_h
            width = (xmax - xmin) / img_w
            height = (ymax - ymin) / img_h

            class_id = int(coords[4]) if len(coords) > 4 else 0

            return {
                "class_id": class_id,
                "x_center": x_center,
                "y_center": y_center,
                "width": width,
                "height": height,
                "confidence": float(coords[5]) if len(coords) > 5 else 1.0,
            }
        return None

    def _parse_xywh_normalized(self, coords: np.ndarray) -> Dict:
        """Парсинг нормализованного формата [x_center, y_center, width, height]"""
        if len(coords) >= 4:
            return {
                "class_id": int(coords[4]) if len(coords) > 4 else 0,
                "x_center": float(coords[0]),
                "y_center": float(coords[1]),
                "width": float(coords[2]),
                "height": float(coords[3]),
                "confidence": float(coords[5]) if len(coords) > 5 else 1.0,
            }
        return None

    def _parse_xyxy_normalized(self, coords: np.ndarray) -> Dict:
        """Парсинг нормализованного формата [xmin, ymin, xmax, ymax]"""
        if len(coords) >= 4:
            xmin = float(coords[0])
            ymin = float(coords[1])
            xmax = float(coords[2])
            ymax = float(coords[3])

            x_center = (xmin + xmax) / 2
            y_center = (ymin + ymax) / 2
            width = xmax - xmin
            height = ymax - ymin

            return {
                "class_id": int(coords[4]) if len(coords) > 4 else 0,
                "x_center": x_center,
                "y_center": y_center,
                "width": width,
                "height": height,
                "confidence": float(coords[5]) if len(coords) > 5 else 1.0,
            }
        return None

    def create_yolo_annotation_file(
        self, annotations: List[Dict], output_path: str, include_states: bool = True
    ) -> str:
        """
        Создание файла аннотаций YOLO

        Args:
            annotations: список аннотаций
            output_path: путь для сохранения файла
            include_confidence: включать confidence score

        Returns:
            Путь к созданному файлу
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for ann in annotations:
                line = f"{ann['class_id']} {ann['x_center']:.6f} {ann['y_center']:.6f} "
                line += f"{ann['width']:.6f} {ann['height']:.6f}"

                if include_states:
                    states = list(map(int, ann["states"]))
                    while len(states) < 7:
                        states.append(0)
                    line += f" " + " ".join(list(map(str, states)))

                f.write(line + "\n")

        return str(output_path)

    def create_yaml_config(
        self,
        dataset_path: str,
        train_images: str = "images/train",
        val_images: str = "images/val",
        test_images: str = "images/test",
        nc: int = None,
        names: List[str] = None,
        config_name: str = "data.yaml",
    ) -> str:
        """
        Создание YAML конфигурационного файла для YOLO

        Args:
            dataset_path: путь к датасету
            train_images: путь к тренировочным изображениям
            val_images: путь к валидационным изображениям
            test_images: путь к тестовым изображениям
            nc: количество классов
            names: список названий классов
            config_name: имя конфигурационного файла

        Returns:
            Путь к созданному файлу
        """
        config = {
            "path": str(Path(dataset_path).absolute()),
            "train": train_images,
            "val": val_images,
            "test": test_images,
            "nc": nc or len(self.classes),
            "names": names or self.classes,
        }

        # Дополнительные параметры
        config.update(
            {
                "download": None,
                "license": "MIT",
                "date_created": "2024-01-01",
                "version": "1.0",
            }
        )

        config_path = Path(dataset_path) / config_name
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

        return str(config_path)

    def create_dataset_structure(
        self, base_path: str, create_subdirs: bool = True
    ) -> Dict[str, Path]:
        """
        Создание структуры каталогов для YOLO датасета

        Args:
            base_path: базовый путь датасета
            create_subdirs: создавать подкаталоги train/val/test

        Returns:
            Словарь путей к созданным каталогам
        """
        base = Path(base_path)

        dirs = {
            "base": base,
            "images": base / "images",
            "labels": base / "labels",
        }

        if create_subdirs:
            subsets = ["train", "val", "test"]
            for subset in subsets:
                dirs[f"images_{subset}"] = base / "images" / subset
                dirs[f"labels_{subset}"] = base / "labels" / subset

        # Создаем каталоги
        for dir_path in dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)

        return dirs

    def validate_annotations(
        self, annotations: List[Dict], image_width: int = None, image_height: int = None
    ) -> List[str]:
        """
        Валидация аннотаций

        Args:
            annotations: список аннотаций
            image_width: ширина изображения (для проверки)
            image_height: высота изображения (для проверки)

        Returns:
            Список ошибок
        """
        errors = []

        for i, ann in enumerate(annotations):
            # Проверяем обязательные поля
            required_fields = ["class_id", "x_center", "y_center", "width", "height"]
            for field in required_fields:
                if field not in ann:
                    errors.append(f"Аннотация {i}: отсутствует поле '{field}'")

            # Проверяем типы данных
            try:
                class_id = int(ann["class_id"])
                x_center = float(ann["x_center"])
                y_center = float(ann["y_center"])
                width = float(ann["width"])
                height = float(ann["height"])
            except (ValueError, TypeError):
                errors.append(f"Аннотация {i}: неверный формат чисел")
                continue

            # Проверяем диапазоны
            if class_id < 0:
                errors.append(f"Аннотация {i}: class_id отрицательный: {class_id}")

            if not (0 <= x_center <= 1):
                errors.append(f"Аннотация {i}: x_center вне диапазона 0-1: {x_center}")

            if not (0 <= y_center <= 1):
                errors.append(f"Аннотация {i}: y_center вне диапазона 0-1: {y_center}")

            if not (0 <= width <= 1):
                errors.append(f"Аннотация {i}: width вне диапазона 0-1: {width}")

            if not (0 <= height <= 1):
                errors.append(f"Аннотация {i}: height вне диапазона 0-1: {height}")

            # Проверяем confidence если есть
            if "confidence" in ann:
                try:
                    conf = float(ann["confidence"])
                    if not (0 <= conf <= 1):
                        errors.append(
                            f"Аннотация {i}: confidence вне диапазона 0-1: {conf}"
                        )
                except (ValueError, TypeError):
                    errors.append(f"Аннотация {i}: неверный формат confidence")

        return errors


def generate_from_katacr_format(data, output):
    """Пример генерации из формата данных"""
    # Ваши данные
    coord_list = data

    # Создаем генератор
    generator = YOLOAnnotationGenerator()

    # Парсим координаты
    # В вашем формате: [x_center, y_center, width, height, confidence, class_id]
    annotations = generator.parse_coordinates(
        coord_list, format_type="yolo_normalized"  # Указываем формат явно
    )

    # Валидируем
    errors = generator.validate_annotations(annotations)
    if errors:
        print("Ошибки валидации:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("Аннотации валидны!")

    # Создаем файл аннотаций
    output_file = output
    generator.create_yolo_annotation_file(annotations, output_file, include_states=True)

    print(f"Файл аннотаций создан: {output_file}")
    print("\nСодержимое:")
    with open(output_file, "r") as f:
        print(f.read())

    # Создаем структуру датасета
    dataset_path = "yolo_dataset"
    dirs = generator.create_dataset_structure(dataset_path)

    # Создаем YAML конфиг
    yaml_path = generator.create_yaml_config(
        dataset_path,
        train_images="images/train",
        val_images="images/val",
        test_images="images/test",
    )

    print(f"\nYAML конфиг создан: {yaml_path}")
    print("\nСодержимое YAML:")
    with open(yaml_path, "r") as f:
        print(f.read())

    return annotations, yaml_path


def batch_generate_from_numpy_arrays(
    data_dir: str,
    output_dir: str,
    classes: List[str],
    image_size: Tuple[int, int] = (640, 480),
):
    """
    Пакетная генерация из numpy массивов

    Args:
        data_dir: каталог с numpy файлами (.npy) или текстовыми файлами
        output_dir: выходной каталог
        classes: список классов
        image_size: размер изображений (ширина, высота)
    """
    data_path = Path(data_dir)
    output_path = Path(output_dir)

    # Создаем структуру датасета
    generator = YOLOAnnotationGenerator(classes)
    dirs = generator.create_dataset_structure(output_dir)

    # Ищем файлы с аннотациями
    annotation_files = list(data_path.glob("*.npy")) + list(data_path.glob("*.txt"))

    print(f"Найдено {len(annotation_files)} файлов с аннотациями")

    for ann_file in annotation_files:
        try:
            # Загружаем данные
            if ann_file.suffix == ".npy":
                data = np.load(ann_file)
            else:  # .txt файл
                data = []
                with open(ann_file, "r") as f:
                    for line in f:
                        if line.strip():
                            row = [float(x) for x in line.strip().split()]
                            if row:
                                data.append(row)
                data = np.array(data)

            # Генерируем аннотации
            img_width, img_height = image_size
            annotations = generator.parse_coordinates(
                data, img_width, img_height, format_type="auto"
            )

            # Сохраняем в формате YOLO
            output_ann_file = dirs["labels"] / f"{ann_file.stem}.txt"
            generator.create_yolo_annotation_file(annotations, output_ann_file)

            print(f"Обработан: {ann_file.name} -> {len(annotations)} аннотаций")

        except Exception as e:
            print(f"Ошибка обработки {ann_file}: {e}")

    # Создаем YAML конфиг
    yaml_path = generator.create_yaml_config(
        output_dir,
        train_images="images/train",
        val_images="images/val",
        test_images="images/test",
        names=classes,
    )

    print(f"\nYAML конфиг создан: {yaml_path}")

    # Создаем файл классов
    classes_file = output_path / "classes.txt"
    with open(classes_file, "w", encoding="utf-8") as f:
        for cls in classes:
            f.write(f"{cls}\n")

    print(f"Файл классов создан: {classes_file}")


def create_advanced_yaml_config(
    dataset_path: str,
    classes: List[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
) -> Dict:
    """
    Создание расширенного YAML конфига для YOLO

    Args:
        dataset_path: путь к датасету
        classes: список классов
        train_ratio: доля тренировочных данных
        val_ratio: доля валидационных данных
        test_ratio: доля тестовых данных

    Returns:
        Конфигурация YAML
    """
    config = {
        # Основные пути
        "path": str(Path(dataset_path).absolute()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        # Классы
        "nc": len(classes),
        "names": classes,
        # Метаданные
        "description": "Custom YOLO Dataset",
        "version": "1.0",
        "license": "MIT",
        "date_created": "2024",
        # Статистика (заполняется позже)
        "stats": {
            "total_images": 0,
            "train_images": 0,
            "val_images": 0,
            "test_images": 0,
            "total_annotations": 0,
            "annotations_per_class": {cls: 0 for cls in classes},
        },
        # Гиперпараметры для обучения (пример)
        "hyperparameters": {
            "lr0": 0.01,  # начальный learning rate
            "lrf": 0.01,  # конечный learning rate
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "warmup_epochs": 3.0,
            "warmup_momentum": 0.8,
            "warmup_bias_lr": 0.1,
            "box": 0.05,  # box loss gain
            "cls": 0.5,  # cls loss gain
            "cls_pw": 1.0,  # cls BCELoss positive_weight
            "obj": 1.0,  # obj loss gain (scale with pixels)
            "obj_pw": 1.0,  # obj BCELoss positive_weight
            "iou_t": 0.20,  # IoU training threshold
            "anchor_t": 4.0,  # anchor-multiple threshold
            "fl_gamma": 0.0,  # focal loss gamma (efficientDet default gamma=1.5)
            "hsv_h": 0.015,  # image HSV-Hue augmentation (fraction)
            "hsv_s": 0.7,  # image HSV-Saturation augmentation (fraction)
            "hsv_v": 0.4,  # image HSV-Value augmentation (fraction)
            "degrees": 0.0,  # image rotation (+/- deg)
            "translate": 0.1,  # image translation (+/- fraction)
            "scale": 0.5,  # image scale (+/- gain)
            "shear": 0.0,  # image shear (+/- deg)
            "perspective": 0.0,  # image perspective (+/- fraction), range 0-0.001
            "flipud": 0.0,  # image flip up-down (probability)
            "fliplr": 0.5,  # image flip left-right (probability)
            "mosaic": 1.0,  # image mosaic (probability)
            "mixup": 0.0,  # image mixup (probability)
            "copy_paste": 0.0,  # segment copy-paste (probability)
        },
        # Архитектура модели
        "model": {"backbone": "CSPDarknet", "neck": "PANet", "head": "YOLOv5Head"},
        # Рекомендации по обучению
        "training": {
            "epochs": 100,
            "batch_size": 16,
            "imgsz": 640,
            "device": "0",  # '0' for GPU 0, '0,1,2,3' for multiple GPUs
            "workers": 8,
            "patience": 50,  # early stopping patience
            "save_period": -1,  # save checkpoint every x epochs
            "resume": False,  # resume training from last checkpoint
            "single_cls": False,  # train as single-class dataset
            "rect": False,  # rectangular training
            "cache": False,  # cache images for faster training
            "image_weights": False,  # use weighted image selection for training
            "multi_scale": False,  # vary img-size +/- 50%
        },
    }

    return config


def generate_complete_yolo_dataset(
    annotations_data: List[List[float]],
    output_dir: str = "yolo_dataset_complete",
    image_size: Tuple[int, int] = (640, 480),
    classes: List[str] = None,
    dataset_name: str = "Custom Dataset",
):
    """
    Создание полного YOLO датасета

    Args:
        annotations_data: список аннотаций в вашем формате
        output_dir: выходной каталог
        image_size: размер изображений
        classes: список классов (если None, будет определён автоматически)
        dataset_name: название датасета
    """
    print(f"Создание YOLO датасета: {dataset_name}")
    print(f"Выходной каталог: {output_dir}")

    # Автоматическое определение классов если не заданы
    if classes is None:
        # Извлекаем class_id из данных
        class_ids = set()
        for ann in annotations_data:
            if len(ann) >= 6:
                # В вашем формате class_id последний
                class_id = int(ann[5])
                class_ids.add(class_id)

        classes = [f"class_{i}" for i in sorted(class_ids)]
        print(f"Автоматически определено классов: {len(classes)}")

    # Создаем генератор
    generator = YOLOAnnotationGenerator(classes)

    # Создаем структуру каталогов
    dirs = generator.create_dataset_structure(output_dir, create_subdirs=True)

    # Разделяем данные на train/val/test
    np.random.seed(42)
    indices = np.random.permutation(len(annotations_data))

    train_end = int(len(indices) * 0.7)
    val_end = train_end + int(len(indices) * 0.2)

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    subsets = {"train": train_indices, "val": val_indices, "test": test_indices}

    img_width, img_height = image_size
    total_annotations = 0
    class_counts = {i: 0 for i in range(len(classes))}

    # Обрабатываем каждое подмножество
    for subset_name, subset_indices in subsets.items():
        print(f"\nОбработка подмножества: {subset_name}")
        print(f"Количество образцов: {len(subset_indices)}")

        subset_ann_count = 0

        for i, idx in enumerate(subset_indices):
            try:
                # Получаем аннотации для этого образца
                ann_data = annotations_data[idx]

                # Парсим аннотации
                annotations = generator.parse_coordinates(
                    [ann_data], img_width, img_height, format_type="yolo_normalized"
                )

                if annotations:
                    # Сохраняем аннотации
                    ann_filename = f"{subset_name}_{i:06d}.txt"
                    ann_path = dirs[f"labels_{subset_name}"] / ann_filename
                    generator.create_yolo_annotation_file(annotations, ann_path)

                    # Обновляем статистику
                    subset_ann_count += len(annotations)
                    total_annotations += len(annotations)

                    for ann in annotations:
                        class_counts[ann["class_id"]] += 1

                    # Создаем пустой файл изображения (заглушка)
                    img_filename = f"{subset_name}_{i:06d}.jpg"
                    img_path = dirs[f"images_{subset_name}"] / img_filename

                    # Создаем черное изображение-заглушку
                    placeholder_img = np.zeros(
                        (img_height, img_width, 3), dtype=np.uint8
                    )
                    cv2.imwrite(str(img_path), placeholder_img)

            except Exception as e:
                print(f"Ошибка обработки образца {idx}: {e}")

        print(f"  Создано аннотаций: {subset_ann_count}")

    # Создаем расширенный YAML конфиг
    config = create_advanced_yaml_config(output_dir, classes)

    # Обновляем статистику в конфиге
    config["stats"].update(
        {
            "total_images": len(annotations_data),
            "train_images": len(train_indices),
            "val_images": len(val_indices),
            "test_images": len(test_indices),
            "total_annotations": total_annotations,
            "annotations_per_class": {
                classes[i]: count for i, count in class_counts.items()
            },
        }
    )

    # Сохраняем конфиг
    config_path = Path(output_dir) / "dataset_config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(
            config, f, default_flow_style=False, allow_unicode=True, sort_keys=False
        )

    # Создаем стандартный YOLO data.yaml
    yolo_yaml_path = generator.create_yaml_config(
        output_dir,
        train_images="images/train",
        val_images="images/val",
        test_images="images/test",
        names=classes,
        config_name="data.yaml",
    )

    # Создаем README файл
    readme_path = Path(output_dir) / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(f"# {dataset_name}\n\n")
        f.write("YOLO формат датасета для обнаружения объектов\n\n")
        f.write("## Структура\n")
        f.write("```\n")
        f.write(f"{output_dir}/\n")
        f.write("├── images/\n")
        f.write("│   ├── train/      # Тренировочные изображения\n")
        f.write("│   ├── val/        # Валидационные изображения\n")
        f.write("│   └── test/       # Тестовые изображения\n")
        f.write("├── labels/\n")
        f.write("│   ├── train/      # Тренировочные аннотации\n")
        f.write("│   ├── val/        # Валидационные аннотации\n")
        f.write("│   └── test/       # Тестовые аннотации\n")
        f.write("├── data.yaml       # Конфиг для YOLO\n")
        f.write("├── dataset_config.yaml # Расширенный конфиг\n")
        f.write("└── README.md       # Этот файл\n")
        f.write("```\n\n")

        f.write("## Классы\n")
        for i, cls in enumerate(classes):
            f.write(f"{i}: {cls} ({class_counts[i]} аннотаций)\n")

        f.write("\n## Использование с YOLO\n")
        f.write("```bash\n")
        f.write(
            f"yolo train data={output_dir}/data.yaml model=yolov5s.pt epochs=100 imgsz=640\n"
        )
        f.write("```\n")

    print(f"\n{'='*60}")
    print("ДАСТАСЕТ УСПЕШНО СОЗДАН!")
    print(f"{'='*60}")
    print(f"Всего изображений: {len(annotations_data)}")
    print(f"Всего аннотаций: {total_annotations}")
    print(f"Классы: {len(classes)}")
    print(f"  Train: {len(train_indices)} изображений")
    print(f"  Val:   {len(val_indices)} изображений")
    print(f"  Test:  {len(test_indices)} изображений")
    print(f"\nСозданные файлы:")
    print(f"  - data.yaml: {yolo_yaml_path}")
    print(f"  - dataset_config.yaml: {config_path}")
    print(f"  - README.md: {readme_path}")
    print(f"{'='*60}")

    return config
