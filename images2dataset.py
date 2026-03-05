# images -> dataset -> cvat -> yolo dataset for training -> train -> new model ->

import os
import shutil
from pathlib import Path
import yaml
from ultralytics import YOLO

def create_yolo_dataset(model_path, data_yaml_path, images_folder, output_dir='dataset'):
    # Загрузка модели
    model = YOLO(model_path)

    # Загрузка информации о классах из data.yaml
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    class_names = data_config.get('names', [])
    nc = len(class_names)

    # Пути для сохранения
    images_train_dir = Path(output_dir) / 'images' / 'train'
    labels_train_dir = Path(output_dir) / 'labels' / 'train'

    images_train_dir.mkdir(parents=True, exist_ok=True)
    labels_train_dir.mkdir(parents=True, exist_ok=True)

    # Список изображений
    image_paths = list(Path(images_folder).rglob('*'))
    image_paths = [p for p in image_paths if p.suffix.lower() in ('.jpg', '.jpeg', '.png')]

    train_txt_path = Path(output_dir) / 'train.txt'
    with open(train_txt_path, 'w') as train_file:
        for img_path in image_paths:
            # Инференс
            results = model.predict(source=str(img_path), save=False, conf=0.001, iou=0.45)

            # Имя файла без расширения
            stem = img_path.stem

            # Копируем изображение
            dest_img = images_train_dir / img_path.name
            shutil.copy(str(img_path), str(dest_img))

            # Относительный путь для train.txt
            rel_img_path = f"images/train/{img_path.name}"
            train_file.write(rel_img_path + '\n')

            # Сохраняем аннотации в формате YOLO
            label_path = labels_train_dir / f"{stem}.txt"
            with open(label_path, 'w') as lbl_file:
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            cls = int(box.cls.item())
                            xywh = box.xywhn[0].tolist()
                            line = f"{cls} " + " ".join(map(str, xywh))
                            lbl_file.write(line + "\n")

    # Создание data.yaml
    new_data_yaml = {
        'path': '.',
        'train': 'train.txt',
        'names': class_names
    }

    with open(Path(output_dir) / 'data.yaml', 'w') as f:
        yaml.dump(new_data_yaml, f, default_flow_style=False, sort_keys=False)

    print(f"[INFO] Датасет успешно создан в '{output_dir}'")


# Пример использования
if __name__ == "__main__":
    # Пути к файлам и папкам
    model_path = "runs/detect/train/weights/best.pt"  # Ваша обученная модель
    data_yaml_path = "bar-dataset.yaml"  # Путь к вашему yaml файлу
    images_folder = "test_images"  # Папка с изображениями для обработки
    output_folder = "test_yolo_dataset"  # Выходная папка для датасета

    create_yolo_dataset(model_path, data_yaml_path, images_folder, output_dir=output_folder)


    

