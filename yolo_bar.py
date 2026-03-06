import math
from ultralytics import YOLO
import yaml

import os
import shutil
import yaml
from pathlib import Path

from KataCR.katacr.build_dataset.constant import path_yaml

from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import torch

from KataCR.katacr.yolov8.train import YOLO_CR
from image2cards import get_image_cards_format
from image2yolo import get_image_yolo_format

def prepare_yolov8_dataset(data_path, output_path):
    """
    Convert your dataset to YOLOv8 format
    data_path: path containing images and txt files
    output_path: path where organized dataset will be saved
    """
    
    # Create train/val folders
    for split in ['train', 'val']:
        os.makedirs(f"{output_path}/{split}/images", exist_ok=True)
        os.makedirs(f"{output_path}/{split}/labels", exist_ok=True)
    
    # Get all image files
    image_files = list(Path(data_path).glob("*.png")) + \
                  list(Path(data_path).glob("*.jpg")) + \
                  list(Path(data_path).glob("*.jpeg"))
    
    # Split data (80% train, 20% val)
    from sklearn.model_selection import train_test_split
    train_files, val_files = train_test_split(image_files, test_size=0.2, random_state=42)
    
    # Copy files to train folder
    for img_path in train_files:
        # Copy image
        shutil.copy(img_path, f"{output_path}/train/images/{img_path.name}")
        
        # Copy corresponding label file
        label_path = img_path.with_suffix('.txt')
        if label_path.exists():
            shutil.copy(label_path, f"{output_path}/train/labels/{label_path.name}")
    
    # Copy files to val folder
    for img_path in val_files:
        shutil.copy(img_path, f"{output_path}/val/images/{img_path.name}")
        label_path = img_path.with_suffix('.txt')
        if label_path.exists():
            shutil.copy(label_path, f"{output_path}/val/labels/{label_path.name}")
    
    print(f"Dataset organized: {len(train_files)} training, {len(val_files)} validation images")

def create_data_yaml(output_path, class_names):
    """
    Create data.yaml file for YOLOv8
    class_names: list of class names
    """
    data = {
        'path': os.path.abspath(output_path),
        'train': 'train/images',
        'val': 'val/images',
        'nc': len(class_names),
        'names': class_names
    }
    
    with open(f"{output_path}/data.yaml", 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    
    print(f"data.yaml created with {len(class_names)} classes")


# Usage
def dataset_copy():
    data_path = "KataCR/logs/generation"  # Your dataset folder with images and txt files
    output_path = "yolo_dataset_bars"  # Output folder for organized dataset

    with open(f"bar-dataset.yaml", 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)


    class_names = data['names']  # Your class names
    
    prepare_yolov8_dataset(data_path, output_path)
    create_data_yaml(output_path, class_names)

import cv2
import torch
from ultralytics import YOLO
import numpy as np

def visualize_yolo_detection(model_path, image_path, class_names=None, confidence_threshold=0.5):
    """
    Visualize YOLO detection results
    
    Args:
        model_path: Path to trained YOLO model (.pt file)
        image_path: Path to image for detection
        class_names: List of class names
        confidence_threshold: Minimum confidence threshold
    """
    
    # Load the trained model
    model = YOLO(model_path)
    
    # Run inference
    img = get_image_yolo_format(cv2.imread(image_path))

    results = model(img, conf=0)

    img_height, img_width = img.shape[:2]
    
    # Process results
    for result in results:
        boxes = result.boxes  # Boxes object for bbox outputs
        print(boxes)
        if boxes is not None:
            # Extract bounding boxes, confidences, and class IDs
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                
                # Filter by confidence threshold
                if confidence >= confidence_threshold:
                    # Draw bounding box
                    color = (0, 255, 0)  # Green
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    
                    # Add label with confidence
                    if class_id < len(class_names):
                        class_name = class_names[class_id]
                        label = f"{class_name}: {confidence:.2f}"
                        
                        # Get text size for background
                        (text_width, text_height), _ = cv2.getTextSize(label, 
                                                                      cv2.FONT_HERSHEY_SIMPLEX, 
                                                                      0.5, 2)
                        
                        # Draw background rectangle for text
                        cv2.rectangle(img, (x1, y1 - text_height - 10), 
                                    (x1 + text_width, y1), color, -1)
                        
                        # Draw text
                        cv2.putText(img, label, (x1, y1 - 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    # Display result
    height, width = img.shape[:2]
    resized = cv2.resize(img, (width // 2, height // 2), 
                        interpolation=cv2.INTER_AREA)
    cv2.imshow('YOLO Labels Visualization', resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return img


def test():
    path_source = "vid2.png"
    model = YOLO('detector1_v0.7.131.pt')
    model.save('model')

    results = model(source=path_source)
    print(list(results[0].boxes))
    # Визуализация результатов
    field = [[None for _ in range(8)] for _ in range(8)]
    for r in results:
        im_array = r.plot()  # изображение с bounding boxes
        centers = []
        if len(r.boxes.data) == 0:
            continue
        mn = mx = [r.boxes.data[0][0], r.boxes.data[0][1]]
        mn = mx.copy()
        for i in r.boxes.data:
            mn[0] = min(mn[0], i[0])
            mn[1] = min(mn[1], i[1])
            mx[0] = max(mx[0], i[2])
            mx[1] = max(mx[1], i[3])
        for i in r.boxes.data:
            centers.append([float((i[0] + i[2]) / 2 - mn[0]) / (mx[0] - mn[0]), float((i[1] + i[3]) / 2 - mn[1]) / (mx[1] - mn[1])])
            for j in range(2):
                centers[-1][j] = math.floor(centers[-1][j] * 8)
            if (field[centers[-1][1]][centers[-1][0]] is None):
                field[centers[-1][1]][centers[-1][0]] = int(i[5])

        # print(centers, r.boxes.cls, r.orig_shape)
        cv2.imwrite('result.png', im_array)

def main():
    model_path = 'runs/detect/cr_bot/train_bars4/weights/best.pt'  # Your trained model
    # model_path = 'KataCR/runs/detector1_v0.7.13.pt'  # Your trained model
    # image_path = 'KataCR/logs/generation/gen_97.jpg'
    image_path = 'yolo_dataset_bars/train/images/gen_10.jpg'
    # image_path = 'screenshot/IMG_0836.PNG'
    with open(f"bar-dataset.yaml", 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)

    class_names = data['names']  # Your class names
    class_names = [str(i) for i in range(1000)]  # Your class names

    result_img = visualize_yolo_detection(model_path, image_path, class_names, confidence_threshold=0.5)

    # Посмотрите метрики обучения
    # import matplotlib.pyplot as plt
    # import pandas as pd

    # # Загрузите логи обучения
    # df = pd.read_csv('runs/detect/train/results.csv')
    # plt.plot(df['metrics/mAP50(B)'])
    # plt.title('mAP50 during training')
    # plt.show()
    
    dataset_copy()
    train()

if __name__ == '__main__':
    main()