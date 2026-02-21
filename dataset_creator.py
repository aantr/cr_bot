import cv2
import os
import json
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import argparse
import shutil

class YOLODatasetCreator:
    def __init__(self, model_path, input_dir, output_dir, class_names=None, confidence_threshold=0.3):
        """
        Инициализация создателя датасета
        
        Args:
            model_path: путь к обученной модели (.pt)
            input_dir: директория с исходными изображениями
            output_dir: директория для сохранения датасета
            class_names: список имен классов
            confidence_threshold: порог уверенности для детекций
        """
        self.model = YOLO(model_path)
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.confidence_threshold = confidence_threshold
        
        # Создание структуры директорий
        self.images_dir = self.output_dir / 'images'
        self.labels_dir = self.output_dir / 'labels'
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Имена классов
        self.class_names = class_names if class_names else [f'class_{i}' for i in range(80)]
        
        print(f"Model loaded from: {model_path}")
        print(f"Input directory: {input_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Confidence threshold: {confidence_threshold}")
    
    def convert_bbox_to_yolo(self, x1, y1, x2, y2, img_width, img_height):
        """
        Конвертация bounding box из формата (x1,y1,x2,y2) в YOLO формат
        """
        x_center = ((x1 + x2) / 2) / img_width
        y_center = ((y1 + y2) / 2) / img_height
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height
        return x_center, y_center, width, height
    
    def run_inference_and_save(self, image_path):
        """
        Запуск инференса и сохранение результатов
        """
        try:
            # Загрузка изображения
            img = cv2.imread(str(image_path))
            if img is None:
                print(f"Could not load image: {image_path}")
                return False
            
            img_height, img_width = img.shape[:2]
            
            # Запуск модели
            results = self.model(img, conf=self.confidence_threshold)
            
            # Сохранение изображения
            output_image_path = self.images_dir / image_path.name
            shutil.copy2(image_path, output_image_path)
            
            # Создание файла аннотаций
            label_filename = image_path.stem + '.txt'
            label_path = self.labels_dir / label_filename
            
            detections_found = False
            
            # Обработка результатов
            with open(label_path, 'w') as f:
                for result in results:
                    if result.boxes is not None:
                        for box in result.boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                            confidence = box.conf[0].cpu().numpy()
                            class_id = int(box.cls[0].cpu().numpy())
                            
                            if confidence >= self.confidence_threshold:
                                # Конвертация в YOLO формат
                                x_center, y_center, width, height = self.convert_bbox_to_yolo(
                                    x1, y1, x2, y2, img_width, img_height
                                )
                                
                                # Запись в файл
                                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                                detections_found = True
            
            if not detections_found:
                # Создание пустого файла если нет детекций
                with open(label_path, 'w') as f:
                    pass  # Пустой файл
            
            print(f"Processed: {image_path.name}")
            return True
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return False
    
    def create_dataset_yaml(self):
        """
        Создание файла конфигурации dataset.yaml
        """
        yaml_content = f"""path: {self.output_dir.absolute()}
train: images
val: images

names:
"""
        for i, name in enumerate(self.class_names):
            yaml_content += f"  {i}: {name}\n"
        
        yaml_path = self.output_dir / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        print(f"Dataset config saved to: {yaml_path}")
    
    def process_all_images(self):
        """
        Обработка всех изображений в директории
        """
        # Поддерживаемые форматы
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        
        # Поиск изображений
        for ext in image_extensions:
            image_files.extend(self.input_dir.glob(f'*{ext}'))
            image_files.extend(self.input_dir.glob(f'*{ext.upper()}'))
        
        print(f"Found {len(image_files)} images to process")
        
        if not image_files:
            print("No images found in input directory!")
            return
        
        # Обработка изображений
        processed_count = 0
        for image_file in image_files:
            if self.run_inference_and_save(image_file):
                processed_count += 1
        
        print(f"\nProcessing complete!")
        print(f"Successfully processed: {processed_count}/{len(image_files)} images")
        
        # Создание конфигурационного файла
        self.create_dataset_yaml()
        
        return processed_count

def main():
    parser = argparse.ArgumentParser(description='Create YOLO dataset with pre-labeling using trained model')
    parser.add_argument('--model', required=True, help='Path to trained model (.pt file)')
    parser.add_argument('--input', required=True, help='Input directory with images')
    parser.add_argument('--output', required=True, help='Output directory for dataset')
    parser.add_argument('--conf', type=float, default=0.3, help='Confidence threshold (default: 0.3)')
    parser.add_argument('--classes', nargs='+', help='Class names (optional)')
    
    args = parser.parse_args()
    
    # Создание создателя датасета
    creator = YOLODatasetCreator(
        model_path=args.model,
        input_dir=args.input,
        output_dir=args.output,
        class_names=args.classes,
        confidence_threshold=args.conf
    )
    
    # Обработка изображений
    creator.process_all_images()

# Альтернативный способ использования без командной строки
def create_dataset_example():
    """
    Пример использования без командной строки
    """
    # Параметры
    model_path = 'best.pt'  # Ваша обученная модель
    input_dir = 'raw_images/'  # Директория с исходными изображениями
    output_dir = 'yolo_dataset/'  # Директория для датасета
    class_names = ['object1', 'object2', 'object3']  # Ваши классы
    
    # Создание датасета
    creator = YOLODatasetCreator(
        model_path=model_path,
        input_dir=input_dir,
        output_dir=output_dir,
        class_names=class_names,
        confidence_threshold=0.3
    )
    
    # Обработка
    creator.process_all_images()
    
    print("\nDataset creation completed!")
    print(f"Images: {output_dir}/images/")
    print(f"Labels: {output_dir}/labels/")
    print(f"Config: {output_dir}/dataset.yaml")

if __name__ == '__main__':
    # Использование через командную строку
    main()
    
    # Или раскомментируйте для прямого вызова:
    # create_dataset_example()
