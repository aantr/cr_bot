import torch
import cv2
import os
from pathlib import Path
import argparse
import numpy as np

def detect_and_save_yolo_format(model_path, source_folder, output_folder):
    # Load the YOLO model
    try:
        # Try YOLOv8 first
        from ultralytics import YOLO
        model = YOLO(model_path)
        yolo_version = "v8"
        print("Using YOLOv8")
    except ImportError:
        # Fallback to YOLOv5
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
        yolo_version = "v5"
        print("Using YOLOv5")
    
    # Create output directories
    images_output = Path(output_folder) / 'images'
    labels_output = Path(output_folder) / 'labels'
    images_output.mkdir(parents=True, exist_ok=True)
    labels_output.mkdir(parents=True, exist_ok=True)
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    # Process each image in the source folder
    for image_file in Path(source_folder).iterdir():
        if image_file.suffix.lower() in image_extensions:
            print(f"Processing: {image_file.name}")
            
            # Read original image
            img = cv2.imread(str(image_file))
            if img is None:
                print(f"Could not read image: {image_file}")
                continue
                
            img_height, img_width = img.shape[:2]
            
            # Save original image in PNG format (without any detection drawings)
            save_path = images_output / f"{image_file.stem}.png"
            cv2.imwrite(str(save_path), img)
            
            # Run detection for labels only
            if yolo_version == "v8":
                results = model.predict(source=str(image_file), imgsz=1136, conf=0.007)
                result = results[0]
            else:
                results = model(str(image_file))
                result = results
            
            # Save labels in YOLO format
            label_path = labels_output / f"{image_file.stem}.txt"
            
            with open(label_path, 'w') as f:
                if yolo_version == "v8":
                    # YOLOv8 format
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            # Get normalized coordinates
                            xywhn = box.xywhn[0].cpu().numpy()
                            class_id = int(box.cls[0].cpu().numpy())
                            
                            x_center, y_center, width, height = xywhn
                            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                else:
                    # YOLOv5 format
                    if len(result.xywhn[0]) > 0:  # Check if detections exist
                        for *box, conf, cls in result.xywhn[0].tolist():
                            class_id = int(cls)
                            x_center, y_center, width, height = box
                            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    print("Detection completed! Images saved in PNG format without labels.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLO detection on dataset images")
    parser.add_argument("--model", type=str, required=True, help="Path to YOLO model file (.pt)")
    parser.add_argument("--source", type=str, required=True, help="Source folder containing images")
    parser.add_argument("--output", type=str, default="output_dataset", help="Output folder for results")
    
    args = parser.parse_args()
    
    detect_and_save_yolo_format(args.model, args.source, args.output)
