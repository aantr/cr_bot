import cv2
import os
import numpy as np

def visualize_yolo_dataset(images_dir, labels_dir, class_names=None):
    """
    Visualize YOLO dataset with bounding boxes
    
    Args:
        images_dir: Directory containing images
        labels_dir: Directory containing YOLO format labels (.txt files)
        class_names: List of class names (optional)
    """
    
    # Default COCO class names if not provided
    if class_names is None:
        class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
                      'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
                      'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
                      'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
                      'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
                      'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
                      'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
                      'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
                      'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
                      'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
                      'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
                      'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                      'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    
    # Get list of image files
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    for image_file in image_files:
        # Load image
        image_path = os.path.join(images_dir, image_file)
        img = cv2.imread(image_path)
        
        if img is None:
            print(f"Could not load image: {image_path}")
            continue
            
        # Get corresponding label file
        label_file = os.path.splitext(image_file)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_file)
        
        # Load annotations if label file exists
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                
            h, w = img.shape[:2]
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # Convert YOLO format to pixel coordinates
                    x1 = int((x_center - width/2) * w)
                    y1 = int((y_center - height/2) * h)
                    x2 = int((x_center + width/2) * w)
                    y2 = int((y_center + height/2) * h)
                    
                    # Draw bounding box
                    color = (0, 255, 0)  # Green
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    
                    # Add class name
                    if class_id < len(class_names):
                        class_name = class_names[class_id]
                        cv2.putText(img, class_name, (x1, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Display image
        cv2.imshow('YOLO Dataset Visualization', img)
        key = cv2.waitKey(0)
        if key == ord('q'):  # Press 'q' to quit
            break
    
    cv2.destroyAllWindows()

# Usage
images_dir = 'yolo_dataset/train/images'
labels_dir = 'yolo_dataset/train/labels'
class_names = ['class1', 'class2']  # Your class names

visualize_yolo_dataset(images_dir, labels_dir, class_names)
