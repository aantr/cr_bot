import cv2
import os
import numpy as np

import cv2
import os
import numpy as np
from pathlib import Path

def visualize_yolo_labels(images_dir, labels_dir, class_names=None, output_dir=None):
    """
    Visualize YOLO dataset labels with bounding boxes
    
    Args:
        images_dir: Directory containing images
        labels_dir: Directory containing YOLO format labels (.txt files)
        class_names: List of class names
        output_dir: Directory to save visualized images (optional)
    """
    
    # Default class names if not provided
    if class_names is None:
        class_names = [f'class_{i}' for i in range(80)]  # Up to 80 classes
    
    # Create colors for different classes
    colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 128),  # Purple
        (255, 165, 0),  # Orange
    ]
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(Path(images_dir).glob(f'*{ext}'))
    
    print(f"Found {len(image_files)} images")
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Process each image
    for img_path in image_files:
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Could not load image: {img_path}")
            continue
            
        h, w = img.shape[:2]
        
        # Load corresponding label file
        label_filename = img_path.stem + '.txt'
        label_path = Path(labels_dir) / label_filename
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            # Draw bounding boxes
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
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
                        
                        # Get color for this class
                        color = colors[class_id % len(colors)]
                        
                        # Draw bounding box
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                        
                        # Add class name
                        if class_id < len(class_names):
                            class_name = class_names[class_id]
                            # Draw background for text
                            cv2.rectangle(img, (x1, y1-20), (x1+len(class_name)*10, y1), color, -1)
                            cv2.putText(img, class_name, (x1, y1-5), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                        
                        # Add confidence if available (for detection results)
                        if len(parts) > 5:
                            confidence = float(parts[5])
                            cv2.putText(img, f'{confidence:.2f}', (x1, y2+15),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                            
                    except ValueError as e:
                        print(f"Error parsing line in {label_path}: {line}")
                        continue
        
        # Display or save image
        if output_dir:
            output_path = Path(output_dir) / img_path.name
            cv2.imwrite(str(output_path), img)
            print(f"Saved: {output_path}")
        else:
            # Display image
            cv2.imshow('YOLO Labels Visualization', img)
            print(f"Showing: {img_path.name}")
            print("Press any key to continue, 'q' to quit...")
            
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break
    
    cv2.destroyAllWindows()

# Usage example
images_dir = 'create_bars_dataset/output_dataset/images'
labels_dir = 'create_bars_dataset/output_dataset/labels'

visualize_yolo_labels(images_dir, labels_dir)

# # Usage
# images_dir = 'yolo_dataset/train/images'
# labels_dir = 'yolo_dataset/train/labels'

# visualize_yolo_dataset(images_dir, labels_dir, class_names)
