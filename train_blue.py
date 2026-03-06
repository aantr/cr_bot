from ultralytics import YOLO

model = YOLO("yolov8n.pt")

print("Starting training...")
results = model.train(
    data="yolo_dataset_blue/data.yaml" ,
    epochs=50,
    imgsz=1136,
    batch=16,
    device=0,
    # workers=8,        # Number of dataloading workers
    # patience=50,       # Early stopping patience
    # optimizer='auto',  # Optimizer
    lr0=0.01,          # Initial learning rate
    augment=True,      # Apply augmentation
    project='cr_bot', # Project name for saving results
    name='train_blue', # Experiment name
)

# By default, they are saved to 'runs/detect/train/' (or train2, train3, etc.)
print("Training finished!")
print(f"Best model weights saved to: {results.save_dir}/weights/best.pt")