from ultralytics import YOLO

if __name__ == '__main__':

    # Load a pre-trained classification model
    model = YOLO("yolo11n-cls.pt")  # 'n' for nano, 's' for small, 'm' for medium, etc.

    # Train the model
    results = model.train(
        data="split_dataset_cache",  # path to the folder containing train/ and val/
        epochs=100,              # number of training cycles
        imgsz=224,               # input image size (224x224 pixels)
        batch=16,                # images per batch (adjust based on GPU memory)
        device=0                 # GPU device (0 for first GPU, 'cpu' for CPU)
    )