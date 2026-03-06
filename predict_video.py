from ultralytics import YOLO
import cv2

# Загрузка модели
model = YOLO('runs/detect/cr_bot/train_blue/weights/best.pt')  # или 'yolov8n.pt'

# Детекция на видео и сохранение результата
results = model.predict(source='screenshot/rec1.mp4', 
                       conf=0.5,
                       show=True,
                       save=True,  # Сохранить видео с детекцией
                       project='detection_results',  # Папка для сохранения
                       name='video_detection')  # Имя эксперимента

print("Детекция завершена!")
