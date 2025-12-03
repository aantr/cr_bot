import math
from ultralytics import YOLO
import yaml
model = YOLO('yolov8n.pt')

results = model.train(
    data='KataCR/logs/generation/ClashRoyale_detection.yaml'
)


from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Загрузка предобученной модели YOLOv8
model = YOLO('best.pt')  # yolov8n.pt - самая легкая версия
model.save('model')

# Распознавание на изображении
results = model('screnrenr.png')

# Визуализация результатов
field = [[None for _ in range(8)] for _ in range(8)]
for r in results:
    im_array = r.plot()  # изображение с bounding boxes
    centers = []
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
print(*field, sep='\n')