import cv2
import numpy as np  

size = (1136, 1792)  # (width, height)

crop_coords = {
    (1136, 1792): (0, 0, 1136, 1792),
    (1206, 2622): (35, 150, 1206 - 35, 2622 - 512) # my iphone 16 pro
}

def get_image_yolo_format(image):
    
    # Загрузка изображения если передан путь
    img = image.copy()

    # Получение координат обрезки
    crop_key = img.shape[:2:][::-1]
    if crop_key in crop_coords:
        x1, y1, x2, y2 = crop_coords[crop_key]
    else:
        raise ValueError('Wrong input image size')
    
    # Проверка границ
    h, w = img.shape[:2]
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    
    # Обрезка изображения
    cropped_img = img[y1:y2, x1:x2]
    
    # Изменение размера до заданного размера
    target_width, target_height = size
    resized_img = cv2.resize(cropped_img, (target_width, target_height), 
                           interpolation=cv2.INTER_AREA)
    
    return resized_img
