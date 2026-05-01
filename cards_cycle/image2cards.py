import cv2
import numpy as np

size = (920, 512)  # (width, height)

crop_coords = {
    # (1206, 2622): (251, 2622 - 512, 1206 - 35, 2622),  # my iphone 16 pro
    (720, 1280): (85, 70, 720 - 15, 180),  # ldplayer 1280x720
    (900, 1600): (100, 100, 900 - 25, 250),  # ldplayer 1600x900
}


def get_image_cards_format(image):
    # Загрузка изображения если передан путь
    if isinstance(image, str):
        img = cv2.imread(image)
        if img is None:
            raise ValueError(f"Could not load image from {image}")
    else:
        img = image.copy()

    # Получение координат обрезки
    crop_key = img.shape[:2][::-1]  # (width, height)
    if crop_key in crop_coords:
        x1, y1, x2, y2 = crop_coords[crop_key]
    else:
        raise ValueError(
            f"Wrong input image size: {crop_key}. Available sizes: {list(crop_coords.keys())}"
        )

    # Проверка границ
    h, w = img.shape[:2]
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))

    # Обрезка изображения
    cropped_img = img[y1:y2, x1:x2]

    # Изменение размера до заданного размера
    # target_width, target_height = size
    # resized_img = cv2.resize(
    #     cropped_img, (target_width, target_height), interpolation=cv2.INTER_AREA
    # )

    return cropped_img

def get_card_by_index(image, index):    
    # Загрузка изображения если передан путь
    if isinstance(image, str):
        img = cv2.imread(image)
        if img is None:
            raise ValueError(f"Could not load image from {image}")
    else:
        img = image.copy()

    # Получение координат обрезки
    width, height = img.shape[:2][::-1]  # (width, height)
    
    # Обрезка изображения
    cropped_img = img[:, width * index // 8: width * (index + 1) // 8]
    
    return cropped_img

# Пример использования:
if __name__ == "__main__":
    # Путь к изображению задается в переменной
    image_path = "screenshot/34.png"  # Замените на путь к вашему изображению

    try:
        # Вариант 1: Передача пути к файлу
        processed_image = get_image_cards_format(image_path)
        print(f"Original image processed successfully!")
        print(f"Output image shape: {processed_image.shape}")

        # Сохранение результата
        cv2.imshow("output_processed_image.jpg", processed_image)
        cv2.waitKey(0)

    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
