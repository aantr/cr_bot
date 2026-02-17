import cv2
import numpy as np

import cv2
import numpy as np


def filter_by_color(image_path, target_color, tolerance=30):
    """
    Фильтрует изображение, оставляя только пиксели заданного цвета

    Args:
        image_path: путь к изображению
        target_color: целевой цвет в формате [B, G, R]
        tolerance: допуск по цвету
    """

    # Загружаем изображение
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Не удалось загрузить изображение")

    # Преобразуем цвет в numpy массив
    target = np.array(target_color, dtype=np.uint8)

    # Создаем границы цвета с допуском
    lower_bound = np.maximum(target - tolerance, 0)
    upper_bound = np.minimum(target + tolerance, 255)

    # Создаем маску
    mask = cv2.inRange(image, lower_bound, upper_bound)

    # Применяем маску к изображению
    filtered_image = cv2.bitwise_and(image, image, mask=mask)

    # Сохраняем результаты
    cv2.imwrite("color_mask.jpg", mask)
    cv2.imwrite("filtered_image.jpg", filtered_image)

    print("Результаты сохранены:")
    print("- 'color_mask.jpg' - черно-белая маска")
    print("- 'filtered_image.jpg' - отфильтрованное изображение")

    return mask, filtered_image


# Пример использования:
# mask, filtered_img = filter_by_color(
#     "game_screenshot.png",
#     target_color=[0, 255, 0],  # Зеленый
#     tolerance=30
# )


def detect_health_bar_simple(
    image_path, start_x, start_y, roi_height=30, roi_width=200
):
    # Загружаем изображение
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Не удалось загрузить изображение")

    # Обрезаем регион интереса
    roi = image[start_y : start_y + roi_height, start_x : start_x + roi_width]

    # Преобразуем в градации серого
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Детектируем границы
    edges = cv2.Canny(
        gray,
        50,
        150,
    )

    # Ищем линии
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=50,
        maxLineGap=5,
    )

    max_length = 0
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x1 > 3:
                continue
            print(line)
            # Проверяем, что линия почти горизонтальная
            if abs(y2 - y1) <= 3:  # Допуск по вертикали
                length = abs(x2 - x1)
                if length > max_length:
                    max_length = length

    return max_length


import cv2
import numpy as np


def detect_colored_rectangle(
    image_path,
    start_x,
    start_y,
    target_color,
    tolerance=30,
    min_width=10,
    max_width=300,
):
    """
    Находит прямоугольник сплошного цвета, начинающийся с заданной точки

    Args:
        image_path: путь к изображению
        start_x, start_y: начальные координаты
        target_color: целевой цвет в формате [B, G, R]
        tolerance: допуск по цвету
        min_width: минимальная ширина полоски
        max_width: максимальная ширина полоски
    """

    # Загружаем изображение
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Не удалось загрузить изображение")

    height, width = image.shape[:2]

    # Проверяем, что начальные координаты в пределах изображения
    if start_x >= width or start_y >= height:
        raise ValueError("Начальные координаты выходят за пределы изображения")

    # Целевой цвет
    target = np.array(target_color, dtype=np.uint8)

    # Создаем маску для цвета с допуском
    lower_bound = np.maximum(target - tolerance, 0)
    upper_bound = np.minimum(target + tolerance, 255)

    # Создаем маску
    mask = cv2.inRange(image, lower_bound, upper_bound)

    # Находим контуры
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Ищем прямоугольник, который содержит начальную точку
    best_rect = None
    max_area = 0

    for contour in contours:
        # Получаем ограничивающий прямоугольник
        x, y, w, h = cv2.boundingRect(contour)

        # Проверяем, находится ли начальная точка внутри прямоугольника
        if x <= start_x <= x + w and y <= start_y <= y + h:
            area = w * h
            # Выбираем прямоугольник с максимальной площадью
            if area > max_area:
                max_area = area
                best_rect = (x, y, w, h)

    if best_rect:
        x, y, w, h = best_rect
        print(f"Найден прямоугольник: x={x}, y={y}, ширина={w}, высота={h}")
        return w, (x, y, w, h)
    else:
        print("Прямоугольник не найден")
        return 0, None


def visualize_detection(image_path, start_x, start_y, target_color, tolerance=30):
    """
    Визуализирует результаты детекции
    """
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Не удалось загрузить изображение")

    # Создаем копию для отображения
    result_image = image.copy()

    # Рисуем начальную точку
    cv2.circle(result_image, (start_x, start_y), 5, (0, 0, 255), -1)

    # Детектируем прямоугольник
    length, rect_info = detect_colored_rectangle(
        image_path, start_x, start_y, target_color, tolerance
    )

    if rect_info:
        x, y, w, h = rect_info
        # Рисуем найденный прямоугольник
        cv2.rectangle(result_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Добавляем текст с информацией
        cv2.putText(
            result_image,
            f"Length: {length}px",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

    # Сохраняем результат
    cv2.imwrite("detection_result.jpg", result_image)
    print("Результат сохранен в 'detection_result.jpg'")

    return length


import cv2
import numpy as np


def find_health_bar_length(image, start_x, start_y, bar_color, tolerance=10):
    height, width = image.shape

    # Преобразуем цвет в BGR (если используешь RGB — поменяй порядок)
    target_color = np.array(bar_color)  # например [0, 255, 0] для зелёного

    length = 0
    for x in range(start_x, width):
        pixel = image[y, x]
        if np.all(np.abs(pixel - target_color) < tolerance):
            length += 1
        else:
            break  # Конец полоски
    return length

import os

def quick_find_png(where) -> list:
    """
    Быстрый поиск PNG файлов в текущей папке и подпапках
    """
    print("🔍 Поиск PNG файлов...")
    
    path = []
    png_count = 0
    for root, dirs, files in os.walk(where):
        for file in files:
            if file.lower().endswith('.png'):
                full_path = os.path.join(root, file)
                path.append(full_path)
    return path
    
    print(f"\n✅ Найдено {png_count} PNG файлов")


# Загрузка изображения и шаблона
img = cv2.imread("ph12.png", 0)
for template in [
    # '15_y.png',
    # "clash_royale_dataset/dataset/fisherman/fisherman_001.png",
    *quick_find_png('clash_royale_dataset/dataset/battle-ram'),
    *quick_find_png('clash_royale_dataset/dataset/inferno-dragon'),
    *quick_find_png('clash_royale_dataset/dataset/rocket'),
    *quick_find_png('clash_royale_dataset/dataset/goblin-gang'),
    *quick_find_png('clash_royale_dataset/dataset/the-log'),
    *quick_find_png('clash_royale_dataset/dataset/cannon'),
    *quick_find_png('clash_royale_dataset/dataset/goblin-gang'),
    # *quick_find_png('clash_royale_dataset/dataset/lightning'),
    *quick_find_png('clash_royale_dataset/dataset/musketeer-hero'),
    # *quick_find_png('clash_royale_dataset/dataset/musketeer-hero'),
]:
    name = template
    template = cv2.imread(template, 0)
    coeff = 1.25
    (width, height) = int(template.shape[1] * coeff), int(template.shape[0] * coeff)
    template = cv2.resize(template, (width, height), interpolation=cv2.INTER_LINEAR)
    
    cv2.imwrite('resized.png', template)

    w, h = template.shape[::-1]

    # Сопоставление
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.4
    loc = np.where(res >= threshold)

    # Рисование прямоугольников
    for pt in zip(*loc[::-1]):
        print('rect ', name)
        cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 4)

        x, y = pt[0] + w // 2, pt[1] + h // 2

# 66041b
# 700225.   


cv2.imshow("s", img)

cv2.waitKey(0)
