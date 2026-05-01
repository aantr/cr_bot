import os
import random
import cv2
import numpy as np
from pathlib import Path

def create_centered_dataset(
    base_path="path/to/folder",
    output_dir="dataset_centered",
    image_size=(128, 128),
    images_per_unit=100,
    trash_folder="trash_ims",
    trash_count_range=(1, 5),  # диапазон количества мусорных объектов
    trash_size_range=(0.3, 0.5)  # размер мусора относительно фона
):
    """
    Создает датасет с юнитами в центре и мусорными объектами на фоне.
    
    Args:
        base_path: путь к папке с фонами и юнитами
        output_dir: директория для сохранения
        image_size: размер выходного изображения (width, height)
        images_per_unit: количество изображений на юнит
        trash_folder: имя папки с мусорными изображениями
        trash_count_range: диапазон количества мусорных объектов (min, max)
        trash_size_range: диапазон размера мусора относительно фона (min, max)
    """
    base_path = Path(base_path)
    output_dir = Path(output_dir)
    
    # Загружаем фоны
    background_dir = base_path / "backgrounds"
    backgrounds = []
    if background_dir.exists():
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
            backgrounds.extend(background_dir.glob(ext))
    
    if not backgrounds:
        raise ValueError(f"Не найдены фоны в {background_dir}")
    
    # Загружаем мусорные изображения
    trash_dir = base_path / trash_folder
    trash_images = []
    if trash_dir.exists():
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
            trash_images.extend(trash_dir.glob(ext))
        print(f"Загружено мусорных изображений: {len(trash_images)}")
    else:
        print(f"Предупреждение: папка с мусором {trash_dir} не найдена")
    
    # Находим все папки с юнитами
    unit_dirs = []
    for item in base_path.iterdir():
        if item.is_dir() and item.name not in ['backgrounds', trash_folder]:
            unit_dirs.append(item)
    
    if not unit_dirs:
        raise ValueError("Не найдены папки с юнитами")
    
    print(f"Найдено фонов: {len(backgrounds)}")
    print(f"Найдено юнитов: {len(unit_dirs)}")
    
    total_images = 0
    
    # Для каждой папки с юнитами
    for unit_dir in unit_dirs:
        unit_name = unit_dir.name
        
        # Загружаем изображения юнитов
        unit_images = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
            unit_images.extend(unit_dir.glob(ext))
        
        if not unit_images:
            print(f"Предупреждение: нет изображений в {unit_dir}")
            continue
        
        # Создаем папку для юнита
        unit_output_dir = output_dir / unit_name
        unit_output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nГенерация для {unit_name} ({len(unit_images)} изображений)")
        
        for i in range(images_per_unit):
            # Выбираем случайный фон
            bg_path = random.choice(backgrounds)
            bg = cv2.imread(str(bg_path))
            
            if bg is None:
                continue
            
            # Получаем область фона 64x64
            h, w = bg.shape[:2]
            target_h, target_w = image_size[1], image_size[0]
            
            if h >= target_h and w >= target_w:
                # Вырезаем случайную область
                x = random.randint(0, w - target_w)
                y = random.randint(0, h - target_h)
                bg_crop = bg[y:y+target_h, x:x+target_w].copy()
            else:
                # Центрируем маленький фон
                canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                x_offset = (target_w - min(w, target_w)) // 2
                y_offset = (target_h - min(h, target_h)) // 2
                canvas[y_offset:y_offset+min(h, target_h), 
                       x_offset:x_offset+min(w, target_w)] = bg[:min(h, target_h), 
                                                                :min(w, target_w)]
                bg_crop = canvas
            
            
            # Выбираем случайный юнит
            unit_path = random.choice(unit_images)
            unit = cv2.imread(str(unit_path), cv2.IMREAD_UNCHANGED)
            
            if unit is None:
                continue
            
            result = bg_crop.copy()
            
            # Центрируем юнит
            unit_h, unit_w = unit.shape[:2]
            
            # Масштабируем юнит, если он слишком большой
            # max_unit_size = min(target_h, target_w) // 2
            # if unit_h > max_unit_size or unit_w > max_unit_size:
            #     scale = min(max_unit_size / unit_h, max_unit_size / unit_w)
            #     new_h = max(5, int(unit_h * scale))
            #     new_w = max(5, int(unit_w * scale))
            #     unit = cv2.resize(unit, (new_w, new_h), interpolation=cv2.INTER_AREA)
            #     unit_h, unit_w = unit.shape[:2]
            
            x_center = (target_w - unit_w) // 2
            y_center = (target_h - unit_h) // 2
            
            # Вычисляем видимую область
            x1 = max(0, x_center)
            y1 = max(0, y_center)
            x2 = min(target_w, x_center + unit_w)
            y2 = min(target_h, y_center + unit_h)
            
            if x1 < x2 and y1 < y2:
                # Вычисляем область в юните
                u_x1 = max(0, -x_center)
                u_y1 = max(0, -y_center)
                u_x2 = u_x1 + (x2 - x1)
                u_y2 = u_y1 + (y2 - y1)
                
                unit_part = unit[u_y1:u_y2, u_x1:u_x2]
                
                # Накладываем с учетом альфа-канала
                if unit_part.shape[2] == 4:
                    alpha = unit_part[:, :, 3] / 255.0
                    unit_rgb = unit_part[:, :, :3]
                    
                    for c in range(3):
                        result[y1:y2, x1:x2, c] = \
                            (1 - alpha) * result[y1:y2, x1:x2, c] + \
                            alpha * unit_rgb[:, :, c]
                else:
                    result[y1:y2, x1:x2] = unit_part[:, :, :3]

            # РИСУЕМ МУСОРНЫЕ ОБЪЕКТЫ НА ФОНЕ
            
            if trash_images:
                num_trash = random.randint(trash_count_range[0], trash_count_range[1])
                
                for _ in range(num_trash):
                    # Выбираем случайный мусор
                    trash_path = random.choice(trash_images)
                    trash = cv2.imread(str(trash_path), cv2.IMREAD_UNCHANGED)
                    
                    if trash is None:
                        continue
                    
                    # Масштабируем мусор
                    trash_h, trash_w = trash.shape[:2]
                    scale_factor = random.uniform(trash_size_range[0], trash_size_range[1])
                    new_h = max(5, int(trash_h * scale_factor))
                    new_w = max(5, int(trash_w * scale_factor))
                    
                    if new_h > target_h or new_w > target_w:
                        continue
                    
                    scaled_trash = cv2.resize(trash, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    
                    # Случайная позиция (но не в центре, где будет юнит)
                    center_margin = min(target_h, target_w) // 4
                    x_pos = random.randint(0, target_w - new_w)
                    y_pos = random.randint(0, target_h - new_h)
                    
                    # Проверяем, не слишком близко к центру
                    center_x = target_w // 2
                    center_y = target_h // 2
                    obj_center_x = x_pos + new_w // 2
                    obj_center_y = y_pos + new_h // 2
                    
                    dist_to_center = np.sqrt((obj_center_x - center_x)**2 + (obj_center_y - center_y)**2)
                    if dist_to_center < center_margin:
                        continue  # пропускаем, слишком близко к центру
                    
                    # Накладываем мусор с учетом альфа-канала
                    if scaled_trash.shape[2] == 4:
                        alpha = scaled_trash[:, :, 3] / 255.0
                        trash_rgb = scaled_trash[:, :, :3]
                        
                        for c in range(3):
                            result[y_pos:y_pos+new_h, x_pos:x_pos+new_w, c] = \
                                (1 - alpha) * result[y_pos:y_pos+new_h, x_pos:x_pos+new_w, c] + \
                                alpha * trash_rgb[:, :, c]
                    else:
                        result[y_pos:y_pos+new_h, x_pos:x_pos+new_w] = scaled_trash[:, :, :3]
            
            
            # Сохраняем
            filename = f"{unit_name}_center_{i:04d}.png"
            output_path = unit_output_dir / filename
            cv2.imwrite(str(output_path), result)
            
            total_images += 1
            
            if (i + 1) % 10 == 0:
                print(f"  Создано {i + 1}/{images_per_unit}")
    
    print(f"\nГенерация завершена!")
    print(f"Всего изображений: {total_images}")
    print(f"Датасет сохранен в: {output_dir}")
        
    # Показываем структуру
    print("\nСтруктура датасета:")
    for unit_dir in output_dir.iterdir():
        if unit_dir.is_dir():
            num_files = len(list(unit_dir.glob("*.png")))
            print(f"  {unit_dir.name}/ - {num_files} изображений")

if __name__ == "__main__":
    # Использование
    # create_centered_dataset(
    #     base_path="KataCR\Clash-Royale-Detection-Dataset\images\segment",  # Замените на ваш путь
    #     output_dir="dataset_centered",
    #     image_size=(128, 128),
    #     images_per_unit=100
    # )

     create_centered_dataset(
        base_path= "cards_dataset\\dataset",  # Замените на ваш путь
        output_dir="dataset_cards",
        image_size=(150, 180),
        images_per_unit=20
    )
