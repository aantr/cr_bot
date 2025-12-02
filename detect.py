import cv2
import numpy as np
import time
from collections import deque
import argparse

class MotionDetector:
    def __init__(self):
        # Инициализация методов вычитания фона
        self.back_sub_mog2 = cv2.createBackgroundSubtractorMOG2(
            history=10, varThreshold=400, detectShadows=False
        )
        self.back_sub_knn = cv2.createBackgroundSubtractorKNN(
            history=10, dist2Threshold=100, detectShadows=False
        )
        
        # Параметры для обработки
        self.min_contour_area = 10
        self.movement_threshold = 10
        
        # История для сглаживания и отслеживания
        self.motion_history = deque(maxlen=10)
        self.detection_points = deque(maxlen=10)
        
        # Цвета для визуализации
        self.colors = {
            'contour': (0, 255, 0),        # Зеленый для контуров
            'centroid': (255, 0, 0),       # Синий для центроидов
            'trail': (0, 255, 255),        # Желтый для следов
            'text': (255, 255, 255),       # Белый для текста
            'motion_area': (0, 0, 255)     # Красный для зон движения
        }
        
    def preprocess_frame(self, frame):
        """Предварительная обработка кадра"""
        # Уменьшение шума
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        
        # Увеличение контраста
        lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0).apply(lab[:, :, 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def apply_background_subtraction(self, frame):
        """Применение вычитания фона"""
        # Используем MOG2 для основного обнаружения
        fg_mask_mog2 = self.back_sub_mog2.apply(frame)
        
        # Используем KNN для дополнительной проверки
        fg_mask_knn = self.back_sub_knn.apply(frame)
        
        # Комбинируем маски
        combined_mask = cv2.bitwise_or(fg_mask_mog2, fg_mask_knn)
        
        # Морфологические операции для улучшения маски
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        
        # Убираем тени (значение 127 в MOG2)
        _, combined_mask = cv2.threshold(combined_mask, 200, 255, cv2.THRESH_BINARY)
        
        return combined_mask
    
    def find_contours(self, mask):
        """Поиск и фильтрация контуров"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_contours = []
        centers = []
        bounding_boxes = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Фильтрация по площади
            if area > self.min_contour_area:
                # Вычисление ограничивающего прямоугольника
                x, y, w, h = cv2.boundingRect(contour)
                
                # Фильтрация по соотношению сторон (убираем слишком вытянутые объекты)
                aspect_ratio = w / h
                if 0.2 < aspect_ratio < 5.0:
                    # Вычисление центроида
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        valid_contours.append(contour)
                        centers.append((cx, cy))
                        bounding_boxes.append((x, y, w, h))
        
        return valid_contours, centers, bounding_boxes
    
    def analyze_movement(self, current_centers):
        """Анализ движения между кадрами"""
        if not hasattr(self, 'previous_centers'):
            self.previous_centers = []
            return []
        
        movements = []
        
        for curr_center in current_centers:
            min_distance = float('inf')
            closest_prev_center = None
            
            # Находим ближайший центр из предыдущего кадра
            for prev_center in self.previous_centers:
                distance = np.sqrt(
                    (curr_center[0] - prev_center[0])**2 + 
                    (curr_center[1] - prev_center[1])**2
                )
                
                if distance < min_distance:
                    min_distance = distance
                    closest_prev_center = prev_center
            
            if closest_prev_center and min_distance < self.movement_threshold:
                movements.append({
                    'from': closest_prev_center,
                    'to': curr_center,
                    'distance': min_distance,
                    'speed': min_distance  # Упрощенная скорость (пикселей/кадр)
                })
        
        self.previous_centers = current_centers.copy()
        return movements
    
    def draw_detection_results(self, frame, contours, centers, bounding_boxes, movements, mask):
        """Отрисовка результатов обнаружения"""
        result_frame = frame.copy()
        
        # Отрисовка контуров
        cv2.drawContours(result_frame, contours, -1, self.colors['contour'], 2)
        
        # Отрисовка центроидов и bounding box
        for center, bbox in zip(centers, bounding_boxes):
            x, y, w, h = bbox
            
            # Центроид
            cv2.circle(result_frame, center, 5, self.colors['centroid'], -1)
            
            # Bounding box
            cv2.rectangle(result_frame, (x, y), (x + w, y + h), self.colors['centroid'], 2)
            
            # Подпись с координатами
            cv2.putText(result_frame, f"({center[0]},{center[1]})", 
                       (center[0] + 10, center[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
        
        # Отрисовка движения
        for movement in movements:
            cv2.arrowedLine(result_frame, 
                          movement['from'], 
                          movement['to'], 
                          self.colors['trail'], 2, tipLength=0.3)
            
            # Отображение скорости
            mid_point = (
                (movement['from'][0] + movement['to'][0]) // 2,
                (movement['from'][1] + movement['to'][1]) // 2
            )
            cv2.putText(result_frame, f"{movement['speed']:.1f}", 
                       (mid_point[0], mid_point[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
        
        # Добавление точек в историю для следов
        for center in centers:
            self.detection_points.append(center)
        
        # Отрисовка следов движения
        for i in range(1, len(self.detection_points)):
            if self.detection_points[i - 1] is None or self.detection_points[i] is None:
                continue
            
            thickness = int(np.sqrt(64 / float(i + 1)) * 2)
            # cv2.line(result_frame, self.detection_points[i - 1], 
            #         self.detection_points[i], self.colors['trail'], thickness)
        
        # Создание комбинированного изображения с маской
        mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        combined = np.hstack([result_frame, mask_colored])
        
        return combined
    
    def process_frame(self, frame):
        """Основная обработка кадра"""
        # Предварительная обработка
        processed_frame = self.preprocess_frame(frame)
        
        # Вычитание фона
        mask = self.apply_background_subtraction(processed_frame)
        
        # Поиск контуров
        contours, centers, bounding_boxes = self.find_contours(mask)
        
        # Анализ движения
        movements = self.analyze_movement(centers)
        
        # Отрисовка результатов
        result_frame = self.draw_detection_results(
            frame, contours, centers, bounding_boxes, movements, mask
        )
        
        # Статистика
        stats = {
            'objects_detected': len(contours),
            'movements_detected': len(movements),
            'timestamp': time.time()
        }
        
        self.motion_history.append(stats)
        
        return result_frame, stats

def setup_camera(camera_id=0, width=640, height=480):
    """Настройка камеры"""
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap

def add_statistics_overlay(frame, stats, fps):
    """Добавление статистики на кадр"""
    overlay = frame.copy()
    
    # Полупрозрачный фон для текста
    h, w = frame.shape[:2]
    cv2.rectangle(overlay, (0, 0), (300, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Текст статистики
    texts = [
        f"FPS: {fps:.1f}",
        f"Objects: {stats['objects_detected']}",
        f"Movements: {stats['movements_detected']}",
        "Controls: [Q]uit, [R]eset BG, [S]ave",
        "[P]ause, [C]alibrate"
    ]
    
    for i, text in enumerate(texts):
        cv2.putText(frame, text, (10, 30 + i * 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    return frame

def main():
    parser = argparse.ArgumentParser(description='Motion Detection with OpenCV')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID')
    parser.add_argument('--video', type=str, help='Video file path')
    parser.add_argument('--min_area', type=int, default=500, help='Minimum contour area')
    parser.add_argument('--width', type=int, default=640, help='Frame width')
    parser.add_argument('--height', type=int, default=480, help='Frame height')
    
    args = parser.parse_args()
    
    # Инициализация детектора
    detector = MotionDetector()
    detector.min_contour_area = args.min_area
    
    # Настройка видеопотока
    if args.video:
        cap = cv2.VideoCapture(args.video)
    else:
        cap = setup_camera(args.camera, args.width, args.height)
    
    if not cap.isOpened():
        print("Error: Could not open video source")
        return
    
    print("Motion Detection started!")
    print("Controls:")
    print("  Q - Quit")
    print("  R - Reset background model")
    print("  P - Pause/Resume")
    print("  S - Save current frame")
    print("  C - Calibrate (wait 2 seconds for background stabilization)")
    
    # Переменные для FPS расчета
    fps_counter = 0
    fps_time = time.time()
    current_fps = 0
    
    paused = False
    calibration_mode = False
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("End of video stream")
                break
            
            # Калибровка - пропуск кадров для стабилизации фона
            if calibration_mode and fps_counter < 60:  # 2 секунды при 30 FPS
                fps_counter += 1
                continue
            elif calibration_mode:
                calibration_mode = False
                print("Calibration completed!")
            
            # Обработка кадра
            processed_frame, stats = detector.process_frame(frame)
            
            # Расчет FPS
            fps_counter += 1
            if time.time() - fps_time >= 1.0:
                current_fps = fps_counter
                fps_counter = 0
                fps_time = time.time()
            
            # Добавление статистики
            final_frame = add_statistics_overlay(processed_frame, stats, current_fps)
            
            # Отображение результата
            cv2.imshow('Motion Detection - OpenCV Background Subtraction', final_frame)
        
        # Обработка клавиш
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Сброс моделей фона
            detector.back_sub_mog2 = cv2.createBackgroundSubtractorMOG2()
            detector.back_sub_knn = cv2.createBackgroundSubtractorKNN()
            detector.detection_points.clear()
            print("Background models reset!")
        elif key == ord('p'):
            paused = not paused
            print("Paused" if paused else "Resumed")
        elif key == ord('s'):
            # Сохранение текущего кадра
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"motion_detection_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Frame saved as {filename}")
        elif key == ord('c'):
            # Калибровка
            calibration_mode = True
            fps_counter = 0
            print("Calibration started... please wait 2 seconds")
    
    cap.release()
    cv2.destroyAllWindows()
    print("Motion Detection stopped!")

if __name__ == "__main__":
    main()