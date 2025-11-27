import cv2
import numpy as np
import gradio as gr
import time
from collections import deque

class AdvancedMotionDetector:
    def __init__(self, learning_rate=0.01, update_interval=30):
        # Инициализация методов вычитания фона
        self.backSub_mog2 = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=True
        )
        self.backSub_knn = cv2.createBackgroundSubtractorKNN(
            history=500, dist2Threshold=400, detectShadows=True
        )
        
        # Параметры для онлайн-обновления фона
        self.learning_rate = learning_rate  # Скорость обучения для обновления фона
        self.update_interval = update_interval  # Интервал обновления фона (в кадрах)
        self.frame_count = 0
        self.background_model = None
        self.background_initialized = False
        
        # Статистика для адаптивного обновления
        self.motion_intensity_history = deque(maxlen=100)
        self.stability_threshold = 0.1  # Порог стабильности сцены
        
        # Параметры обработки
        self.min_contour_area = 500
        self.movement_threshold = 50
        
        # История для сглаживания
        self.detection_points = deque(maxlen=100)
        
        # Цвета для визуализации
        self.colors = {
            'contour': (0, 255, 0),
            'centroid': (255, 0, 0),
            'trail': (0, 255, 255),
            'text': (255, 255, 255),
            'motion_area': (0, 0, 255),
            'background_update': (255, 255, 0)
        }

    def initialize_background_model(self, frame):
        """Инициализация модели фона"""
        self.background_model = frame.astype(np.float32)
        self.background_initialized = True
        print("Фоновая модель инициализирована")

    def update_background_model_adaptive(self, frame, motion_mask):
        """Адаптивное обновление фона на основе движения"""
        if not self.background_initialized:
            self.initialize_background_model(frame)
            return

        # Вычисление интенсивности движения
        motion_pixels = np.sum(motion_mask > 0)
        total_pixels = motion_mask.size
        motion_intensity = motion_pixels / total_pixels
        self.motion_intensity_history.append(motion_intensity)
        
        # Адаптивная скорость обучения
        adaptive_learning_rate = self.learning_rate
        
        # Если сцена стабильна (мало движения), увеличиваем скорость обучения
        if len(self.motion_intensity_history) > 10:
            avg_motion = np.mean(self.motion_intensity_history)
            if avg_motion < self.stability_threshold:
                adaptive_learning_rate = min(0.1, self.learning_rate * 2)
            else:
                adaptive_learning_rate = max(0.001, self.learning_rate * 0.5)

        # Обновление фона только в статических областях
        static_mask = (motion_mask == 0).astype(np.uint8)
        
        # Применяем морфологические операции для улучшения маски
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        static_mask = cv2.morphologyEx(static_mask, cv2.MORPH_CLOSE, kernel)
        
        # Обновляем фон только в статических областях
        background_update = cv2.accumulateWeighted(
            frame.astype(np.float32), 
            self.background_model, 
            adaptive_learning_rate,
            mask=static_mask
        )
        
        # Периодическое полное обновление
        if self.frame_count % self.update_interval == 0:
            # Используем медианный фильтр для устойчивости к шумам
            if len(self.motion_intensity_history) > 30:
                recent_motion = list(self.motion_intensity_history)[-30:]
                if np.mean(recent_motion) < self.stability_threshold:
                    # Если сцена стабильна, обновляем фон более агрессивно
                    self.background_model = cv2.accumulateWeighted(
                        frame.astype(np.float32), 
                        self.background_model, 
                        0.3
                    )

    def calculate_region_based_background(self, frame, grid_size=(8, 6)):
        """Обновление фона на основе региональных статистик"""
        h, w = frame.shape[:2]
        grid_h, grid_w = h // grid_size[0], w // grid_size[1]
        
        updated_background = self.background_model.copy()
        
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                y_start, y_end = i * grid_h, (i + 1) * grid_h
                x_start, x_end = j * grid_w, (j + 1) * grid_w
                
                # Избегаем выхода за границы
                y_end = min(y_end, h)
                x_end = min(x_end, w)
                
                region = frame[y_start:y_end, x_start:x_end]
                bg_region = self.background_model[y_start:y_end, x_start:x_end]
                
                # Вычисляем разницу между текущим кадром и фоном
                diff = cv2.absdiff(region.astype(np.uint8), bg_region.astype(np.uint8))
                region_motion = np.mean(diff)
                
                # Если регион стабилен, обновляем фон
                if region_motion < 15:  # Порог для стабильного региона
                    # Используем скользящее среднее для плавного обновления
                    alpha = 0.05  # Медленное обновление для стабильных регионов
                    updated_background[y_start:y_end, x_start:x_end] = (
                        alpha * region.astype(np.float32) + 
                        (1 - alpha) * bg_region
                    )
        
        return updated_background

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
        """Применение вычитания фона с онлайн-обновлением"""
        # Основное вычитание фона
        fg_mask_mog2 = self.backSub_mog2.apply(frame)
        fg_mask_knn = self.backSub_knn.apply(frame)
        
        # Комбинируем маски
        combined_mask = cv2.bitwise_or(fg_mask_mog2, fg_mask_knn)
        
        # Морфологические операции для улучшения маски
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        
        # Убираем тени
        _, combined_mask = cv2.threshold(combined_mask, 200, 255, cv2.THRESH_BINARY)
        
        # Онлайн-обновление фона
        self.frame_count += 1
        self.update_background_model_adaptive(frame, combined_mask)
        
        # Дополнительное вычитание с обновляемым фоном
        if self.background_initialized:
            bg_frame = self.background_model.astype(np.uint8)
            diff = cv2.absdiff(frame, bg_frame)
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            _, adaptive_mask = cv2.threshold(gray_diff, 25, 255, cv2.THRESH_BINARY)
            
            # Комбинируем с основной маской
            final_mask = cv2.bitwise_or(combined_mask, adaptive_mask)
            
            # Финальная морфологическая обработка
            final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
            return final_mask
        
        return combined_mask

    def find_contours(self, mask):
        """Поиск и фильтрация контуров"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_contours = []
        centers = []
        bounding_boxes = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area > self.min_contour_area:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Фильтрация по соотношению сторон
                aspect_ratio = w / h
                if 0.2 < aspect_ratio < 5.0:
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        valid_contours.append(contour)
                        centers.append((cx, cy))
                        bounding_boxes.append((x, y, w, h))
        
        return valid_contours, centers, bounding_boxes

    def draw_detection_results(self, frame, contours, centers, bounding_boxes, mask):
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
        
        # Отрисовка зон обновления фона (для отладки)
        if self.background_initialized:
            motion_pixels = np.sum(mask > 0)
            total_pixels = mask.size
            motion_intensity = motion_pixels / total_pixels
            
            # Отображаем информацию об обновлении фона
            update_info = [
                f"Frame: {self.frame_count}",
                f"Motion: {motion_intensity:.3f}",
                f"Learning rate: {self.learning_rate:.3f}",
                f"BG initialized: {self.background_initialized}"
            ]
            
            for i, text in enumerate(update_info):
                cv2.putText(result_frame, text, (10, 30 + i * 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['background_update'], 2)
        
        # Создание комбинированного изображения с маской
        mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        combined = np.hstack([result_frame, mask_colored])
        
        return combined

    def process_frame(self, frame):
        """Обработка одного кадра"""
        # Предварительная обработка
        processed_frame = self.preprocess_frame(frame)
        
        # Вычитание фона с онлайн-обновлением
        mask = self.apply_background_subtraction(processed_frame)
        
        # Поиск контуров
        contours, centers, bounding_boxes = self.find_contours(mask)
        
        # Отрисовка результатов
        result_frame = self.draw_detection_results(
            frame, contours, centers, bounding_boxes, mask
        )
        
        # Статистика
        stats = {
            'objects_detected': len(contours),
            'frame_count': self.frame_count,
            'background_initialized': self.background_initialized,
            'timestamp': time.time()
        }
        
        return result_frame, stats

    def reset_background_model(self):
        """Сброс модели фона"""
        self.background_initialized = False
        self.background_model = None
        self.frame_count = 0
        self.motion_intensity_history.clear()
        print("Модель фона сброшена")

def webcam_detection_with_online_update():
    """Обнаружение движущихся объектов с веб-камеры с онлайн-обновлением фона"""
    detector = AdvancedMotionDetector(learning_rate=0.01, update_interval=30)
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Ошибка: не удалось открыть веб-камеру")
        return
    
    print("Запуск обнаружения с онлайн-обновлением фона...")
    print("Управление:")
    print("  'q' - Выход")
    print("  'r' - Сброс модели фона")
    print("  '+' - Увеличить скорость обучения")
    print("  '-' - Уменьшить скорость обучения")
    print("  'b' - Показать/скрыть фон")
    
    show_background = False
    paused = False
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Обработка кадра
            processed_frame, stats = detector.process_frame(frame)
            
            # Отображение фона (если включено)
            if show_background and detector.background_initialized:
                bg_display = detector.background_model.astype(np.uint8)
                cv2.imshow('Background Model', bg_display)
            
            # Отображение результата
            cv2.imshow('Motion Detection with Online BG Update', processed_frame)
        
        # Обработка клавиш
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            detector.reset_background_model()
        elif key == ord('+'):
            detector.learning_rate = min(0.1, detector.learning_rate + 0.01)
            print(f"Скорость обучения увеличена: {detector.learning_rate:.3f}")
        elif key == ord('-'):
            detector.learning_rate = max(0.001, detector.learning_rate - 0.01)
            print(f"Скорость обучения уменьшена: {detector.learning_rate:.3f}")
        elif key == ord('b'):
            show_background = not show_background
            print(f"Отображение фона: {'включено' if show_background else 'выключено'}")
        elif key == ord('p'):
            paused = not paused
            print(f"Пауза: {'включена' if paused else 'выключена'}")
    
    cap.release()
    cv2.destroyAllWindows()

def create_gradio_interface():
    """Создание интерфейса Gradio с онлайн-обновлением"""
    detector = AdvancedMotionDetector()
    
    def process_video_with_online_update(video, learning_rate):
        """Обработка видео с онлайн-обновлением фона"""
        if video is None:
            return None
            
        # Обновление параметров
        detector.learning_rate = learning_rate
        
        # Временное сохранение видео
        temp_path = "temp_video.mp4"
        with open(temp_path, "wb") as f:
            f.write(video)
        
        # Обработка видео
        cap = cv2.VideoCapture(temp_path)
        processed_frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            processed_frame, stats = detector.process_frame(frame)
            processed_frames.append(processed_frame)
        
        cap.release()
        
        if len(processed_frames) == 0:
            return None
            
        # Возвращаем последний обработанный кадр
        return processed_frames[-1]
    
    # Создание интерфейса
    iface = gr.Interface(
        fn=process_video_with_online_update,
        inputs=[
            gr.Video(label="Входное видео"),
            gr.Slider(0.001, 0.1, value=0.01, label="Скорость обучения фона")
        ],
        outputs=gr.Image(label="Результат с онлайн-обновлением фона"),
        title="Обнаружение движущихся объектов с онлайн-обновлением фона",
        description="Загрузите видео для обнаружения с адаптивным обновлением фона на основе региональных статистик"
    )
    
    return iface

if __name__ == "__main__":
    # Запуск с веб-камерой
    # webcam_detection_with_online_update()
    
    # Или запуск Gradio интерфейса (раскомментировать при необходимости)
    iface = create_gradio_interface()
    iface.launch(share=True)