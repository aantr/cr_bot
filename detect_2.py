import sys
import cv2
import numpy as np
import gradio as gr
import matplotlib.pyplot as plt

class MotionDetector:
    def __init__(self):
        # Инициализация метода вычитания фона
        self.backSub = cv2.createBackgroundSubtractorMOG2(
              history=100, varThreshold=64, detectShadows=True
        )
        
        # Параметры для обработки
        self.min_contour_area = 10
        
    def process_video(self, vid_path):
        """Основная функция обработки видео"""
        cap = cv2.VideoCapture(vid_path)
        
        if not cap.isOpened():
            print("Error opening video file")
            return None
        
        # Список для хранения обработанных кадров
        processed_frames = []
        
        while cap.isOpened():
            # Захват кадр за кадром
            ret, frame = cap.read()
            if not ret:
                break
                
            # Вычитание фона
            fg_mask = self.backSub.apply(frame)
            
            # Установка глобального порога для удаления теней
            retval, mask_thresh = cv2.threshold(fg_mask, 180, 255, cv2.THRESH_BINARY)
            
            # Морфологические операции для удаления шума
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask_eroded = cv2.morphologyEx(mask_thresh, cv2.MORPH_OPEN, kernel)
            
            # Поиск контуров
            contours, hierarchy = cv2.findContours(mask_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Фильтрация контуров по площади
            large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > self.min_contour_area]
            
            # Рисование контуров и ограничивающих рамок
            frame_out = frame.copy()
            
            # Рисование контуров
            cv2.drawContours(frame_out, large_contours, -1, (0, 255, 0), 2)
            
            # Рисование ограничивающих рамок
            for cnt in large_contours:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame_out, (x, y), (x + w, y + h), (0, 0, 200), 3)
            
            # Добавление обработанного кадра в список
            processed_frames.append(frame_out)
            
            # Отображение для отладки (раскомментировать при необходимости)
            cv2.imshow('Frame', frame_out)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        return processed_frames
    
    def process_frame(self, frame):
        """Обработка одного кадра"""
        # Вычитание фона
        fg_mask = self.backSub.apply(frame)
        
        # Установка глобального порога для удаления теней
        retval, mask_thresh = cv2.threshold(fg_mask, 180, 255, cv2.THRESH_BINARY)
        
        # Морфологические операции для удаления шума
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_eroded = cv2.morphologyEx(mask_thresh, cv2.MORPH_OPEN, kernel)
        
        # Поиск контуров
        contours, hierarchy = cv2.findContours(mask_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Фильтрация контуров по площади
        large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > self.min_contour_area]
        
        # Рисование контуров и ограничивающих рамок
        frame_out = frame.copy()
        
        # Рисование контуров
        cv2.drawContours(frame_out, large_contours, -1, (0, 255, 0), 2)
        
        # Рисование ограничивающих рамок
        for cnt in large_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame_out, (x, y), (x + w, y + h), (0, 0, 200), 3)
        
        return frame_out

def create_gradio_interface():
    """Создание интерфейса Gradio"""
    detector = MotionDetector()
    
    def process_video_with_gradio(video):
        """Обработка видео для Gradio"""
        if video is None:
            return None
            
        # Временное сохранение видео
        temp_path = "temp_video.mp4"
        with open(temp_path, "wb") as f:
            f.write(video)
        
        # Обработка видео
        processed_frames = detector.process_video(temp_path)
        
        if processed_frames is None or len(processed_frames) == 0:
            return None
            
        # Возвращаем первый обработанный кадр для демонстрации
        return processed_frames[0]
    
    # Создание интерфейса
    iface = gr.Interface(
        fn=process_video_with_gradio,
        inputs=gr.Video(label="Входное видео"),
        outputs=gr.Image(label="Обнаруженные движущиеся объекты"),
        title="Обнаружение движущихся объектов с OpenCV",
        description="Загрузите видео для обнаружения движущихся объектов с помощью вычитания фона и обнаружения контуров"
    )
    
    return iface

def main():
    """Основная функция"""
    print("Обнаружение движущихся объектов с OpenCV")
    print("=" * 50)
    
    # Создание детектора
    detector = MotionDetector()
    
    # Обработка видео из файла
    video_path = sys.argv[1]  # Замените на путь к вашему видео
    
    try:
        processed_frames = detector.process_video(video_path)
        print(f"Обработано кадров: {len(processed_frames)}")
        
        # Сохранение результата (опционально)
        if len(processed_frames) > 0:
            cv2.imwrite("result_frame.jpg", processed_frames[0])
            print("Результат сохранен как 'result_frame.jpg'")
            
    except Exception as e:
        print(f"Ошибка при обработке видео: {e}")
        print("Создание демонстрационного интерфейса Gradio...")
        
        # Создание и запуск Gradio интерфейса
        iface = create_gradio_interface()
        iface.launch(share=True)

# Альтернативная версия с веб-камерой
def webcam_detection():
    """Обнаружение движущихся объектов с веб-камеры"""
    detector = MotionDetector()
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Ошибка: не удалось открыть веб-камеру")
        return
    
    print("Запуск обнаружения с веб-камеры...")
    print("Нажмите 'q' для выхода")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Обработка кадра
        processed_frame = detector.process_frame(frame)
        
        # Отображение результата
        cv2.imshow('Motion Detection - Webcam', processed_frame)
        
        # Выход по нажатию 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Запуск основного скрипта
    main()
    
    # Или запуск с веб-камерой (раскомментировать при необходимости)
    # webcam_detection()