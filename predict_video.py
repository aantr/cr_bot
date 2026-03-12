import torch
from ultralytics import YOLO
import cv2
from ultralytics import YOLO
import cv2

# import pandas as pd
from collections import defaultdict
from efficient_net_predict import load_trained_model, predict_single_image
from image2yolo import get_image_yolo_format

model = YOLO("runs/detect/cr_bot/train_bars8/weights/best.pt")
cap = cv2.VideoCapture("screenshot/rec2.mp4")

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Настройка записи видео


fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # или 'XVID' для avi
out = None
out_filename = 'screenshot/output_video.mp4'

classification_model_path = "best_model.pth"

# 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = 'best_model.pth'  # путь к вашей модели

# Загрузка модели
classification_model, classification_classes = load_trained_model(classification_model_path, device)

# Предсказание
# predicted_class = predict_single_image(classification_model, image_path, classification_classes, device)

# end preprocess class. model

# Для сбора статистики по кадрам
frame_stats = []
object_history = defaultdict(list)  # история позиций объектов

bars_place = defaultdict(list)
bar_for_level = defaultdict(int)
LEN_POSES = 5
SIZE_OF_RECT = 128
SIZE_RESCTRICTIONS = (10, 10), (50, 60)

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = get_image_yolo_format(frame)

    # Трекинг на текущем кадре
    # results = model.track(frame, persist=True, conf=0.5)
    results = model.track(
        frame,
        persist=True,  # maintain track IDs across frames
        conf=0.3,  # confidence threshold
        iou=0.5,  # IoU threshold for NMS
        tracker="bytetrack.yaml",  # tracking configuration
        project="detection_results",  # Папка для сохранения
        name="video_tracking",
    )

    # Собираем данные о кадре
    frame_data = {"frame_number": frame_count, "num_objects": 0, "objects": []}

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        confs = results[0].boxes.conf.cpu().numpy()
        class_ids = results[0].boxes.cls.int().cpu().tolist()

        frame_data["num_objects"] = len(track_ids)
        cls_text = ""

        for i, (box, track_id, conf, class_id) in enumerate(
            zip(boxes, track_ids, confs, class_ids)
        ):

            x1, y1, x2, y2 = box
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            # Ищем совпадающие бары и левелы
            if (
                class_id == 1
                and SIZE_RESCTRICTIONS[0][0] <= x2 - x1 <= SIZE_RESCTRICTIONS[1][0]
                and SIZE_RESCTRICTIONS[0][1] <= y2 - y1 <= SIZE_RESCTRICTIONS[1][1]
            ):

                bars_place[track_id].append((x1, y1, x2, y2))
                while len(bars_place[track_id]) > LEN_POSES:
                    bars_place[track_id].pop(0)

                if bar_for_level[track_id]:
                    pass
                rect = (int(x1), int(y1)), (
                    int(x2 + bar_for_level[track_id]),
                    int(y2),
                )
                cv2.rectangle(frame, *rect, (0, 0, 255), 3)

                cv2.rectangle(
                    frame,
                    (int((rect[0][0] + rect[1][0]) / 2 - SIZE_OF_RECT / 2), rect[1][1]),
                    (
                        int((rect[0][0] + rect[1][0]) / 2 + SIZE_OF_RECT / 2),
                        rect[1][1] + SIZE_OF_RECT,
                    ),
                    (255, 0, 0),
                    2,
                )

                blue_rect = frame[
                    rect[1][1] : rect[1][1] + SIZE_OF_RECT,
                    max(0, int((rect[0][0] + rect[1][0]) / 2 - SIZE_OF_RECT / 2)) : int(
                        (rect[0][0] + rect[1][0]) / 2 + SIZE_OF_RECT / 2
                    ),
                ]

                predicted_class = predict_single_image(classification_model, 
                                                       blue_rect, classification_classes, device, verbose=False)
            
                cls_text = predicted_class
            else:
                cls_text = 'None'
            if class_id == 0:
                x_left = x1
                y_left = center_y
                for key, level_bar in bars_place.items():
                    count_good_pos = 0
                    sum_len = 0
                    for pos in level_bar:
                        x_level1, y_level1, x_level2, y_level2 = pos

                        if (
                            y_level1 <= y_left <= y_level2
                            and (x_level1 + x_level2) / 2
                            <= x_left
                            <= (x_level1 + x_level2) / 2 + x_level2 - x_level1
                        ):

                            count_good_pos += 1
                            sum_len += x2 - x1

                    if len(level_bar) >= LEN_POSES and count_good_pos > LEN_POSES / 2:
                        bar_for_level[key] = sum_len / count_good_pos

            # Данные объекта
            obj_data = {
                "track_id": track_id,
                "class": model.names[class_id],
                "confidence": conf,
                "bbox": [x1, y1, x2, y2],
                "center": [center_x, center_y],
            }
            frame_data["objects"].append(obj_data)

            # Сохраняем в историю для анализа траекторий
            object_history[track_id].append(
                {
                    "frame": frame_count,
                    "center": (center_x, center_y),
                    "bbox": (x1, y1, x2, y2),
                }
            )

            # Визуализация с дополнительной информацией
            # cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label = f"ID:{track_id} {conf:.2f} cls {cls_text}"
            cv2.putText(
                frame,
                label,
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                3,
            )

    # Показываем номер кадра
    cv2.putText(
        frame,
        f"Frame: {frame_count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )
    height, width = frame.shape[:2]
    new_width = width // 2
    new_height = height // 2
    resized_frame = cv2.resize(frame, (new_width, new_height))
    cv2.imshow("Tracking", resized_frame)
    # Внутри цикла, после обработки кадра (перед cv2.imshow)
    if out is None:
        out = cv2.VideoWriter(out_filename, fourcc, fps, (width, height))
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()
out.release()

# Сохраняем статистику
# df = pd.DataFrame(frame_stats)
# df.to_csv('tracking_stats.csv', index=False)
