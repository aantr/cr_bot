import cv2
import numpy as np

def feature_based_motion_detection():
    cap = cv2.VideoCapture('screenshot/IMG_0694.MP4')
    
    # Параметры для Shi-Tomasi угловых признаков
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    
    # Параметры для Lucas-Kanade
    lk_params = dict(winSize=(15,15), maxLevel=2, 
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    
    # Находим начальные точки для отслеживания
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Вычисляем оптический поток для конкретных точек
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        
        if p1 is not None:
            # Выбираем хорошие точки
            good_new = p1[st==1]
            good_old = p0[st==1]
            
            # Рисуем треки
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                
                # Определяем движущиеся точки (длина вектора > порога)
                movement = np.sqrt((a-c)**2 + (b-d)**2)
                if movement > 2.0:  # Порог движения
                    cv2.circle(frame, (int(a), int(b)), 5, (0, 255, 0), -1)
                    cv2.line(frame, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
            
            # Кластеризуем движущиеся точки для обнаружения объектов
            if len(good_new) > 0:
                # Простая группировка по близости
                from sklearn.cluster import DBSCAN
                
                clustering = DBSCAN(eps=50, min_samples=3).fit(good_new)
                labels = clustering.labels_
                
                # Рисуем bounding boxes вокруг кластеров
                unique_labels = set(labels)
                for label in unique_labels:
                    if label != -1:  # Игнорируем шум
                        cluster_points = good_new[labels == label]
                        if len(cluster_points) > 2:
                            x, y, w, h = cv2.boundingRect(cluster_points)
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        cv2.imshow('Feature Tracking', frame)
        
        # Обновляем точки для следующего кадра
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
        
        # Периодически обновляем точки для отслеживания
        if len(p0) < 25:
            p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

feature_based_motion_detection()