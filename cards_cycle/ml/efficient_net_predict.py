import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import time

def load_trained_model(model_path, device, verbose=True):
    """
    Загрузка обученной модели
    """
    # Загружаем чекпоинт
    checkpoint = torch.load(model_path, map_location=device)
    
    # Получаем классы из чекпоинта
    classes = checkpoint['classes']
    num_classes = len(classes)
    model_name = checkpoint.get('model_name', 'efficientnet-b0')
    
    # Создаем модель
    if 'efficientnet-b0' in model_name:
        model = models.efficientnet_b0(weights=None)
    elif 'efficientnet-b1' in model_name:
        model = models.efficientnet_b1(weights=None)
    # ... добавьте другие версии по необходимости
    
    # Меняем классификатор
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, num_classes)
    
    # Загружаем веса
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    if verbose:
        print(f"✅ Модель загружена: {model_name}")
        print(f"📊 Количество классов: {num_classes}")
        print(f"🏷️  Классы: {classes}")
    
    return model, classes

import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import os

def predict_single_image(model, image_input, classes, device, img_size=224, verbose=True):
    """
    Предсказание для одного изображения с визуализацией
    
    Параметры:
    - model: модель PyTorch
    - image_input: путь к файлу (str) или изображение cv2 (numpy.ndarray)
    - classes: список названий классов
    - device: устройство (cuda/cpu)
    - img_size: размер входного изображения для модели
    - verbose: флаг для вывода подробной информации
    """
    # Трансформации для предсказания
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Загрузка и подготовка изображения
    try:
        # Проверяем тип входных данных
        if isinstance(image_input, str):
            # Если это путь к файлу
            image = Image.open(image_input).convert('RGB')
            original_image = image.copy()
            image_name = os.path.basename(image_input)
        elif isinstance(image_input, np.ndarray):
            # Если это cv2 изображение (numpy array)
            # Конвертируем BGR (cv2) в RGB (PIL)
            image_rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image_rgb)
            original_image = image.copy()
            image_name = "cv2_image"
        else:
            raise TypeError("image_input должен быть строкой (путь) или numpy.ndarray (cv2 изображение)")
            
    except Exception as e:
        print(f"❌ Ошибка загрузки изображения: {e}")
        return None
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Предсказание
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top_prob, top_class = probabilities.topk(5, dim=1)  # Топ-5 предсказаний
    
    # Результаты
    if verbose:
        print("\n" + "="*60)
        print(f"📷 Изображение: {image_name}")
        print("="*60)
        print("🏆 Топ-5 предсказаний:")
        print("-"*60)
    
    results = []
    for i in range(5):
        class_idx = top_class[0][i].item()
        prob = top_prob[0][i].item() * 100
        class_name = classes[class_idx]
        results.append((class_name, prob))
        if verbose:
            print(f"{i+1}. {class_name}: {prob:.2f}%")
            
    if verbose:
        print("="*60)
        
    # # Визуализация (если есть matplotlib)
    # try:
    #     import matplotlib.pyplot as plt
    #     import numpy as np
        
    #     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
    #     # Изображение
    #     ax1.imshow(original_image)
    #     ax1.set_title('Входное изображение')
    #     ax1.axis('off')
        
    #     # График вероятностей
    #     names = [r[0][:15] + '...' if len(r[0]) > 15 else r[0] for r in results]
    #     probs = [r[1] for r in results]
    #     colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(5)]
        
    #     ax2.barh(range(5), probs[::-1], color=colors[::-1])
    #     ax2.set_yticks(range(5))
    #     ax2.set_yticklabels(names[::-1])
    #     ax2.set_xlabel('Вероятность (%)')
    #     ax2.set_title('Вероятности классов')
    #     ax2.set_xlim(0, 100)
        
    #     plt.tight_layout()
    #     plt.show()
    # except:
    #     pass
    
    return results[0][0]  # возвращаем лучший класс

# Пример использования
if __name__ == "__main__":
    # Настройки
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'best_model_cards.pth'  # путь к вашей модели
    model, classes = load_trained_model(model_path, device, verbose=False)
    
    for image_path in  r'''C:\Users\aantr\cr_bot\cards_cycle\ml\test\Screenshot 2026-04-30 180607.png
C:\Users\aantr\cr_bot\cards_cycle\ml\test\Screenshot 2026-04-30 180730.png
C:\Users\aantr\cr_bot\cards_cycle\ml\test\Screenshot 2026-04-30 180824.png
C:\Users\aantr\cr_bot\cards_cycle\ml\test\Screenshot 2026-04-30 180829.png
C:\Users\aantr\cr_bot\cards_cycle\ml\test\Screenshot 2026-04-30 180832.png
C:\Users\aantr\cr_bot\cards_cycle\ml\test\Screenshot 2026-04-30 180836.png
C:\Users\aantr\cr_bot\cards_cycle\ml\test\Screenshot 2026-04-30 180843.png'''.split('\n'):
        
    
        # Загрузка модели
        
        # Предсказание
        predicted_class = predict_single_image(model, image_path.strip(), classes, device, verbose=True)
        print(predicted_class)