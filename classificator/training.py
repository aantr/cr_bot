import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
from PIL import Image
import os
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.spatial.distance import cityblock
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Пути к данным
train_data_dir = 'dataset/train'  # Папка с тренировочными данными

dataset_folder = 'cards_dataset/images'
input_image_path = 'examples/example5.png'

# Преобразования для тренировки
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.RandomHorizontalFlip(p=0.5),
    # transforms.RandomRotation(10),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Преобразования для инференса
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Создание модели
class CustomResNet(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(CustomResNet, self).__init__()
        self.resnet = models.resnet50(pretrained=pretrained)
        # Сохраняем размер последнего слоя перед заменой
        num_ftrs = self.resnet.fc.in_features
        # Заменяем последний слой для нужного количества классов
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)
        
    def forward(self, x):
        return self.resnet(x)
    
    def get_features(self, x):
        # Получаем признаки до последнего слоя
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        return x

# Функция обучения модели
def train_model(data_dir, num_epochs=10, batch_size=32, learning_rate=0.001):
    print("Загрузка данных для обучения...")
    
    # Создание датасета
    train_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    # Получаем количество классов
    num_classes = len(train_dataset.classes)
    print(f"Найдено классов: {num_classes}")
    print(f"Размер обучающей выборки: {len(train_dataset)}")
    
    # Создание модели
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomResNet(num_classes=num_classes, pretrained=True)
    model = model.to(device)
    
    # Оптимизатор и функция потерь
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Обучение
    print("Начало обучения...")
    model.train()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        scheduler.step()
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        print(f'Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}%')
    
    print("Обучение завершено!")
    return model, train_dataset.classes

# Функция извлечения признаков
def extract_features(model, img_path, device):
    try:
        model.eval()
        img = Image.open(img_path).convert('RGB')
        img_t = eval_transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            features = model.get_features(img_t)
        return features.cpu().numpy().flatten()
    except Exception as e:
        print(f"Ошибка при обработке {img_path}: {e}")
        return None

# Функция поиска похожих изображений
def find_most_similar(model, input_feat, dataset_feats, image_paths, device):
    results = {}
    
    # Косинусное сходство
    cos_sim = cosine_similarity([input_feat], dataset_feats)[0]
    best_cos_idx = np.argmax(cos_sim)
    results['cosine'] = {
        'index': best_cos_idx,
        'score': cos_sim[best_cos_idx],
        'path': image_paths[best_cos_idx]
    }
    
    # Евклидово расстояние
    eucl_dist = euclidean_distances([input_feat], dataset_feats)[0]
    best_eucl_idx = np.argmin(eucl_dist)
    results['euclidean'] = {
        'index': best_eucl_idx,
        'score': eucl_dist[best_eucl_idx],
        'path': image_paths[best_eucl_idx]
    }
    
    # Манхэттенское расстояние
    manhattan_dist = []
    for feat in dataset_feats:
        dist = cityblock(input_feat, feat)
        manhattan_dist.append(dist)
    manhattan_dist = np.array(manhattan_dist)
    best_manh_idx = np.argmin(manhattan_dist)
    results['manhattan'] = {
        'index': best_manh_idx,
        'score': manhattan_dist[best_manh_idx],
        'path': image_paths[best_manh_idx]
    }
    
    return results

# Основная функция
def main():

    load = True
    if not load:
        # Обучение модели
        print("="*60)
        print("ЭТАП 1: ОБУЧЕНИЕ МОДЕЛИ")
        print("="*60)
        
        trained_model, class_names = train_model(
            data_dir=train_data_dir,
            num_epochs=5,      # Можно увеличить
            batch_size=16,     # Можно изменить в зависимости от памяти
            learning_rate=0.001
        )
        
        # Сохранение модели
        torch.save(trained_model.state_dict(), 'custom_resnet_model.pth')
        print("Модель сохранена как 'custom_resnet_model.pth'")

    else:
        print("="*60)
        print("ЭТАП 1: Загрузка МОДЕЛИ")
        print("="*60)

        train_dataset = datasets.ImageFolder(root=train_data_dir, transform=train_transform)
        num_classes = len(train_dataset.classes)
        trained_model = load_trained_model('custom_resnet_model.pth', num_classes=num_classes)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ЭТАП 2: Поиск похожих изображений
    print("\n" + "="*60)
    print("ЭТАП 2: ПОИСК ПОХОЖИХ ИЗОБРАЖЕНИЙ")
    print("="*60)
    
    # Извлечение признаков для входного изображения
    print("Извлечение признаков для входного изображения...")
    input_features = extract_features(trained_model, input_image_path, device)
    if input_features is None:
        raise ValueError("Не удалось извлечь признаки для входного изображения")
    
    # Извлечение признаков для всех изображений в датасете поиска
    print("Извлечение признаков для датасета поиска...")
    features_list = []
    image_paths = []
    
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    
    for filename in tqdm(os.listdir(dataset_folder)):
        if filename.lower().endswith(valid_extensions):
            path = os.path.join(dataset_folder, filename)
            features = extract_features(trained_model, path, device)
            if features is not None:
                features_list.append(features)
                image_paths.append(path)
    
    if len(features_list) == 0:
        raise ValueError("Не удалось обработать ни одно изображение из датасета")
    
    features_list = np.array(features_list)
    print(f"Обработано {len(features_list)} изображений")
    
    # Поиск наиболее похожих изображений
    print("Поиск наиболее похожих изображений...")
    similar_results = find_most_similar(trained_model, input_features, features_list, image_paths, device)
    
    # Вывод результатов
    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ ПОИСКА НАИБОЛЕЕ ПОХОЖИХ ИЗОБРАЖЕНИЙ")
    print("="*60)
    
    print(f"\nВходное изображение: {input_image_path}")
    print(f"Количество изображений в датасете: {len(image_paths)}")
    
    print("\n1. КОСИНУСНОЕ СХОДСТВО (лучшее совпадение):")
    print(f"   Путь: {similar_results['cosine']['path']}")
    print(f"   Оценка: {similar_results['cosine']['score']:.4f}")
    
    print("\n2. ЕВКЛИДОВО РАССТОЯНИЕ (наименьшее расстояние):")
    print(f"   Путь: {similar_results['euclidean']['path']}")
    print(f"   Расстояние: {similar_results['euclidean']['score']:.4f}")
    
    print("\n3. МАНХЭТТЕНСКОЕ РАССТОЯНИЕ (наименьшее расстояние):")
    print(f"   Путь: {similar_results['manhattan']['path']}")
    print(f"   Расстояние: {similar_results['manhattan']['score']:.4f}")
    
    # Топ-5 по косинусному сходству
    cos_scores = cosine_similarity([input_features], features_list)[0]
    top5_cos_indices = np.argsort(cos_scores)[-5:][::-1]
    print("\nТоп-5 по косинусному сходству:")
    for i, idx in enumerate(top5_cos_indices, 1):
        print(f"   {i}. {os.path.basename(image_paths[idx])} - оценка: {cos_scores[idx]:.4f}")

# Функция для загрузки предобученной модели (если уже есть)
def load_trained_model(model_path, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomResNet(num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    return model

if __name__ == "__main__":
    main()
