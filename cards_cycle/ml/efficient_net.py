import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

import os
import shutil
import random
from sklearn.model_selection import train_test_split

def split_dataset(source_dir, target_dir='split_dataset', val_size=0.2, random_seed=42):
    """
    Автоматическое разделение датасета на train/val
    
    Args:
        source_dir: исходная папка с классами (dataset/class1/, dataset/class2/, ...)
        target_dir: целевая папка для разделенного датасета
        val_size: доля валидационной выборки (0.2 = 20%)
        random_seed: для воспроизводимости
    """
    
    print(f"Разделение датасета из {source_dir}")
    print(f"Валидационная выборка: {val_size*100}%")
    
    # Создаем структуру папок
    train_dir = os.path.join(target_dir, 'train')
    val_dir = os.path.join(target_dir, 'val')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Получаем список классов
    classes = [d for d in os.listdir(source_dir) 
               if os.path.isdir(os.path.join(source_dir, d))]
    
    print(f"Найдено классов: {len(classes)}")
    
    for class_name in classes:
        class_path = os.path.join(source_dir, class_name)
        images = [f for f in os.listdir(class_path) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))]
        
        if len(images) == 0:
            print(f"  ⚠️ В классе {class_name} нет изображений")
            continue
            
        # Разделяем на train/val
        train_images, val_images = train_test_split(
            images, 
            test_size=val_size, 
            random_state=random_seed
        )
        
        print(f"  📁 {class_name}: всего {len(images)}, "
              f"train: {len(train_images)}, val: {len(val_images)}")
        
        # Создаем папки для класса
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
        
        # Копируем файлы (используйте shutil.move() если хотите переместить)
        for img in train_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(train_dir, class_name, img)
            shutil.copy2(src, dst)  # copy2 сохраняет метаданные
            
        for img in val_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(val_dir, class_name, img)
            shutil.copy2(src, dst)
    
    print(f"\n✅ Датасет разделен и сохранен в {target_dir}")
    print(f"   train: {os.path.join(target_dir, 'train')}")
    print(f"   val: {os.path.join(target_dir, 'val')}")
    
    return train_dir, val_dir

def prepare_data_from_single_folder(data_path, batch_size=32, img_size=224, 
                                   val_split=0.2, use_split_cache=True):
    """
    Подготовка данных из единой папки с классами
    
    Args:
        data_path: путь к папке с классами (dataset/)
        batch_size: размер батча
        img_size: размер изображения
        val_split: доля валидационной выборки
        use_split_cache: использовать ли кэш разделенного датасета
    """
    
    if use_split_cache:
        # Создаем временную папку для разделенного датасета
        cache_dir = os.path.join(os.path.dirname(data_path), 'split_dataset_cache')
        print('data path', data_path)
        # Проверяем, есть ли уже разделенный датасет
        if os.path.exists(cache_dir) and len(os.listdir(cache_dir)) > 0 and False:
            print(f"🔄 Используем кэшированный разделенный датасет из {cache_dir}")
            train_dir = os.path.join(cache_dir, 'train')
            val_dir = os.path.join(cache_dir, 'val')
        else:
            # Разделяем датасет
            train_dir, val_dir = split_dataset(
                source_dir=data_path,
                target_dir=cache_dir,
                val_size=val_split
            )
    else:
        # Создаем временную папку и сразу удалим после использования
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            train_dir, val_dir = split_dataset(
                source_dir=data_path,
                target_dir=tmpdir,
                val_size=val_split
            )
    
    # Трансформации для тренировочных данных
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Трансформации для валидационных данных
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    print('train dir', train_dir)
    # Загрузка данных
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    
    # Создание загрузчиков
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"\n📊 Статистика датасета:")
    print(f"   Найдено классов: {len(train_dataset.classes)}")
    print(f"   Тренировочных изображений: {len(train_dataset)}")
    print(f"   Валидационных изображений: {len(val_dataset)}")
    print(f"   Соотношение train/val: {len(train_dataset)/(len(train_dataset)+len(val_dataset))*100:.1f}/{len(val_dataset)/(len(train_dataset)+len(val_dataset))*100:.1f}%")
    
    if len(train_dataset.classes) <= 10:
        print(f"   Классы: {train_dataset.classes}")
    
    return train_loader, val_loader, train_dataset.classes

def create_model(num_classes, model_name='efficientnet-b0', pretrained=True):
    """
    Создание модели EfficientNet с заморозкой первых слоев
    """
    
    # Словарь соответствия имен моделей
    efficientnet_models = {
        'efficientnet-b0': models.efficientnet_b0,
        'efficientnet-b1': models.efficientnet_b1,
        'efficientnet-b2': models.efficientnet_b2,
        'efficientnet-b3': models.efficientnet_b3,
        'efficientnet-b4': models.efficientnet_b4,
        'efficientnet-b5': models.efficientnet_b5,
        'efficientnet-b6': models.efficientnet_b6,
        'efficientnet-b7': models.efficientnet_b7,
    }
    
    # Загрузка предобученной модели
    if pretrained:
        weights = 'DEFAULT'
    else:
        weights = None
    
    model = efficientnet_models[model_name](weights=weights)
    
    # Заморозка первых слоев (опционально)
    # for param in model.features.parameters():
    #     param.requires_grad = False
    
    # Замена классификатора под наше количество классов
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, num_classes)
    
    return model


def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Одна эпоха обучения
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for inputs, labels in pbar:
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
        
        pbar.set_postfix({'loss': loss.item(), 'acc': 100.*correct/total})
    
    return running_loss / len(train_loader), 100. * correct / total

def validate(model, val_loader, criterion, device):
    """
    Валидация модели
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc='Validation'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(val_loader), 100. * correct / total

def check_class_distribution(data_path):
    """
    Проверка распределения изображений по классам
    """
    print("📊 Распределение по классам:")
    print("-" * 50)
    
    classes = [d for d in os.listdir(data_path) 
               if os.path.isdir(os.path.join(data_path, d))]
    
    distribution = []
    for class_name in classes:
        class_path = os.path.join(data_path, class_name)
        images = [f for f in os.listdir(class_path) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        distribution.append(len(images))
        print(f"   {class_name}: {len(images)} изображений")
    
    print("-" * 50)
    print(f"   Всего классов: {len(classes)}")
    print(f"   Всего изображений: {sum(distribution)}")
    print(f"   Среднее: {sum(distribution)/len(distribution):.1f}")
    print(f"   Мин: {min(distribution)}, Макс: {max(distribution)}")
    
    # Проверка на дисбаланс
    if max(distribution) / min(distribution) > 10:
        print("\n⚠️  Обнаружен сильный дисбаланс классов!")
        print("   Рекомендуется использовать weighted sampler или аугментацию")
    
    return distribution

def main():
    # Параметры
    data_path = 'dataset_cards'  # Папка с классами
    model_output = 'best_model_cards.pth'

    # data_path = 'dataset_centered'  # Папка с классами
    # model_output = 'best_model.pth'
    
    batch_size = 32
    num_epochs = 30
    learning_rate = 0.001
    model_name = 'efficientnet-b0'
    val_split = 0.2  # 20% данных на валидацию
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"🚀 Запуск обучения EfficientNet")
    print(f"   Модель: {model_name}")
    print(f"   Устройство: {device}")
    print(f"   Размер батча: {batch_size}")
    print(f"   Эпох: {num_epochs}")
    print(f"   Разделение train/val: {80}/{val_split*100}%\n")
    
    # Подготовка данных (автоматическое разделение)
    train_loader, val_loader, classes = prepare_data_from_single_folder(
        data_path=data_path,
        batch_size=batch_size,
        img_size=224,  # для b0
        val_split=val_split,
        use_split_cache=True  # сохраняем разделенный датасет для повторного использования
    )
    
    num_classes = len(classes)
    
    # Создание модели
    model = create_model(num_classes, model_name, pretrained=True)
    model = model.to(device)
    
    # Выводим информацию о модели
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📦 Параметры модели:")
    print(f"   Всего: {total_params:,}")
    print(f"   Обучаемых: {trainable_params:,}")
    
    # Функция потерь и оптимизатор
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.1
    )
    
    # TensorBoard
    writer = SummaryWriter(f'runs/efficientnet_{model_name}_{len(classes)}classes')
    
    # Лучшая модель
    best_val_acc = 0.0
    
    # Обучение
    for epoch in range(num_epochs):
        print(f'\n📅 Эпоха {epoch+1}/{num_epochs}')
        print('=' * 60)
        
        # Обучение
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Валидация
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Обновление learning rate
        scheduler.step(val_loss)
        
        # Логирование
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        print(f'\n📊 Результаты эпохи:')
        print(f'   Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'   Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'   LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Сохранение лучшей модели
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
                'classes': classes,
                'model_name': model_name
            }, model_output)
            print(f'   💾 Сохранена лучшая модель! (acc: {val_acc:.2f}%)')
    
    writer.close()
    print(f'\n🎉 Обучение завершено!')
    print(f'   Лучшая точность на валидации: {best_val_acc:.2f}%')
    print(f'   Модель сохранена в {model_output}')
    
    # Сохраняем информацию о классах
    with open('classes.txt', 'w') as f:
        for class_name in classes:
            f.write(f"{class_name}\n")
    print(f'   Список классов сохранен в classes.txt')

def predict_image(model, image_path, classes, device, img_size=224):
    """
    Предсказание для одного изображения
    """
    from PIL import Image
    
    # Трансформации для предсказания
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Загрузка и подготовка изображения
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Предсказание
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top_prob, top_class = probabilities.topk(3, dim=1)
    
    # Результаты
    print("\nТоп-3 предсказания:")
    for i in range(3):
        class_idx = top_class[0][i].item()
        prob = top_prob[0][i].item()
        print(f"{i+1}. {classes[class_idx]}: {prob:.2%}")
    
    return top_class[0][0].item()

if __name__ == '__main__':
    main()