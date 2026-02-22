from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import os
from sklearn.model_selection import train_test_split

class CardDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.classes = sorted(list(set(labels)))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.class_to_idx[self.labels[idx]]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def create_transforms():
    """Создание преобразований для обучения"""
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def prepare_data(dataset_path):
    """Подготовка данных для обучения"""
    images_dir = Path(dataset_path) / 'images'
    image_files = list(images_dir.glob('*.png'))
    
    image_paths = []
    labels = []
    
    for file_path in image_files:
        filename = file_path.stem
        if '_' in filename:
            card_name = filename.split('_')[0]
            image_paths.append(str(file_path))
            labels.append(card_name)
    
    # Разделение на train/val
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    train_transform, val_transform = create_transforms()
    
    train_dataset = CardDataset(train_paths, train_labels, train_transform)
    val_dataset = CardDataset(val_paths, val_labels, val_transform)
    
    return train_dataset, val_dataset

# Создание датасетов
train_dataset, val_dataset = prepare_data('cards_dataset')
print(f"Train samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Number of classes: {len(train_dataset.classes)}")
