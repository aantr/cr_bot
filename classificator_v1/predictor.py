import torch
from PIL import Image
import torch.nn.functional as F

class CardClassifier:
    def __init__(self, model_path, class_names, device=None):
        """Инициализация классификатора"""
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = class_names
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Загрузка модели
        self.model = create_model(len(class_names))
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, image_path, top_k=5):
        """Предсказание класса изображения"""
        # Загрузка и предобработка изображения
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Предсказание
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            top_probs, top_indices = torch.topk(probabilities, top_k)
        
        # Формирование результата
        results = []
        for i in range(top_k):
            class_idx = top_indices[0][i].item()
            prob = top_probs[0][i].item()
            class_name = self.class_names[class_idx]
            results.append((class_name, prob))
        
        return results
    
    def predict_single(self, image_path):
        """Предсказание одного наиболее вероятного класса"""
        results = self.predict(image_path, top_k=1)
        return results[0] if results else None

# Использование классификатора
def main():
    # Создание классификатора
    classifier = CardClassifier(
        model_path='best_card_classifier.pth',
        class_names=train_dataset.classes
    )
    
    # Пример классификации нового изображения
    test_image_path = 'path/to/new/card/image.png'
    
    # Получение топ-5 предсказаний
    predictions = classifier.predict(test_image_path, top_k=5)
    
    print("Top 5 predictions:")
    for i, (class_name, probability) in enumerate(predictions, 1):
        print(f"{i}. {class_name}: {probability:.4f} ({probability*100:.2f}%)")
    
    # Получение наиболее вероятного класса
    best_prediction = classifier.predict_single(test_image_path)
    if best_prediction:
        class_name, probability = best_prediction
        print(f"\nMost likely class: {class_name} (confidence: {probability:.4f})")

if __name__ == "__main__":
    main()
