import torch
from torchvision import models, transforms
from PIL import Image
import os
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.spatial.distance import cityblock  # Manhattan distance
import numpy as np
from tqdm import tqdm

# Путь к папке с изображениями
dataset_folder = 'classificator/cards_dataset/images'
input_image_path = 'classificator/examples/example2.png'

# Преобразования для изображений
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Загрузка предобученной модели ResNet
model = models.resnet152(pretrained=True)
# models.resnet18
# Удаляем последний слой классификации для получения признаков
model = torch.nn.Sequential(*(list(model.children())[:-1]))
model.eval()

# Функция извлечения признаков
def extract_features(img_path):
    try:
        img = Image.open(img_path).convert('RGB')
        img_t = transform(img).unsqueeze(0)
        with torch.no_grad():
            features = model(img_t)
        return features.squeeze().numpy()
    except Exception as e:
        print(f"Ошибка при обработке {img_path}: {e}")
        return None

# Извлечение признаков для входного изображения
print("Извлечение признаков для входного изображения...")
input_features = extract_features(input_image_path)
if input_features is None:
    raise ValueError("Не удалось извлечь признаки для входного изображения")

# Извлечение признаков для всех изображений в датасете
print("Извлечение признаков для датасета...")
features_list = []
image_paths = []

for filename in tqdm(os.listdir(dataset_folder)):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        path = os.path.join(dataset_folder, filename)
        features = extract_features(path)
        if features is not None:
            features_list.append(features)
            image_paths.append(path)

features_list = np.array(features_list)
print(f"Обработано {len(features_list)} изображений")

# Функция для поиска наиболее похожих изображений по разным метрикам
def find_most_similar(input_feat, dataset_feats, image_paths):
    results = {}
    
    # 1. Косинусное сходство (чем больше, тем лучше)
    # cos_sim = cosine_similarity([input_feat], dataset_feats)[0]
    # best_cos_idx = np.argmax(cos_sim)
    # results['cosine'] = {
    #     'index': best_cos_idx,
    #     'score': cos_sim[best_cos_idx],
    #     'path': image_paths[best_cos_idx]
    # }
    
    # 2. Евклидово расстояние (чем меньше, тем лучше)
    eucl_dist = euclidean_distances([input_feat], dataset_feats)[0]
    best_eucl_idx = np.argmin(eucl_dist)
    results['euclidean'] = {
        'index': best_eucl_idx,
        'score': eucl_dist[best_eucl_idx],
        'path': image_paths[best_eucl_idx]
    }
    
    # 3. Манхэттенское расстояние (L1) (чем меньше, тем лучше)
    # manhattan_dist = []
    # for feat in dataset_feats:
    #     dist = cityblock(input_feat, feat)
    #     manhattan_dist.append(dist)
    # manhattan_dist = np.array(manhattan_dist)
    # best_manh_idx = np.argmin(manhattan_dist)
    # results['manhattan'] = {
    #     'index': best_manh_idx,
    #     'score': manhattan_dist[best_manh_idx],
    #     'path': image_paths[best_manh_idx]
    # }
    
    # 4. Корреляционное сходство (чем больше, тем лучше)
    def correlation_similarity(a, b):
        return np.corrcoef(a, b)[0, 1]
    
    corr_scores = []
    for feat in dataset_feats:
        corr = correlation_similarity(input_feat, feat)
        corr_scores.append(corr)
    corr_scores = np.array(corr_scores)
    best_corr_idx = np.argmax(corr_scores)
    results['correlation'] = {
        'index': best_corr_idx,
        'score': corr_scores[best_corr_idx],
        'path': image_paths[best_corr_idx]
    }
    
    return results

# Поиск наиболее похожих изображений
print("Поиск наиболее похожих изображений...")
similar_results = find_most_similar(input_features, features_list, image_paths)

# Вывод результатов
print("\n" + "="*50)
print("РЕЗУЛЬТАТЫ ПОИСКА НАИБОЛЕЕ ПОХОЖИХ ИЗОБРАЖЕНИЙ")
print("="*50)

print(f"\nВходное изображение: {input_image_path}")
print(f"Количество изображений в датасете: {len(image_paths)}")

# print("\n1. КОСИНУСНОЕ СХОДСТВО (лучшее совпадение):")
# print(f"   Путь: {similar_results['cosine']['path']}")
# print(f"   Оценка: {similar_results['cosine']['score']:.4f}")

print("\n2. ЕВКЛИДОВО РАССТОЯНИЕ (наименьшее расстояние):")
print(f"   Путь: {similar_results['euclidean']['path']}")
print(f"   Расстояние: {similar_results['euclidean']['score']:.4f}")

# print("\n3. МАНХЭТТЕНСКОЕ РАССТОЯНИЕ (наименьшее расстояние):")
# print(f"   Путь: {similar_results['manhattan']['path']}")
# print(f"   Расстояние: {similar_results['manhattan']['score']:.4f}")

print("\n4. КОРРЕЛЯЦИОННОЕ СХОДСТВО (наивысшая корреляция):")
print(f"   Путь: {similar_results['correlation']['path']}")
print(f"   Корреляция: {similar_results['correlation']['score']:.4f}")

# Дополнительно: ранжирование всех изображений по каждой метрике
print("\n" + "="*50)
print("ТОП-5 ИЗОБРАЖЕНИЙ ПО КАЖДОЙ МЕТРИКЕ")
print("="*50)

# Топ-5 по косинусному сходству
# cos_scores = cosine_similarity([input_features], features_list)[0]
# top5_cos_indices = np.argsort(cos_scores)[-5:][::-1]
# print("\nТоп-5 по косинусному сходству:")
# for i, idx in enumerate(top5_cos_indices, 1):
#     print(f"   {i}. {os.path.basename(image_paths[idx])} - оценка: {cos_scores[idx]:.4f}")

# Топ-5 по евклидову расстоянию
eucl_distances = euclidean_distances([input_features], features_list)[0]
top5_eucl_indices = np.argsort(eucl_distances)[:5]
print("\nТоп-5 по евклидову расстоянию (меньше = лучше):")
for i, idx in enumerate(top5_eucl_indices, 1):
    print(f"   {i}. {os.path.basename(image_paths[idx])} - расстояние: {eucl_distances[idx]:.4f}")

# # Топ-5 по манхэттенскому расстоянию
# manh_distances = [cityblock(input_features, feat) for feat in features_list]
# top5_manh_indices = np.argsort(manh_distances)[:5]
# print("\nТоп-5 по манхэттенскому расстоянию (меньше = лучше):")
# for i, idx in enumerate(top5_manh_indices, 1):
#     print(f"   {i}. {os.path.basename(image_paths[idx])} - расстояние: {manh_distances[idx]:.4f}")
