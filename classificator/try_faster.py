import torch
from torchvision import models, transforms
from PIL import Image
import os
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances  # Изменено
import faiss
from concurrent.futures import ThreadPoolExecutor
import pickle
from tqdm import tqdm
import argparse

class FastImageSearcher:
    def __init__(self, use_gpu=False, use_faiss=True, metric='cosine'):  # Добавлен параметр metric
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.use_faiss = use_faiss
        self.metric = metric  # 'cosine' или 'euclidean'
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        
        self._init_model()
        self.features = None
        self.image_paths = []
        self.index = None
        
    def _init_model(self):
        print("Загрузка модели ResNet50...")
        self.model = models.resnet50(pretrained=True)
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
    def extract_feature(self, img_path):
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                feature = self.model(img_tensor)
                feature = torch.flatten(feature).cpu().numpy()
            
            return feature
        except Exception as e:
            print(f"Ошибка при обработке {img_path}: {e}")
            return None
    
    def build_dataset_features(self, dataset_folder, cache_file='features_cache.pkl', max_workers=4):
        if os.path.exists(cache_file):
            print("Загрузка признаков из кэша...")
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                self.features = cache_data['features']
                self.image_paths = cache_data['image_paths']
                print(f"Загружено {len(self.image_paths)} изображений из кэша")
                self._build_faiss_index()
                return
        
        print("Извлечение признаков для датасета...")
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        temp_image_paths = []
        
        for filename in os.listdir(dataset_folder):
            if filename.lower().endswith(image_extensions):
                temp_image_paths.append(os.path.join(dataset_folder, filename))
        
        features_list = []
        successful_paths = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.extract_feature, img_path) for img_path in temp_image_paths]
            
            for i, future in enumerate(tqdm(futures, desc="Извлечение признаков")):
                feature = future.result()
                if feature is not None:
                    features_list.append(feature)
                    successful_paths.append(temp_image_paths[i])
        
        if features_list:
            self.features = np.array(features_list)
            self.image_paths = successful_paths
            
            cache_data = {
                'features': self.features,
                'image_paths': self.image_paths
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            print(f"Извлечены признаки для {len(self.image_paths)} изображений")
            self._build_faiss_index()
        else:
            raise ValueError("Не удалось извлечь признаки ни для одного изображения")
    
    def _build_faiss_index(self):
        if not self.use_faiss or self.features is None:
            return
            
        print("Построение FAISS индекса...")
        dimension = self.features.shape[1]
        
        if self.use_gpu and hasattr(faiss, 'StandardGpuResources'):
            res = faiss.StandardGpuResources()
            if self.metric == 'cosine':
                cpu_index = faiss.IndexFlatIP(dimension)
            else:  # euclidean
                cpu_index = faiss.IndexFlatL2(dimension)  # Изменено
            self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        else:
            if self.metric == 'cosine':
                self.index = faiss.IndexFlatIP(dimension)
            else:  # euclidean
                self.index = faiss.IndexFlatL2(dimension)  # Изменено
        
        # Для косинусного сходства нормализуем, для евклидова - нет
        if self.metric == 'cosine':
            faiss.normalize_L2(self.features)
        
        self.index.add(self.features.astype('float32'))
        print("FAISS индекс построен")
    
    def search_similar(self, query_image_path, top_k=5):
        print("Извлечение признаков для запроса...")
        query_feature = self.extract_feature(query_image_path)
        
        if query_feature is None:
            raise ValueError("Не удалось извлечь признаки для запроса")
        
        query_feature = query_feature.reshape(1, -1)
        
        if self.use_faiss and self.index is not None:
            if self.metric == 'cosine':
                faiss.normalize_L2(query_feature)
            
            distances, indices = self.index.search(query_feature.astype('float32'), min(top_k, len(self.image_paths)))
            
            results = []
            for distance, idx in zip(distances[0], indices[0]):
                if idx < len(self.image_paths):
                    # Для косинусного сходства: больше = лучше, для евклидова: меньше = лучше
                    score = 1.0 / (1.0 + distance) if self.metric == 'euclidean' else float(distance)
                    results.append({
                        'path': self.image_paths[idx],
                        'score': score,
                        'distance': float(distance),  # Добавляем оригинальное расстояние
                        'filename': os.path.basename(self.image_paths[idx])
                    })
        else:
            # Традиционный поиск
            if self.metric == 'euclidean':
                distances = euclidean_distances(query_feature, self.features)[0]
                top_indices = np.argsort(distances)[:top_k]  # Для евклидова расстояния берем наименьшие
            else:
                from sklearn.metrics.pairwise import cosine_similarity
                similarities = cosine_similarity(query_feature, self.features)[0]
                top_indices = np.argsort(similarities)[-top_k:][::-1]
                distances = 1 - similarities  # Преобразуем в "расстояния"
            
            results = []
            for idx in top_indices:
                score = float(1.0 / (1.0 + distances[idx])) if self.metric == 'euclidean' else float(1 - distances[idx])
                results.append({
                    'path': self.image_paths[idx],
                    'score': score,
                    'distance': float(distances[idx]),
                    'filename': os.path.basename(self.image_paths[idx])
                })
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Быстрый поиск похожих изображений с ResNet')
    parser.add_argument('--dataset', type=str, required=True, help='Путь к папке с изображениями')
    parser.add_argument('--query', type=str, required=True, help='Путь к запросу (изображению для поиска)')
    parser.add_argument('--top-k', type=int, default=5, help='Количество результатов поиска')
    parser.add_argument('--cache-file', type=str, default='features_cache.pkl', help='Файл кэша признаков')
    parser.add_argument('--use-gpu', action='store_true', help='Использовать GPU если доступно')
    parser.add_argument('--no-faiss', action='store_true', help='Отключить использование FAISS')
    parser.add_argument('--workers', type=int, default=4, help='Количество потоков для параллельной обработки')
    parser.add_argument('--metric', type=str, default='euclidean', choices=['cosine', 'euclidean'], 
                       help='Метрика для сравнения (cosine или euclidean)')  # Добавлен параметр
    
    args = parser.parse_args()
    
    searcher = FastImageSearcher(use_gpu=args.use_gpu, use_faiss=not args.no_faiss, metric=args.metric)
    
    searcher.build_dataset_features(
        dataset_folder=args.dataset,
        cache_file=args.cache_file,
        max_workers=args.workers
    )

    for index in range(1, 11):
        print(f"\nПоиск похожих изображений для: {args.query.replace('xxx', str(index))}")
        results = searcher.search_similar(args.query.replace('xxx', str(index)), top_k=args.top_k)
        
        print("\n" + "="*60)
        print(f"РЕЗУЛЬТАТЫ ПОИСКА ПОХОЖИХ ИЗОБРАЖЕНИЙ ({args.metric.upper()} МЕТРИКА)")
        print("="*60)
        
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['filename']}")
            print(f"   Путь: {result['path']}")
            print(f"   Расстояние: {result['distance']:.4f}")
            print(f"   Счет: {result['score']:.4f}")
            print()

if __name__ == "__main__":
    main()
