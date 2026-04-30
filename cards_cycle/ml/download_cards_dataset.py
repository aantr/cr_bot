import requests
import os
import re
from bs4 import BeautifulSoup
import time
from urllib.parse import urljoin
from tqdm import tqdm
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

class ClashRoyaleDatasetDownloader:
    def __init__(self, base_url="https://royaleapi.com/cards/popular?sort=rating", output_dir="cards_dataset"):
        """
        Инициализация загрузчика датасета
        
        Args:
            base_url: базовый URL сайта
            output_dir: директория для сохранения изображений
        """
        self.base_url = base_url
        self.output_dir = output_dir
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Создаем директорию для датасета
        os.makedirs(output_dir, exist_ok=True)
        
    def parse_html_cards(self, html_content):
        """
        Парсит HTML и извлекает информацию о картах
        
        Args:
            html_content: HTML содержимое страницы
            
        Returns:
            list: список словарей с информацией о картах
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        cards = []
        # Ищем все элементы с карточками
        card_elements = soup.find_all('div', class_='grid_item')
        
        for card_element in card_elements:
            try:
                # Получаем имя карты
                name_element = card_element.find('div', class_='card_name')
                card_name = name_element.get_text(strip=True) if name_element else "Unknown"
                
                # Получаем data-card атрибут
                data_card = card_element.get('data-card', '')
                if not data_card:
                    continue
                    
                # Ищем изображение
                img_element = card_element.find('img', class_='deck_card')
                if not img_element:
                    continue
                    
                img_src = img_element.get('src', '')
                if not img_src:
                    continue
                
                # Очищаем имя файла
                safe_filename = re.sub(r'[<>:"/\\|?*]', '_', card_name)
                safe_filename = safe_filename.strip()
                card_info = {
                    'name': card_name,
                    'data_card': data_card,
                    'image_url': img_src,
                    'filename': f"{safe_filename}_{data_card}"
                }
                
                cards.append(card_info)
                
            except Exception as e:
                print(f"Ошибка при парсинге карточки: {e}")
                continue
                
        return cards
    
    def download_image(self, image_url, filename, folder="images"):
        """
        Скачивает одно изображение
        
        Args:
            image_url: URL изображения
            filename: имя файла для сохранения
            folder: папка для сохранения
            
        Returns:
            bool: успех операции
        """
        try:
            # Создаем папку если не существует
            folder_path = os.path.join(self.output_dir, folder)
            os.makedirs(folder_path, exist_ok=True)
            
            # Полный путь к файлу
            file_path = os.path.join(folder_path, f"{filename}.png")
            
            # Проверяем, существует ли файл
            if os.path.exists(file_path):
                print(f"Файл уже существует: {filename}")
                return True
            
            # Скачиваем изображение
            response = self.session.get(image_url, timeout=30)
            response.raise_for_status()
            
            # Сохраняем изображение
            with open(file_path, 'wb') as f:
                f.write(response.content)
            
            print(f"Скачано: {filename}")
            return True
            
        except Exception as e:
            print(f"Ошибка при скачивании {filename}: {e}")
            return False
    
    def download_all_cards(self, cards_info, max_workers=5):
        """
        Скачивает все изображения карт параллельно
        
        Args:
            cards_info: список информации о картах
            max_workers: максимальное количество потоков
        """
        print(f"Начинаем скачивание {len(cards_info)} карт...")

        successful_downloads = 0
        failed_downloads = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Создаем задачи для скачивания
            future_to_card = {
                executor.submit(
                    self.download_image, 
                    card['image_url'], 
                    card['filename']
                ): card for card in cards_info
            }
            
            # Обрабатываем результаты
            for future in tqdm(as_completed(future_to_card), total=len(cards_info)):
                card = future_to_card[future]
                try:
                    success = future.result()
                    if success:
                        successful_downloads += 1
                    else:
                        failed_downloads += 1
                except Exception as e:
                    print(f"Ошибка при обработке {card['filename']}: {e}")
                    failed_downloads += 1
        
        print(f"\nРезультаты скачивания:")
        print(f"Успешно скачано: {successful_downloads}")
        print(f"Ошибок: {failed_downloads}")
        print(f"Общее количество: {len(cards_info)}")
        
        return successful_downloads, failed_downloads
    
    def create_dataset_structure(self, cards_info):
        """
        Создает структуру датасета с отдельными папками для каждой карты
        """
        print("Создание структуры датасета...")
        
        dataset_dir = os.path.join(self.output_dir, "dataset")
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Создаем папку для каждой карты
        for card in cards_info:
            card_folder = os.path.join(dataset_dir, card['data_card'])
            os.makedirs(card_folder, exist_ok=True)
            
            # Копируем изображение в папку карты (если нужно)
            original_path = os.path.join(self.output_dir, "images", f"{card['filename']}.png")
            new_path = os.path.join(card_folder, f"{card['data_card']}_001.png")
            
            if os.path.exists(original_path):
                try:
                    import shutil
                    shutil.copy2(original_path, new_path)
                except Exception as e:
                    print(f"Ошибка при копировании {card['filename']}: {e}")
        
        print(f"Структура датасета создана в: {dataset_dir}")
    
    def save_cards_info(self, cards_info, filename="cards_info.txt"):
        """
        Сохраняет информацию о картах в текстовый файл
        """
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("Информация о картах Clash Royale\n")
            f.write("=" * 50 + "\n\n")
            
            for i, card in enumerate(cards_info, 1):
                f.write(f"{i}. {card['name']}\n")
                f.write(f"   Data-card: {card['data_card']}\n")
                f.write(f"   Image URL: {card['image_url']}\n")
                f.write(f"   Filename: {card['filename']}\n\n")
        
        print(f"Информация о картах сохранена в: {filepath}")

# Пример использования:
def main():
    """
    Основная функция для скачивания датасета
    """
    
    # Ваш HTML контент (вы можете загрузить его из файла или веб-страницы)
    
    # Инициализация загрузчика
    downloader = ClashRoyaleDatasetDownloader(
        output_dir="cards_dataset"
    )

    response = requests.get('https://royaleapi.com/cards/popular?sort=rating', headers={
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    response.raise_for_status()

    # html_content = open('cards_dataset/data.html', encoding='utf8').read()
    html_content = response.text

    
    # Парсим HTML и получаем информацию о картах
    print("Парсинг HTML...")
    cards_info = downloader.parse_html_cards(html_content)
    
    if not cards_info:
        print("Не найдено карт для скачивания")
        return
    
    print(f"Найдено {len(cards_info)} карт для скачивания")
    
    # Показываем первые несколько карт
    print("\nПервые 5 карт:")
    for i, card in enumerate(cards_info[:5]):
        print(f"{i+1}. {card['name']} ({card['data_card']})")
    
    # Скачиваем все изображения
    print("\nСкачивание изображений...")
    successful, failed = downloader.download_all_cards(cards_info, max_workers=3)
    
    # Сохраняем информацию о картах
    downloader.save_cards_info(cards_info)
    
    # Создаем структуру датасета
    downloader.create_dataset_structure(cards_info)
    
    print(f"\nГотово! Датасет сохранен в папке 'cards_dataset'")

# Альтернативный способ - скачивание с веб-страницы
def download_from_webpage(url):
    """
    Скачивает датасет с веб-страницы
    """
    try:
        downloader = ClashRoyaleDatasetDownloader()
        
        # Загружаем веб-страницу
        response = requests.get(url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        response.raise_for_status()
        
        # Парсим HTML
        cards_info = downloader.parse_html_cards(response.text)
        
        if cards_info:
            print(f"Найдено {len(cards_info)} карт")
            
            # Скачиваем изображения
            successful, failed = downloader.download_all_cards(cards_info)
            
            # Сохраняем информацию
            downloader.save_cards_info(cards_info)
            
            return successful, failed
        else:
            print("Карты не найдены")
            return 0, 0
            
    except Exception as e:
        print(f"Ошибка при загрузке с веб-страницы: {e}")
        return 0, 0

if __name__ == "__main__":
    main()
