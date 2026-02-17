import os
import yaml
import argparse
from pathlib import Path
from typing import Set, List, Dict, Tuple
import sys
from tqdm import tqdm
import shutil
import json
import cv2

class YOLOFilterYAML:
    """Фильтрация YOLO аннотаций с загрузкой классов из YAML и поддержкой формата folder/img.jpg + folder/img.txt"""
    
    def __init__(self, yaml_path: str = None):
        """
        Инициализация фильтра
        
        Args:
            yaml_path: путь к YAML файлу с классами
        """
        self.allowed_classes = set()
        self.class_name_to_id = {}
        self.class_id_to_name = {}
        self.dataset_info = {}
        
        self.yaml_path = yaml_path
        if yaml_path:
            self.load_classes_from_yaml()
    
    def load_classes_from_yaml(self, allowed_path: str = None) -> Dict:
        """
        Загрузка классов из YAML файла
        
        Args:
            yaml_path: путь к YAML файлу
            
        Returns:
            Информация о датасете из YAML
        """
        yaml_path: str = self.yaml_path
        if not os.path.exists(yaml_path):
            print(f"YAML файл не найден: {yaml_path}")
            return {}
        
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                self.dataset_info = yaml.safe_load(f)
            
            # Извлекаем классы из YAML
            # Формат YOLO YAML: {'names': ['class1', 'class2', ...]}
            if 'names' in self.dataset_info:
                classes = self.dataset_info['names']
            elif 'nc' in self.dataset_info:
                # Если есть только количество классов, создаем generic имена
                classes = [f"class_{i}" for i in range(self.dataset_info['nc'])]
            else:
                # Пробуем найти классы в других форматах
                classes = self._extract_classes_from_yaml(self.dataset_info)
            if not allowed_path:
                self.allowed_classes = classes.values()
            else:
                with open(allowed_path, 'r') as f:
                    self.allowed_classes = f.read().strip().split('\n')   
            self.classes = classes
            self.class_name_to_id = {name: idx for idx, name in classes.items()}
            self.class_id_to_name = {idx: name for name, idx in self.class_name_to_id.items()}
            assert all([type(k) == int for k, v in self.class_id_to_name.items()])
            # assert all([v in self.allowed_classes for k, v in self.class_id_to_name.items()])
            # print ([v for k, v in self.class_id_to_name.items() if v not in self.allowed_classes])
            # print(f"Загружено {len(self.classes)} классов из YAML")
            # print(f"Классы: {', '.join(map(str, self.allowed_classes))}")
            
            # # Выводим информацию о датасете
            # if 'path' in self.dataset_info:
            #     print(f"Путь датасета: {self.dataset_info['path']}")
            # if 'train' in self.dataset_info:
            #     print(f"Train images: {self.dataset_info['train']}")
            # if 'val' in self.dataset_info:
            #     print(f"Val images: {self.dataset_info['val']}")
            
            return self.dataset_info
            
        except yaml.YAMLError as e:
            print(f"Ошибка парсинга YAML: {e}")
            return {}
    
    def _extract_classes_from_yaml(self, yaml_data: Dict) -> List[str]:
        """Извлечение классов из различных форматов YAML"""
        classes = []
        
        # Пробуем разные ключи
        possible_keys = ['classes', 'labels', 'categories', 'class_names', 'object_names']
        for key in possible_keys:
            if key in yaml_data:
                if isinstance(yaml_data[key], list):
                    classes = yaml_data[key]
                    break
                elif isinstance(yaml_data[key], dict):
                    # Если это словарь вида {0: 'class1', 1: 'class2'}
                    classes = [yaml_data[key][k] for k in sorted(yaml_data[key].keys())]
                    break
        
        # Если ничего не нашли, пробуем найти вложенные структуры
        if not classes:
            for key, value in yaml_data.items():
                if isinstance(value, list) and len(value) > 0 and isinstance(value[0], str):
                    classes = value
                    break
        return classes
    
    def find_yaml_file(self, dataset_dir: str) -> str:
        """
        Поиск YAML файла в директории
        
        Args:
            dataset_dir: директория датасета
            
        Returns:
            Путь к найденному YAML файлу или None
        """
        dataset_path = Path(dataset_dir)
        
        # Ищем YAML файлы
        yaml_files = list(dataset_path.rglob('*.yaml')) + list(dataset_path.rglob('*.yml'))
        
        # Приоритетные имена файлов
        priority_names = ['data.yaml', 'dataset.yaml', 'config.yaml', 'labels.yaml']
        
        for priority in priority_names:
            for yaml_file in yaml_files:
                if yaml_file.name == priority:
                    return str(yaml_file)
        
        # Возвращаем первый найденный
        if yaml_files:
            return str(yaml_files[0])
        
        return None
    
    def process_folder_structure(self, root_dir: str) -> List[Tuple[str, str]]:
        """
        Поиск пар изображение-аннотация в структуре folder/img.jpg + folder/img.txt
        
        Args:
            root_dir: корневая директория
            
        Returns:
            Список кортежей (путь_к_изображению, путь_к_аннотации)
        """
        root_path = Path(root_dir)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        pairs = []
        
        print("Поиск пар изображение-аннотация...")
        
        # Рекурсивно обходим все директории
        for folder in tqdm(list(root_path.rglob('*/')), desc="Сканирование папок"):
            if folder.is_dir():
                # Ищем изображения в этой папке
                for img_ext in image_extensions:
                    for img_file in folder.glob(f'*{img_ext}'):
                        # Соответствующий файл аннотаций
                        ann_file = folder / f"{img_file.stem}.txt"
                        
                        if ann_file.exists():
                            pairs.append((str(img_file), str(ann_file)))
                        else:
                            # Пробуем найти с другими расширениями изображений
                            for other_ext in image_extensions - {img_ext}:
                                alt_ann = folder / f"{img_file.stem}{other_ext}"
                                if alt_ann.with_suffix('.txt').exists():
                                    pairs.append((str(img_file), str(alt_ann.with_suffix('.txt'))))
                                    break
        
        print(f"Найдено {len(pairs)} пар изображение-аннотация")
        return pairs
    
    def filter_annotation(self, annotation_path: str, 
                         output_path: str = None,
                         dry_run: bool = False) -> Tuple[bool, int, int]:
        """
        Фильтрация одного файла аннотаций
        
        Args:
            annotation_path: путь к файлу аннотаций
            output_path: путь для сохранения (если None, заменяет оригинал)
            dry_run: режим просмотра без изменений
            
        Returns:
            (изменен_ли_файл, всего_аннотаций, оставлено_аннотаций)
        """
        if not os.path.exists(annotation_path):
            print(f"Файл аннотаций не найден: {annotation_path}")
            return False, 0, 0
        
        with open(annotation_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        filtered_lines = []
        total_count = 0
        kept_count = 0
        removed_count = 0
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                filtered_lines.append(line)
                continue
            
            parts = line.split()
            if len(parts) < 5:
                print(f"Предупреждение: строка {line_num} в {annotation_path} имеет неверный формат")
                continue
            
            try:
                class_id = int(parts[0])
                total_count += 1
                
                # Проверяем, разрешен ли класс
                if class_id in self.class_id_to_name:
                    # Класс найден в маппинге
                    class_name = self.class_id_to_name[class_id]
                    if class_name in self.allowed_classes:
                        filtered_lines.append(line)
                        kept_count += 1
                    else:
                        removed_count += 1
                        if not dry_run:
                            print(f"  Удален класс {class_id} ({class_name}) в {Path(annotation_path).name}")
                else:
                    # Класс не найден в маппинге
                    if class_id < len(self.allowed_classes):
                        # Предполагаем, что классы пронумерованы от 0
                        sorted_classes = sorted(self.allowed_classes)
                        class_name = sorted_classes[class_id]
                        if class_name in self.allowed_classes:
                            filtered_lines.append(line)
                            kept_count += 1
                        else:
                            removed_count += 1
                            if not dry_run:
                                print(f"  Удален неизвестный класс {class_id} в {Path(annotation_path).name}")
                    else:
                        removed_count += 1
                        if not dry_run:
                            print(f"  Удален неизвестный класс {class_id} в {Path(annotation_path).name}")
                    
            except ValueError as e:
                print(f"Ошибка в строке {line_num} файла {annotation_path}: {e}")
                filtered_lines.append(line)
        
        # Если файл был изменен и не dry_run
        if removed_count > 0 and not dry_run:
            output_file = output_path or annotation_path
            
            # Создаем директорию если нужно
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(filtered_lines))
            
            return True, total_count, kept_count
        
        return False, total_count, kept_count
    
    def process_dataset(self, dataset_dir: str,
                       output_dir: str = None,
                       dry_run: bool = False,
                       backup: bool = True) -> Dict:
        """
        Обработка всего датасета
        
        Args:
            dataset_dir: директория датасета
            output_dir: выходная директория
            dry_run: режим просмотра
            backup: создавать резервные копии
            
        Returns:
            Статистика обработки
        """
        stats = {
            'total_pairs': 0,
            'processed_pairs': 0,
            'modified_files': 0,
            'total_annotations': 0,
            'kept_annotations': 0,
            'removed_annotations': 0,
            'errors': 0
        }
        
        # Находим все пары
        pairs = self.process_folder_structure(dataset_dir)
        stats['total_pairs'] = len(pairs)
        
        if not pairs:
            print("Не найдено пар изображение-аннотация")
            return stats
        
        # Создаем backup директорию если нужно
        if backup and not dry_run:
            backup_dir = Path(dataset_dir) / 'backup_annotations'
            backup_dir.mkdir(exist_ok=True)
            print(f"Резервные копии будут сохранены в: {backup_dir}")
        
        # Обрабатываем каждую пару
        for img_path, ann_path in tqdm(pairs, desc="Обработка пар"):
            try:
                stats['processed_pairs'] += 1
                
                # Определяем выходной путь
                if output_dir:
                    rel_path = Path(ann_path).relative_to(dataset_dir)
                    output_ann_path = Path(output_dir) / rel_path
                else:
                    output_ann_path = ann_path
                
                # Создаем резервную копию если нужно
                if backup and not dry_run and output_dir is None:
                    backup_path = backup_dir / Path(ann_path).relative_to(dataset_dir)
                    backup_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(ann_path, backup_path)
                
                # Копируем изображение если указана output_dir
                if output_dir and not dry_run:
                    rel_img_path = Path(img_path).relative_to(dataset_dir)
                    output_img_path = Path(output_dir) / rel_img_path
                    output_img_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(img_path, output_img_path)
                
                # Фильтруем аннотацию
                modified, total, kept = self.filter_annotation(
                    ann_path, 
                    str(output_ann_path) if output_dir else None,
                    dry_run
                )
                
                if modified:
                    stats['modified_files'] += 1
                
                stats['total_annotations'] += total
                stats['kept_annotations'] += kept
                stats['removed_annotations'] += (total - kept)
                
            except Exception as e:
                stats['errors'] += 1
                print(f"Ошибка обработки пары {img_path}: {e}")
        
        return stats
    
    def analyze_dataset(self, dataset_dir: str) -> Dict:
        """
        Анализ датасета: какие классы используются
        
        Args:
            dataset_dir: директория датасета
            
        Returns:
            Статистика использования классов
        """
        class_stats = {
            'total_annotations': 0,
            'class_counts': {class_name: 0 for class_name in self.allowed_classes},
            'unknown_classes': {},
            'files_per_class': {class_name: set() for class_name in self.allowed_classes}
        }
        
        pairs = self.process_folder_structure(dataset_dir)
        
        for img_path, ann_path in tqdm(pairs, desc="Анализ датасета"):
            try:
                with open(ann_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        
                        parts = line.split()
                        if parts:
                            try:
                                class_id = int(parts[0])
                                class_stats['total_annotations'] += 1
                                
                                # Определяем имя класса
                                class_name = None
                                if class_id in self.class_id_to_name:
                                    class_name = self.class_id_to_name[class_id]
                                elif class_id < len(self.allowed_classes):
                                    sorted_classes = sorted(self.allowed_classes)
                                    class_name = sorted_classes[class_id]
                                
                                if class_name and class_name in self.allowed_classes:
                                    class_stats['class_counts'][class_name] += 1
                                    class_stats['files_per_class'][class_name].add(Path(img_path).name)
                                else:
                                    # Неизвестный класс
                                    if class_id not in class_stats['unknown_classes']:
                                        class_stats['unknown_classes'][class_id] = 0
                                    class_stats['unknown_classes'][class_id] += 1
                                    
                            except ValueError:
                                continue
            except Exception as e:
                print(f"Ошибка чтения файла {ann_path}: {e}")
        
        return class_stats
    
    def create_filtered_yaml(self, original_yaml_path: str, 
                           used_classes: Set[str],
                           output_yaml_path: str = None) -> str:
        """
        Создание отфильтрованного YAML файла
        
        Args:
            original_yaml_path: путь к оригинальному YAML
            used_classes: множество использованных классов
            output_yaml_path: путь для сохранения
            
        Returns:
            Путь к созданному YAML файлу
        """
        if not os.path.exists(original_yaml_path):
            print(f"Оригинальный YAML файл не найден: {original_yaml_path}")
            return None
        
        with open(original_yaml_path, 'r', encoding='utf-8') as f:
            yaml_data = yaml.safe_load(f)
        
        # Фильтруем классы
        if 'names' in yaml_data:
            # Сохраняем порядок из оригинального файла
            filtered_names = [name for name in yaml_data['names'] if name in used_classes]
            yaml_data['names'] = filtered_names
            yaml_data['nc'] = len(filtered_names)
        
        # Обновляем количество классов
        if 'nc' in yaml_data:
            yaml_data['nc'] = len(used_classes)
        
        # Определяем выходной путь
        if output_yaml_path is None:
            output_yaml_path = str(Path(original_yaml_path).with_stem(
                Path(original_yaml_path).stem + '_filtered'
            ))
        
        # Сохраняем
        with open(output_yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True)
        
        print(f"Создан отфильтрованный YAML: {output_yaml_path}")
        return output_yaml_path

def main():
    parser = argparse.ArgumentParser(
        description='Фильтрация YOLO аннотаций с загрузкой классов из YAML'
    )
    
    parser.add_argument('dataset_dir', type=str,
                       help='Директория с датасетом (структура: folder/img.jpg + folder/img.txt)')
    parser.add_argument('--yaml', '-y', type=str,
                       help='Путь к YAML файлу с классами (автопоиск если не указан)')
    parser.add_argument('--allowed', '-a', type=str,
                       help='Путь к allowed файлу с классами')
    parser.add_argument('--output-dir', '-o', type=str,
                       help='Выходная директория для отфильтрованных файлов')
    parser.add_argument('--dry-run', action='store_true',
                       help='Только анализ без изменений')
    parser.add_argument('--no-backup', action='store_true',
                       help='Не создавать резервные копии')
    parser.add_argument('--analyze-only', action='store_true',
                       help='Только анализ датасета')
    parser.add_argument('--min-annotations', type=int, default=1,
                       help='Минимальное количество аннотаций для сохранения класса')
    
    args = parser.parse_args()
    
    # Проверяем директорию
    if not os.path.isdir(args.dataset_dir):
        print(f"Ошибка: директория {args.dataset_dir} не существует")
        sys.exit(1)
    
    # Создаем фильтр
    filter_obj = YOLOFilterYAML(args.yaml)
    
    # Ищем или загружаем YAML файл
    yaml_path = args.yaml
    if not yaml_path:
        yaml_path = filter_obj.find_yaml_file(args.dataset_dir)
        
    if yaml_path:
        print(f"Используется YAML файл: {yaml_path}")
        filter_obj.load_classes_from_yaml(args.allowed)
    else:
        print("YAML файл не найден. Используйте --yaml для указания пути.")
        sys.exit(1)
    
    if not filter_obj.allowed_classes:
        print("Ошибка: не удалось загрузить классы из YAML")
        sys.exit(1)
    
    if args.analyze_only:
        # Только анализ
        print("\n" + "="*60)
        print("АНАЛИЗ ДАТАСЕТА")
        print("="*60)
        
        stats = filter_obj.analyze_dataset(args.dataset_dir)
        
        print(f"\nВсего аннотаций: {stats['total_annotations']}")
        print(f"Используемые классы:")
        
        used_classes = set()
        for class_name, count in sorted(stats['class_counts'].items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                used_classes.add(class_name)
                file_count = len(stats['files_per_class'][class_name])
                print(f"  {class_name}: {count} аннотаций в {file_count} файлах")
        
        print(f"\nНеиспользуемые классы:")
        unused = filter_obj.allowed_classes - used_classes
        for class_name in sorted(unused):
            print(f"  {class_name}")
        
        if stats['unknown_classes']:
            print(f"\nНеизвестные классы (не найдены в YAML):")
            for class_id, count in sorted(stats['unknown_classes'].items()):
                print(f"  class_{class_id}: {count} аннотаций")
        
        # Предлагаем создать отфильтрованный YAML
        if unused and yaml_path:
            response = input(f"\nСоздать отфильтрованный YAML только с используемыми классами? (y/n): ")
            if response.lower() == 'y':
                filtered_yaml = filter_obj.create_filtered_yaml(
                    yaml_path, 
                    used_classes
                )
                print(f"Создан: {filtered_yaml}")
        
        sys.exit(0)
    
    if args.dry_run:
        print("\n" + "="*60)
        print("РЕЖИМ ПРОСМОТРА (без изменений)")
        print("="*60)
    
    # Обрабатываем датасет
    stats = filter_obj.process_dataset(
        args.dataset_dir,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        backup=not args.no_backup
    )
    
    # Выводим статистику
    print("\n" + "="*60)
    print("СТАТИСТИКА ОБРАБОТКИ")
    print("="*60)
    
    print(f"Найдено пар изображение-аннотация: {stats['total_pairs']}")
    print(f"Обработано пар: {stats['processed_pairs']}")
    
    if not args.dry_run:
        print(f"Изменено файлов аннотаций: {stats['modified_files']}")
    
    print(f"Всего аннотаций: {stats['total_annotations']}")
    print(f"Сохранено аннотаций: {stats['kept_annotations']}")
    print(f"Удалено аннотаций: {stats['removed_annotations']}")
    
    if stats['total_annotations'] > 0:
        percentage = (stats['kept_annotations'] / stats['total_annotations']) * 100
        print(f"Процент сохраненных: {percentage:.1f}%")
    
    print(f"Ошибок: {stats['errors']}")
    
    # Сохраняем отчет
    report_data = {
        'dataset_dir': args.dataset_dir,
        'yaml_file': yaml_path,
        'allowed_classes': list(filter_obj.allowed_classes),
        'stats': stats,
        'timestamp': __import__('datetime').datetime.now().isoformat()
    }
    
    report_dir = Path(args.output_dir or args.dataset_dir)
    report_path = report_dir / 'filter_report.json'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nОтчет сохранен в: {report_path}")
    
    # Создаем отфильтрованный YAML если указана выходная директория
    if args.output_dir and yaml_path:
        # Анализируем какие классы остались
        temp_stats = filter_obj.analyze_dataset(args.output_dir)
        used_classes = {name for name, count in temp_stats['class_counts'].items() if count > 0}
        
        filtered_yaml = filter_obj.create_filtered_yaml(
            yaml_path,
            used_classes,
            str(Path(args.output_dir) / 'data_filtered.yaml')
        )

def quick_filter(args):
    """Быстрая функция для фильтрации"""
    import sys
    
    if len(sys.argv) < 2:
        print("Использование: python script.py <dataset_dir> [--yaml path/to/data.yaml]")
        sys.exit(1)
    
    dataset_dir = sys.argv[1]
    yaml_path = None
    
    # Простой парсинг аргументов
    for i, arg in enumerate(sys.argv[2:]):
        if arg == '--yaml' and i+2 < len(sys.argv):
            yaml_path = sys.argv[i+3]
    
    filter_obj = YOLOFilterYAML(yaml_path)
    
    if not yaml_path:
        yaml_path = filter_obj.find_yaml_file(dataset_dir)
        if yaml_path:
            filter_obj.load_classes_from_yaml(yaml_path, args.allowed)
    
    if not filter_obj.allowed_classes:
        print("Не удалось загрузить классы")
        sys.exit(1)
    
    stats = filter_obj.process_dataset(dataset_dir)
    
    print(f"Готово! Удалено {stats['removed_annotations']} из {stats['total_annotations']} аннотаций")

if __name__ == "__main__":
    main()