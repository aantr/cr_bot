import ultralytics.nn.tasks as tasks
import ultralytics.utils.torch_utils as torch_utils
import re
import os
import glob

def replace_torch_load_in_files(paths):
    """
    Заменяет torch.load(*) на torch.load(*, weights_only=False) в файлах
    
    Args:
        file_pattern: шаблон поиска файлов (по умолчанию '*.py')
        directory: директория для поиска
    """
    pattern = r'torch\.load\((?!.*weights_only)(.+)\)'

    replacement = r'torch.load(\1, weights_only=False)'
    
    files_changed = 0
    
    for filepath in paths:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            new_content = re.sub(pattern, replacement, content)
            
            if new_content != content:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                print(f"✓ Изменен: {filepath}")
                files_changed += 1
        except Exception as e:
            print(f"✗ Ошибка при обработке {filepath}: {e}")
    
    print(f"\nИтого изменено файлов: {files_changed}")

paths = [tasks.__file__, torch_utils.__file__]

print(paths)

replace_torch_load_in_files(paths)