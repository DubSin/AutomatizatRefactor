#!/usr/bin/env python3
"""
Главный скрипт для запуска пайплайна обработки тендеров.
Используется для запуска оркестратора из корневой папки проекта.
"""

import sys
import os
import asyncio
import argparse
from pipeline.config import HEADLESS_MODE

# Добавляем путь к папке pipeline (где лежат все модули)
pipeline_dir = os.path.join(os.path.dirname(__file__), 'pipeline')
if not os.path.isdir(pipeline_dir):
    print(f"Ошибка: папка pipeline не найдена по пути {pipeline_dir}")
    print("Убедитесь, что все файлы пайплайна находятся в подпапке 'pipeline'")
    sys.exit(1)

sys.path.insert(0, pipeline_dir)

# Импортируем оркестратор
try:
    from pipeline.orchestrator import run_pipeline
except ImportError as e:
    print(f"Ошибка импорта: {e}")
    print("Убедитесь, что все модули пайплайна присутствуют в папке 'pipeline'")
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Запуск пайплайна обработки тендеров (почта → скачивание → извлечение текста → классификация → отчёт)"
    )
    parser.add_argument(
        '--headless', action='store_true', default=HEADLESS_MODE,
        help='Запуск браузера в headless режиме (без графического интерфейса)'
    )
    parser.add_argument(
        '--links-file', type=str, default=None,
        help='Путь к файлу для сохранения извлечённых ссылок на тендеры (по умолчанию: значение из config.LINKS_FILE)'
    )
    parser.add_argument(
        '--max-pages', type=int, default=None,
        help='Максимальное количество страниц для обработки при извлечении текста (переопределяет значения по умолчанию)'
    )
    parser.add_argument(
        '--log-file', type=str, default=None,
        help='Путь к файлу для логов (по умолчанию: значение из config.LOG_FILE)'
    )
    args = parser.parse_args()

    # Запуск оркестратора с переданными параметрами
    asyncio.run(run_pipeline(
        headless=args.headless,
        links_file=args.links_file,
        max_pages=args.max_pages,
        log_file=args.log_file
    ))


if __name__ == "__main__":
    main()