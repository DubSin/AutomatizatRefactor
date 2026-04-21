import asyncio
import aiohttp
import aiofiles
import os
import re
import json
import logging
import zipfile
from urllib.parse import unquote, urlparse
from typing import Optional, Dict, Any
from asyncio import Semaphore

# Попытка импорта дополнительных библиотек для работы с архивами
try:
    import rarfile
    RARFILE_AVAILABLE = True
    # На macOS unrar ставится через: brew install unrar
    # rarfile сам найдёт его в PATH; явный путь не нужен
except ImportError:
    rarfile = None
    RARFILE_AVAILABLE = False
    logging.getLogger(__name__).warning("rarfile не установлен, поддержка .rar отключена")

try:
    import py7zr
    PY7ZR_AVAILABLE = True
except ImportError:
    py7zr = None
    PY7ZR_AVAILABLE = False
    logging.getLogger(__name__).warning("py7zr не установлен, поддержка .7z отключена")

from config import DEFAULT_ALLOWED_EXTS

logger = logging.getLogger(__name__)

# Расширения архивов, которые мы умеем распаковывать
ARCHIVE_EXTS = {'.zip', '.rar', '.7z'}

try:
    import mammoth
    MAMMOTH_AVAILABLE = True
except ImportError:
    mammoth = None
    MAMMOTH_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "mammoth не установлен, конвертация .doc -> .txt отключена. "
        "Установите: pip install mammoth"
    )


async def convert_doc_to_txt(doc_path: str) -> Optional[str]:
    """
    Конвертирует .doc-файл в .txt с помощью mammoth.
    Возвращает путь к новому .txt-файлу или None при ошибке.

    Зависимость: pip install mammoth
    Ограничение: форматирование не сохраняется, только plain text.
    """
    if not MAMMOTH_AVAILABLE:
        logger.error(f"mammoth не установлен, конвертация пропущена: {doc_path}")
        return None

    txt_path = os.path.splitext(doc_path)[0] + ".txt"

    try:
        # mammoth синхронный — запускаем в пуле потоков, чтобы не блокировать event loop
        def _convert():
            with open(doc_path, "rb") as f:
                result = mammoth.extract_raw_text(f)
            for msg in result.messages:
                logger.debug(f"mammoth [{os.path.basename(doc_path)}]: {msg}")
            return result.value

        text = await asyncio.to_thread(_convert)

        async with aiofiles.open(txt_path, "w", encoding="utf-8") as f:
            await f.write(text)

        logger.debug(f"Конвертирован .doc -> .txt: {txt_path}")
        return txt_path

    except Exception as e:
        logger.exception(f"Ошибка при конвертации {doc_path}: {e}")
        return None

async def extract_archive_if_needed(archive_path: str):
    """
    Проверяет, является ли файл архивом, и если да, распаковывает его содержимое
    в поддиректорию с именем архива (без расширения) рядом с архивом.
    Рекурсивно обрабатывает вложенные архивы.
    Возвращает общее количество распакованных файлов (включая вложенные) или -1 в случае ошибки.
    """
    ext = os.path.splitext(archive_path)[1].lower()
    if ext not in ARCHIVE_EXTS:
        return 0

    target_dir = os.path.splitext(archive_path)[0] + "_files"
    os.makedirs(target_dir, exist_ok=True)

    extracted_count = 0
    try:
        if ext == '.zip':
            # Пытаемся распаковать с указанием кодировки (Python 3.11+)
            try:
                with zipfile.ZipFile(archive_path, 'r', metadata_encoding='cp866') as zf:
                    zf.extractall(target_dir)
                    extracted_count = len(zf.namelist())
            except TypeError:
                # Для старых версий Python распаковываем и переименовываем файлы из cp866
                with zipfile.ZipFile(archive_path, 'r') as zf:
                    zf.extractall(target_dir)
                    extracted_count = len(zf.namelist())
                    for name in zf.namelist():
                        try:
                            # Пробуем декодировать имя из cp866 (обычно через cp437 как промежуточный)
                            new_name = name.encode('cp437').decode('cp866')
                            if new_name != name:
                                old_path = os.path.join(target_dir, name)
                                new_path = os.path.join(target_dir, new_name)
                                if os.path.exists(old_path):
                                    os.rename(old_path, new_path)
                        except:
                            pass
            logger.debug(f"Распакован zip-архив {archive_path} -> {target_dir} ({extracted_count} файлов)")

        elif ext == '.rar':
            if not RARFILE_AVAILABLE:
                logger.warning(f"rarfile не установлен, пропускаем распаковку {archive_path}")
                return 0
            
            if re.search(r'\.part(?:0*[2-9]|[1-9]\d+)\.rar$', archive_path, re.IGNORECASE):
                logger.debug(f"Пропуск тома (распаковывается вместе с первым): {archive_path}")
                return 0
            # Пропускаем старые форматы томов .r01, .r02...
            if re.search(r'\.r\d{2,}$', archive_path, re.IGNORECASE):
                return 0

            with rarfile.RarFile(archive_path, 'r') as rf:
                rf.extractall(target_dir)
                extracted_count = len(rf.namelist())
            logger.debug(f"Распакован RAR (включая все тома): {archive_path}")

        elif ext == '.7z':
            if not PY7ZR_AVAILABLE:
                logger.warning(f"py7zr не установлен, пропускаем распаковку {archive_path}")
                return 0
            with py7zr.SevenZipFile(archive_path, mode='r') as sz:
                sz.extractall(target_dir)
                extracted_count = len(sz.getnames()) if hasattr(sz, 'getnames') else 0
            logger.debug(f"Распакован 7z-архив {archive_path} -> {target_dir} ({extracted_count} файлов)")

        # --- Рекурсивная обработка вложенных архивов и .doc файлов ---
        for root, dirs, files in os.walk(target_dir):
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file_path)[1].lower()
                if file_ext in ARCHIVE_EXTS:
                    nested_count = await extract_archive_if_needed(file_path)
                    if nested_count > 0:
                        extracted_count += nested_count
                elif file_ext == '.doc':
                    txt_path = await convert_doc_to_txt(file_path)
                    if txt_path:
                        try:
                            os.remove(file_path)
                            logger.debug(f"Удалён .doc из архива после конвертации: {file_path}")
                        except OSError as e:
                            logger.warning(f"Не удалось удалить .doc из архива {file_path}: {e}")
        # -------------------------------------------------

    except Exception as e:
        logger.error(f"Ошибка при распаковке архива {archive_path}: {e}")
        return -1

    return extracted_count


async def download_tender_files_async(
    jsonl_path: str,
    base_download_dir: str = "downloaded_files",
    max_concurrent_files: int = 5,
    overwrite: bool = False,
    timeout: int = 30
) -> Dict[str, Any]:
    """
    Асинхронно скачивает файлы тендеров из JSONL-файла.
    После скачивания автоматически распаковывает архивы (zip, rar, 7z)
    в подпапки с именами архивов.

    Параметры:
        jsonl_path (str): путь к JSONL-файлу с данными тендеров.
        base_download_dir (str): корневая папка для сохранения файлов.
        max_concurrent_files (int): максимальное количество одновременно скачиваемых файлов.
        overwrite (bool): перезаписывать ли уже существующие файлы.
        timeout (int): таймаут для HTTP-запросов в секундах.

    Возвращает:
        Dict[str, Any]: статистика по скачиванию, включая список обработанных тендеров.
    """

    if not os.path.exists(jsonl_path):
        logger.error(f"JSONL файл не найден: {jsonl_path}")
        return {"error": "file_not_found"}

    stats = {
        "processed_tenders": 0,
        "total_files_found": 0,
        "downloaded_files": 0,
        "skipped_files": 0,
        "extracted_archives": 0,      # количество распакованных архивов
        "extracted_files": 0,          # общее количество файлов, извлечённых из архивов
        "errors": [],
        "tenders_info": []              # информация по каждому тендеру
    }

    semaphore = Semaphore(max_concurrent_files)

    async with aiohttp.ClientSession(
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    ) as session:

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue

            try:
                tender = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"Строка {line_num}: ошибка парсинга JSON: {e}")
                stats["errors"].append(f"line {line_num}: JSON parse error")
                continue

            tender_number = tender.get("number")
            if not tender_number:
                logger.warning(f"Строка {line_num}: отсутствует поле 'number', пропускаем тендер")
                stats["errors"].append(f"line {line_num}: missing 'number'")
                continue

            files = tender.get("files", [])
            if not files:
                continue

            # Создаём папку для тендера (блокирующая операция – выполняем в потоке)
            tender_dir = os.path.join(base_download_dir, str(tender_number))
            await asyncio.to_thread(os.makedirs, tender_dir, exist_ok=True)

            # Увеличиваем счётчик обработанных тендеров
            stats["processed_tenders"] += 1
            stats["total_files_found"] += len(files)

            # Список задач для этого тендера
            tasks = []
            for file_info in files:
                file_url = file_info.get("url")
                file_name = file_info.get("name")

                if not file_url:
                    logger.warning(f"Тендер {tender_number}: пропущен файл без URL")
                    stats["errors"].append(f"tender {tender_number}: missing URL")
                    continue

                # Определяем имя файла
                if not file_name:
                    parsed = urlparse(file_url)
                    file_name = os.path.basename(unquote(parsed.path))
                    if not file_name:
                        file_name = f"file_{stats['downloaded_files']}"
                        logger.warning(f"Тендер {tender_number}: не удалось определить имя файла, используется '{file_name}'")

                # Санитизация имени
                safe_name = "".join(c for c in file_name if c.isalnum() or c in "._- ").strip()
                if not safe_name:
                    safe_name = f"file_{stats['downloaded_files']}"

                file_path = os.path.join(tender_dir, safe_name)

                # Проверка существования файла (блокирующая)
                exists = await asyncio.to_thread(os.path.exists, file_path)
                if exists and not overwrite:
                    logger.debug(f"Файл уже существует, пропуск: {file_path}")
                    stats["skipped_files"] += 1
                    continue

                tasks.append(
                    _download_file(
                        session, file_url, file_path, semaphore, timeout,
                        tender_number, stats
                    )
                )

            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

            # Добавляем информацию о тендере в итоговый отчёт
            stats["tenders_info"].append({
                "tender_number": tender_number,
                "tender_dir": tender_dir,
                "files_count": len(files)
            })

    logger.info(
        f"Скачивание завершено. Обработано тендеров: {stats['processed_tenders']}, "
        f"найдено файлов: {stats['total_files_found']}, "
        f"скачано: {stats['downloaded_files']}, "
        f"пропущено (уже есть): {stats['skipped_files']}, "
        f"конвертировано .doc -> .txt: {stats.get('converted_doc_to_txt', 0)}, "
        f"распаковано архивов: {stats['extracted_archives']}, "
        f"извлечено файлов из архивов: {stats['extracted_files']}, "
        f"ошибок: {len(stats['errors'])}"
    )
    return stats


async def _download_file(
    session: aiohttp.ClientSession,
    url: str,
    file_path: str,
    semaphore: Semaphore,
    timeout: int,
    tender_number: str,
    stats: dict
):
    """Внутренняя функция для скачивания одного файла с учётом семафора."""
    async with semaphore:
        try:
            logger.debug(f"Скачивание {url} -> {file_path}")
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                resp.raise_for_status()
                async with aiofiles.open(file_path, 'wb') as f:
                    async for chunk in resp.content.iter_chunked(8192):
                        await f.write(chunk)

            # Если у файла нет расширения или оно не в списке разрешённых, добавляем .html
            # Важно: архивы не должны переименовываться
            ext = os.path.splitext(file_path)[1].lower()
            if ext not in DEFAULT_ALLOWED_EXTS and ext not in ARCHIVE_EXTS:
                new_path = file_path + '.html'
                os.rename(file_path, new_path)
                logger.debug(f"Файл без расширения переименован: {file_path} -> {new_path}")
                file_path = new_path

            stats["downloaded_files"] += 1

            # Конвертируем .doc -> .docx через LibreOffice (без доп. зависимостей)
            if os.path.splitext(file_path)[1].lower() == '.doc':
                txt_path = await convert_doc_to_txt(file_path)
                if txt_path:
                    stats.setdefault("converted_doc_to_txt", 0)
                    stats["converted_doc_to_txt"] += 1
                    try:
                        os.remove(file_path)
                        logger.debug(f"Удалён исходный .doc после конвертации: {file_path}")
                    except OSError as e:
                        logger.warning(f"Не удалось удалить исходный .doc {file_path}: {e}")

            # Проверяем, является ли файл архивом, и распаковываем
            extracted = await extract_archive_if_needed(file_path)
            if extracted > 0:
                stats["extracted_archives"] += 1
                stats["extracted_files"] += extracted
            elif extracted == -1:
                # ошибка распаковки уже залогирована внутри функции
                stats["errors"].append(f"tender {tender_number}: failed to extract {os.path.basename(file_path)}")

            logger.debug(f"Успешно скачан: {file_path}")
        except asyncio.TimeoutError:
            logger.error(f"Таймаут при скачивании {url}")
            stats["errors"].append(f"tender {tender_number}: {url} - timeout")
        except aiohttp.ClientError as e:
            logger.error(f"Ошибка HTTP при скачивании {url}: {e}")
            stats["errors"].append(f"tender {tender_number}: {url} - {str(e)}")
        except Exception as e:
            logger.exception(f"Неожиданная ошибка при скачивании {url}")
            stats["errors"].append(f"tender {tender_number}: {url} - unexpected error: {str(e)}")


if __name__ == "__main__":
    import sys
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Пример использования:
    # Укажите путь к вашему JSONL-файлу
    jsonl_file = r"F:\Python projects\Aut\storage\tenders\extracted_texts\all_tenders_data_20260327_011842.jsonl"
    download_dir = "downloaded_test"
    max_concurrent = 5
    overwrite = False
    timeout = 30

    async def main():
        stats = await download_tender_files_async(
            jsonl_path=jsonl_file,
            base_download_dir=download_dir,
            max_concurrent_files=max_concurrent,
            overwrite=overwrite,
            timeout=timeout
        )
        print(json.dumps(stats, indent=2, ensure_ascii=False))

    asyncio.run(main())