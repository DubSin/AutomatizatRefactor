import os
import re
import logging
import asyncio
import zipfile
from typing import List, Optional
from docx import Document
import pandas as pd
import pdfplumber
from bs4 import BeautifulSoup
from config import MAX_PDF_PAGES, MAX_DOCX_PARAGRAPHS, MAX_CONCURRENT_EXTRACT, PROCESS_DOC

# Попытка импорта rarfile для работы с RAR-архивами
try:
    import rarfile
    RARFILE_AVAILABLE = True
except ImportError:
    rarfile = None
    RARFILE_AVAILABLE = False
    # logger будет определён позже, поэтому пока просто заглушка
    pass

logger = logging.getLogger(__name__)

# Поддерживаемые расширения (добавлен .rar)
SUPPORTED_EXTENSIONS = (
    ".pdf", ".docx", ".txt", ".json", ".xlsx", ".xls", ".html", ".rar"
) + ((".doc",) if PROCESS_DOC else ())

# ------------------------------------------------------------
# Санитизация имени файла (удаление недопустимых символов)
# ------------------------------------------------------------
def sanitize_filename(filename: str, replacement: str = "_") -> str:
    """
    Заменяет в имени файла символы, недопустимые в файловых системах, на replacement.
    Также удаляет управляющие символы и обрезает лишние пробелы в конце.
    """
    illegal_chars = r'[\\/*?:"<>|\x00-\x1f]'
    sanitized = re.sub(illegal_chars, replacement, filename)
    sanitized = sanitized.rstrip('. ')
    if not sanitized:
        sanitized = "unnamed" + replacement
    return sanitized

def rename_files_in_folder(folder_path: str, dry_run: bool = False) -> List[tuple]:
    """
    Рекурсивно обходит папку и переименовывает все файлы и папки,
    применяя sanitize_filename к их именам. Если dry_run=True, только возвращает список
    предполагаемых изменений, не переименовывая.
    Возвращает список кортежей (old_path, new_path).
    """
    changes = []
    for root, dirs, files in os.walk(folder_path, topdown=False):
        # Сначала файлы
        for name in files:
            old_path = os.path.join(root, name)
            new_name = sanitize_filename(name)
            if new_name == name:
                continue
            new_path = os.path.join(root, new_name)
            counter = 1
            base, ext = os.path.splitext(new_name)
            while os.path.exists(new_path):
                new_path = os.path.join(root, f"{base}_{counter}{ext}")
                counter += 1
            changes.append((old_path, new_path))
            if not dry_run:
                os.rename(old_path, new_path)
                logger.info("Переименован файл: %s -> %s", old_path, new_path)

        # Затем папки
        for name in dirs:
            old_path = os.path.join(root, name)
            new_name = sanitize_filename(name)
            if new_name == name:
                continue
            new_path = os.path.join(root, new_name)
            counter = 1
            base = new_name
            while os.path.exists(new_path):
                new_path = os.path.join(root, f"{base}_{counter}")
                counter += 1
            changes.append((old_path, new_path))
            if not dry_run:
                os.rename(old_path, new_path)
                logger.info("Переименована папка: %s -> %s", old_path, new_path)
    return changes

# ------------------------------------------------------------
# Проверка валидности DOCX
# ------------------------------------------------------------
def is_valid_docx(file_path: str) -> bool:
    """Проверяет, является ли файл корректным ZIP-архивом с [Content_Types].xml."""
    try:
        with zipfile.ZipFile(file_path, 'r') as zf:
            return '[Content_Types].xml' in zf.namelist()
    except zipfile.BadZipFile:
        return False

# ------------------------------------------------------------
# Работа с архивами (RAR)
# ------------------------------------------------------------
def is_archive(file_path: str) -> bool:
    """Проверяет, является ли файл архивом (по расширению)."""
    ext = os.path.splitext(file_path)[1].lower()
    return ext == '.rar'

def extract_rar(rar_path: str, extract_to: str) -> bool:
    """
    Извлекает RAR-архив в указанную папку.
    Возвращает True в случае успеха, False при ошибке или отсутствии поддержки.
    """
    if not RARFILE_AVAILABLE:
        logger.warning("rarfile не доступен, пропускаем извлечение %s", rar_path)
        return False

    try:
        os.makedirs(extract_to, exist_ok=True)
        with rarfile.RarFile(rar_path) as rf:
            rf.extractall(extract_to)
        logger.info("Распакован RAR: %s -> %s", rar_path, extract_to)
        return True
    except Exception as e:
        logger.error("Ошибка при извлечении RAR %s: %s", rar_path, e)
        return False

# ------------------------------------------------------------
# Очистка текста
# ------------------------------------------------------------
def clean_text(text, short_line_threshold=10, preserve_structure: bool = False):
    """
    Очищает текст от лишних пробелов, пустых строк и отступов.

    Если preserve_structure=True — старается не ломать исходную разметку
    (заголовки, таблицы, списки), не склеивает короткие строки.
    """
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        line = line.rstrip()
        norm = re.sub(r'[ \t]+', ' ', line)
        cleaned.append(norm)

    if preserve_structure:
        return "\n".join(cleaned)

    non_empty = [l.strip() for l in cleaned if l.strip()]

    merged = []
    for line in non_empty:
        if not merged:
            merged.append(line)
        else:
            if len(line) <= short_line_threshold:
                merged[-1] += " " + line
            else:
                merged.append(line)
    return '\n'.join(merged)

# ------------------------------------------------------------
# Преобразование таблиц в Markdown
# ------------------------------------------------------------
def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Преобразует pandas DataFrame в строку Markdown-таблицы."""
    if df.empty:
        return ""
    header = "| " + " | ".join(str(col) for col in df.columns) + " |"
    separator = "|" + "|".join([" --- "] * len(df.columns)) + "|"
    rows = []
    for _, row in df.iterrows():
        row_str = [str(val) if pd.notna(val) else "" for val in row]
        rows.append("| " + " | ".join(row_str) + " |")
    return "\n".join([header, separator] + rows)

def table_to_markdown(table_data) -> str:
    """
    Универсальная функция: принимает таблицу в виде списка списков или DataFrame,
    возвращает Markdown-строку.
    """
    if isinstance(table_data, pd.DataFrame):
        return dataframe_to_markdown(table_data)
    elif isinstance(table_data, list) and all(isinstance(row, list) for row in table_data):
        if not table_data:
            return ""
        if len(table_data) > 1:
            df = pd.DataFrame(table_data[1:], columns=table_data[0])
        else:
            df = pd.DataFrame(table_data)
        return dataframe_to_markdown(df)
    else:
        return ""

# ------------------------------------------------------------
# Извлечение таблиц из разных форматов
# ------------------------------------------------------------
def extract_tables_from_pdf(pdf_path: str, max_pages: Optional[int] = None) -> List[str]:
    """Извлекает таблицы из PDF с помощью pdfplumber и возвращает список Markdown-строк."""
    tables_md = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                if max_pages and page_num >= max_pages:
                    break
                tables = page.extract_tables()
                for table in tables:
                    if table and len(table) > 1:
                        md = table_to_markdown(table)
                        if md:
                            tables_md.append(f"\n## Таблица со страницы {page_num+1}\n\n{md}\n")
    except Exception as e:
        logger.debug(f"Ошибка при извлечении таблиц из PDF {pdf_path}: {e}")
    return tables_md

def extract_tables_from_docx(docx_path: str) -> List[str]:
    """Извлекает таблицы из DOCX с помощью python-docx."""
    tables_md = []
    try:
        doc = Document(docx_path)
        for table_idx, table in enumerate(doc.tables):
            data = []
            for row in table.rows:
                data.append([cell.text.strip() for cell in row.cells])
            if data:
                md = table_to_markdown(data)
                if md:
                    tables_md.append(f"\n## Таблица {table_idx+1}\n\n{md}\n")
    except Exception as e:
        logger.debug(f"Ошибка при извлечении таблиц из DOCX {docx_path}: {e}")
    return tables_md

def extract_tables_from_excel(excel_path: str) -> List[str]:
    """Извлекает все листы Excel как таблицы в Markdown."""
    tables_md = []
    try:
        xl = pd.ExcelFile(excel_path)
        for sheet_name in xl.sheet_names:
            df = xl.parse(sheet_name)
            if not df.empty:
                md = dataframe_to_markdown(df)
                if md:
                    tables_md.append(f"\n## Лист: {sheet_name}\n\n{md}\n")
    except Exception as e:
        logger.debug(f"Ошибка при извлечении таблиц из Excel {excel_path}: {e}")
    return tables_md

def extract_text_from_html(html_path: str) -> str:
    """Извлекает чистый текст из HTML-файла с помощью BeautifulSoup."""
    try:
        with open(html_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'lxml')
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text(separator='\n')
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return '\n'.join(lines)
    except Exception as e:
        logger.debug(f"Ошибка при извлечении текста из HTML {html_path}: {e}")
        return ""

# ------------------------------------------------------------
# Основная функция извлечения текста (для одного файла)
# ------------------------------------------------------------
def extract_text(file_path: str, max_pages: Optional[int] = None) -> str:
    """
    Извлекает текст и таблицы (в Markdown) из файла.
    Возвращает содержимое, готовое для сохранения в .md.
    """
    if not os.path.isfile(file_path):
        logger.error(f"Файл не найден: {file_path}")
        raise FileNotFoundError(f"Файл не найден: {file_path}")

    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    # Для JSON возвращаем блок кода
    if ext == '.json':
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return f"```json\n{content}\n```"
        except Exception as e:
            logger.exception(f"Ошибка при чтении JSON {file_path}: {e}")
            raise

    # Для TXT просто читаем (может быть Markdown или обычный текст)
    if ext == '.txt':
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.exception(f"Ошибка при чтении TXT {file_path}: {e}")
            raise

    # HTML
    if ext in ('.html', '.htm'):
        return extract_text_from_html(file_path)

    # PDF
    if ext == '.pdf':
        pages_limit = max_pages if max_pages is not None else MAX_PDF_PAGES
        full_text = ""

        # 1. Извлекаем текст через pdfplumber с учётом макета страниц
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    if pages_limit and page_num >= pages_limit:
                        break
                    page_text = page.extract_text(layout=True) or page.extract_text() or ""
                    if page_text.strip():
                        full_text += f"\n\n## Страница {page_num + 1}\n\n"
                        full_text += page_text + "\n"
        except Exception as e:
            logger.exception(f"Ошибка при чтении текста из PDF {file_path}: {e}")

        # 2. Извлекаем таблицы через pdfplumber
        tables_md = extract_tables_from_pdf(file_path, max_pages=pages_limit)
        if tables_md:
            full_text += "\n\n## Извлечённые таблицы\n\n"
            full_text += "\n".join(tables_md)

        return clean_text(full_text, preserve_structure=True)

    # DOCX
    if ext == '.docx':
        full_text_lines = []

        # Проверка валидности DOCX
        if not is_valid_docx(file_path):
            logger.warning(f"Файл {file_path} имеет расширение .docx, но не является валидным DOCX. "
                           f"Возможно, это старый .doc. Попытка извлечения через Word (если включено).")
            if PROCESS_DOC:
                try:
                    text = _extract_doc_via_word(file_path)
                    return clean_text(text, preserve_structure=True)
                except Exception as e:
                    logger.warning(f"Не удалось извлечь текст через Word: {e}")
                    return ""
            else:
                logger.warning("PROCESS_DOC отключён, пропускаем файл.")
                return ""

        # Основное извлечение из DOCX
        try:
            doc = Document(file_path)
            limit = max_pages if max_pages is not None else MAX_DOCX_PARAGRAPHS
            for p in doc.paragraphs[:limit]:
                text = p.text.strip()
                if not text:
                    continue
                style_name = (p.style.name or "").lower() if p.style else ""

                # Преобразуем стили Heading в Markdown-заголовки
                level = None
                if "heading" in style_name:
                    m = re.search(r'heading\s*(\d+)', style_name)
                    if m:
                        try:
                            level = int(m.group(1))
                        except ValueError:
                            level = None
                if level is not None and 1 <= level <= 6:
                    prefix = "#" * level
                    full_text_lines.append(f"{prefix} {text}")
                else:
                    full_text_lines.append(text)
        except Exception as e:
            logger.warning(f"Ошибка при чтении текста из DOCX {file_path}: {e}")
            full_text_lines = []

        full_text = "\n".join(full_text_lines) + "\n"

        # Таблицы
        tables_md = extract_tables_from_docx(file_path)
        if tables_md:
            full_text += "\n\n## Извлечённые таблицы\n\n"
            full_text += "\n".join(tables_md)

        return clean_text(full_text, preserve_structure=True)

    # Excel (.xlsx, .xls)
    if ext in ('.xlsx', '.xls'):
        tables_md = extract_tables_from_excel(file_path)
        if tables_md:
            full_text = "\n\n".join(tables_md)
            return full_text.strip()
        else:
            return ""

    # DOC (если включён)
    if ext == '.doc' and PROCESS_DOC:
        try:
            text = _extract_doc_via_word(file_path)
            return clean_text(text, preserve_structure=True)
        except Exception as e:
            logger.exception(f"Ошибка при чтении DOC {file_path}: {e}")
            raise

    # Если файл .rar (архив) – не обрабатываем как текст, а только как архив
    if ext == '.rar':
        logger.warning("extract_text вызван для архива %s, но архивы обрабатываются отдельно. Возвращаем пустую строку.", file_path)
        return ""

    raise ValueError(f"Неподдерживаемый формат файла: {ext}")

def _extract_doc_via_word(file_path):
    """Извлекает текст из .doc через Microsoft Word (COM automation)."""
    try:
        import pythoncom
        import win32com.client
    except ImportError:
        raise RuntimeError(
            "Для обработки .doc нужен MS Word и пакет pywin32. "
            "Установите зависимость: pip install pywin32"
        )

    abs_path = os.path.abspath(file_path)
    pythoncom.CoInitialize()
    try:
        word = win32com.client.Dispatch("Word.Application")
        word.Visible = False
        word.DisplayAlerts = 0
        doc = word.Documents.Open(abs_path, ReadOnly=True)
        try:
            return doc.Content.Text or ""
        finally:
            doc.Close(False)
            word.Quit()
    finally:
        pythoncom.CoUninitialize()

# ------------------------------------------------------------
# Асинхронная обёртка
# ------------------------------------------------------------
async def extract_text_async(file_path, max_pages=None):
    return await asyncio.to_thread(extract_text, file_path, max_pages)

# ------------------------------------------------------------
# Пакетная обработка папки (process_folder) с поддержкой архивов
# ------------------------------------------------------------
async def process_folder(input_folder, output_folder, max_pages=None, concurrency=None,
                         sanitize_first=False, extract_archives=True):
    """
    Обходит папку input_folder, извлекает текст из всех поддерживаемых файлов
    и сохраняет результат в output_folder с расширением .md.
    В начало каждого файла добавляется заголовок с именем исходного файла.

    Если sanitize_first=True, то перед обработкой переименовывает все файлы и папки,
    удаляя недопустимые символы из имён.

    Если extract_archives=True, то файлы с расширениями архивов (.rar) будут
    распакованы в подпапку рядом с архивом, а затем содержимое будет обработано рекурсивно.
    """
    input_folder = os.path.abspath(input_folder)
    output_folder = os.path.abspath(output_folder)

    if sanitize_first:
        logger.info("Запуск санитизации имён файлов в %s", input_folder)
        changes = rename_files_in_folder(input_folder, dry_run=False)
        logger.info("Переименовано %s элементов.", len(changes))

    created = []

    if concurrency is None:
        concurrency = MAX_CONCURRENT_EXTRACT
    sem = asyncio.Semaphore(concurrency)

    def _extract_and_write(src_path, out_path):
        """Извлекает текст и сохраняет в .md с заголовком."""
        filename = os.path.basename(src_path)
        header = f"# Файл: {filename}\n\n"
        text = extract_text(src_path, max_pages)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(header + text)
        return out_path

    async def _handle_file(src_path, out_path):
        async with sem:
            try:
                saved_path = await asyncio.to_thread(_extract_and_write, src_path, out_path)
                created.append(saved_path)
                logger.debug("Сохранён файл: %s", saved_path)
            except Exception as e:
                logger.exception("Ошибка обработки %s: %s", src_path, e)

    async def _handle_archive(archive_path):
        """Распаковывает архив и рекурсивно обрабатывает его содержимое."""
        archive_dir = os.path.dirname(archive_path)
        archive_basename = os.path.basename(archive_path)
        name_without_ext = os.path.splitext(archive_basename)[0]
        extract_to = os.path.join(archive_dir, name_without_ext)

        # Если папка уже существует, считаем, что архив уже распакован (пропускаем повторное извлечение)
        if os.path.exists(extract_to):
            logger.info("Папка %s уже существует, предполагаем, что архив %s уже распакован.", extract_to, archive_path)
        else:
            logger.info("Распаковка архива %s в %s", archive_path, extract_to)
            success = await asyncio.to_thread(extract_rar, archive_path, extract_to)
            if not success:
                logger.error("Не удалось распаковать архив %s", archive_path)
                return

        # Рекурсивно обрабатываем извлечённую папку с теми же параметрами
        await process_folder(
            extract_to,
            output_folder,
            max_pages=max_pages,
            concurrency=concurrency,
            sanitize_first=False,       # санитизация уже выполнена на верхнем уровне
            extract_archives=extract_archives
        )

    # Собираем задачи для обычных файлов и для архивов
    tasks = []
    archive_tasks = []

    for root, _dirs, files in os.walk(input_folder):
        for name in files:
            base, ext = os.path.splitext(name)
            ext_lower = ext.lower()
            src_path = os.path.join(root, name)

            # Если это архив и включена распаковка — обрабатываем отдельно
            if extract_archives and ext_lower == '.rar':
                archive_tasks.append(asyncio.create_task(_handle_archive(src_path)))
                continue

            # Иначе проверяем, поддерживается ли формат для прямого извлечения текста
            if ext_lower not in SUPPORTED_EXTENSIONS or ext_lower == '.rar':
                continue

            # Выходное расширение всегда .md
            out_ext = '.md'
            rel = os.path.relpath(root, input_folder)
            out_dir = output_folder if rel == "." else os.path.join(output_folder, rel)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, base + out_ext)
            tasks.append(asyncio.create_task(_handle_file(src_path, out_path)))

    # Запускаем обработку файлов и архивов параллельно
    if tasks or archive_tasks:
        await asyncio.gather(*(tasks + archive_tasks))

    logger.info("Обработано файлов: %s, сохранено в %s", len(created), output_folder)
    return created

# ------------------------------------------------------------
# Для тестирования
# ------------------------------------------------------------
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    # Пример: обработать папку "input" и сохранить в "output_md"
    asyncio.run(process_folder("input", "output_md", max_pages=10, sanitize_first=True, extract_archives=True))

if __name__ == "__main__":
    main()