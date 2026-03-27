"""
Оркестратор пайплайна: почта -> ссылки на тендеры -> скачивание файлов -> извлечение текста.

Последовательность:
  1. email_agent: получение ссылок на тендеры из непрочитанных писем.
  2. file_parser: для каждой ссылки — авторизация, переход на страницу тендера, скачивание файлов.
  3. text_extractor: извлечение текста из скачанных PDF/DOCX и сохранение в отдельную папку.
  4. RAG-классификация (асинхронная, параллельные запросы к LLM).
  5. Глубокий анализ интересных тендеров (параллельный).
  6. Генерация отчётов и отправка по email.
"""
from datetime import datetime

import asyncio
import logging
import os
from logging.handlers import RotatingFileHandler
import shutil
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

from config import (
    AUTH_URL,
    DOWNLOAD_FOLDER,
    EXTRACTED_TEXT_FOLDER,
    JSONL_OUTPUT_FOLDER,
    LINKS_FILE,
    REPORTS_FOLDER,
    MAX_CONCURRENT_TENDERS,
    LOG_LEVEL,
    LOG_FILE,
    LOG_MAX_BYTES,
    LOG_BACKUP_COUNT,
    RAG_DATA_PATH,
    RAG_EMBEDDINGS_PATH,
    RAG_INDEX_PATH,
    RAG_K,
    RAG_FAISS_THRESHOLD,
    RAG_USE_INVERSE_FREQ,
    RAG_INVERSE_FREQ_MODE,
    RAG_LLM_CONFIG,
    COMPANY_PROFILE,
    RAG_DATA_CACHE_PATH,
    COMPANY_CONTEXT_DIR,
    RAG_INTEREST_THRESHOLD,
    ANALYSIS_MODEL,
    RAG_MODEL_NAME,
    FILES_KEYWORDS
)
from table_builder import generate_html_table, generate_excel_table
from email_agent import fetch_links_from_emails, send_files_via_email
from file_parser import RostenderSession
from text_extractor import process_folder
from load_files import download_tender_files_async

logger = logging.getLogger(__name__)

# Формат логов для всего приложения
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
# Проверка наличия RAG-модулей
try:
    from rag.rag_classifier import TenderClassifierRAG
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    logger.warning("Модуль rag_classifier не найден. RAG-классификация будет пропущена.")

try:
    from rag.deep_rag_anylise import TenderRAGAnalyzer
    DEEP_RAG_AVAILABLE = True
except ImportError:
    DEEP_RAG_AVAILABLE = False
    logger.warning("Модуль deep_rag_anylise не найден. Глубокий анализ документации будет пропущен.")

# Профиль компании для deep RAG (если не задан в config)
DEFAULT_COMPANY_PROFILE = (
    "Компания занимается поставкой, проектированием, монтажом и обслуживанием систем учёта "
    "электроэнергии и воды, АСКУЭ, поверкой счётчиков, а также сопутствующим электротехническим оборудованием."
)

# Конфигурация для параллельной обработки (можно вынести в config)
MAX_CONCURRENT_DEEP_RAG = MAX_CONCURRENT_TENDERS  # по умолчанию равно количеству параллельных тендеров


def _setup_logging(level=logging.INFO, log_file=None):
    """Настраивает комплексное логирование в консоль и опционально в файл."""
    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        # backupCount=0 => при переполнении старые логи удаляются/обнуляются
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=LOG_MAX_BYTES,
            backupCount=LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        handlers.append(file_handler)

    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )


def filter_jsonl_by_categories(
    input_jsonl: str,
    output_jsonl: Optional[str] = None,
    include_categories: Optional[List[str]] = None,
    exclude_categories: Optional[List[str]] = None,
    min_confidence: Optional[float] = None,
    confidence_field: str = "confidence",
    category_field: str = "predicted_category",
    interesting_only: bool = False,
    interesting_field: str = "is_interesting",
) -> List[Dict[str, Any]]:
    """
    Фильтрует JSONL-файл по категориям, уверенности классификации и/или признаку "интересен".
    Если output_jsonl указан, сохраняет отфильтрованные записи в файл.
    Возвращает список отфильтрованных записей.
    """
    if (include_categories is None and exclude_categories is None
            and min_confidence is None and not interesting_only):
        logger.warning("Не задано ни одного критерия фильтрации. Будет возвращён весь файл без изменений.")

    filtered_records = []
    total = 0
    skipped_no_category = 0
    skipped_by_include = 0
    skipped_by_exclude = 0
    skipped_by_confidence = 0
    skipped_by_interesting = 0

    try:
        with open(input_jsonl, 'r', encoding='utf-8') as infile:
            for line_num, line in enumerate(infile, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.error(f"Ошибка парсинга JSON в строке {line_num}: {e}")
                    continue

                total += 1

                # Фильтр по интересности
                if interesting_only:
                    interesting_val = record.get(interesting_field)
                    if interesting_val is not True:
                        skipped_by_interesting += 1
                        continue

                category = record.get(category_field)
                confidence = record.get(confidence_field)

                if category is None:
                    skipped_no_category += 1
                    continue

                if include_categories is not None and category not in include_categories:
                    skipped_by_include += 1
                    continue

                if exclude_categories is not None and category in exclude_categories:
                    skipped_by_exclude += 1
                    continue

                if min_confidence is not None:
                    if confidence is None:
                        skipped_by_confidence += 1
                        continue
                    try:
                        conf_value = float(confidence)
                        if conf_value < min_confidence:
                            skipped_by_confidence += 1
                            continue
                    except (ValueError, TypeError):
                        logger.warning(f"Некорректное значение confidence в строке {line_num}: {confidence}")
                        skipped_by_confidence += 1
                        continue

                filtered_records.append(record)

    except FileNotFoundError:
        logger.error(f"Файл не найден: {input_jsonl}")
        raise

    if output_jsonl and filtered_records:
        try:
            with open(output_jsonl, 'w', encoding='utf-8') as outfile:
                for record in filtered_records:
                    outfile.write(json.dumps(record, ensure_ascii=False) + '\n')
            logger.info(f"Сохранено {len(filtered_records)} записей в {output_jsonl}")
        except Exception as e:
            logger.error(f"Ошибка при записи в {output_jsonl}: {e}")
    elif output_jsonl and not filtered_records:
        logger.warning("Нет записей, соответствующих фильтру. Выходной файл не создан.")

    logger.info(
        f"Статистика фильтрации:\n"
        f"  Всего записей: {total}\n"
        f"  Пропущено (нет категории): {skipped_no_category}\n"
        f"  Пропущено (не в include): {skipped_by_include}\n"
        f"  Пропущено (в exclude): {skipped_by_exclude}\n"
        f"  Пропущено (confidence < {min_confidence}): {skipped_by_confidence}\n"
        f"  Пропущено (не интересно): {skipped_by_interesting}\n"
        f"  Оставлено: {len(filtered_records)}"
    )

    return filtered_records


def collect_json_to_jsonl(
    root_directory: str,
    output_file: str,
    json_filename: str = "data.json",
    recursive: bool = True
) -> Dict[str, Any]:
    """
    Рекурсивно проходит по всем поддиректориям в root_directory,
    находит файлы с именем json_filename (по умолчанию 'data.json'),
    читает их содержимое и добавляет каждую JSON-запись в выходной JSONL файл.
    """
    stats = {
        "total_files_found": 0,
        "total_records_added": 0,
        "failed_files": [],
        "output_file": output_file
    }

    logger.info(f"🔍 Начинаем сбор данных из JSON файлов в директории: {root_directory}")
    logger.info(f"📄 Ищем файлы с именем: {json_filename}")
    logger.info(f"📦 Выходной файл: {output_file}")

    os.makedirs(os.path.dirname(os.path.abspath(output_file)) or ".", exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as outfile:
        if recursive:
            for root, dirs, files in os.walk(root_directory):
                if json_filename in files:
                    json_path = os.path.join(root, json_filename)
                    try:
                        with open(json_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)

                        if isinstance(data, list):
                            for item in data:
                                outfile.write(json.dumps(item, ensure_ascii=False) + '\n')
                                stats["total_records_added"] += 1
                        else:
                            outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                            stats["total_records_added"] += 1

                        stats["total_files_found"] += 1
                        logger.debug(f"Обработан: {json_path}")

                    except Exception as e:
                        logger.error(f"Ошибка чтения {json_path}: {e}")
                        stats["failed_files"].append({"path": json_path, "error": str(e)})
        else:
            for item in os.listdir(root_directory):
                item_path = os.path.join(root_directory, item)
                if os.path.isdir(item_path):
                    json_path = os.path.join(item_path, json_filename)
                    if os.path.exists(json_path):
                        try:
                            with open(json_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)

                            if isinstance(data, list):
                                for entry in data:
                                    outfile.write(json.dumps(entry, ensure_ascii=False) + '\n')
                                    stats["total_records_added"] += 1
                            else:
                                outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                                stats["total_records_added"] += 1

                            stats["total_files_found"] += 1
                            logger.debug(f"Обработан: {json_path}")

                        except Exception as e:
                            logger.error(f"Ошибка чтения {json_path}: {e}")
                            stats["failed_files"].append({"path": json_path, "error": str(e)})

    logger.info("="*50)
    logger.info(f"📊 Статистика сбора данных:")
    logger.info(f"   Найдено JSON файлов: {stats['total_files_found']}")
    logger.info(f"   Добавлено записей в JSONL: {stats['total_records_added']}")
    logger.info(f"   Проблемных файлов: {len(stats['failed_files'])}")
    logger.info(f"   Выходной файл: {stats['output_file']}")

    if stats['failed_files']:
        logger.warning("⚠️ Проблемные файлы:")
        for failed in stats['failed_files'][:5]:
            logger.warning(f"   - {failed['path']}: {failed['error']}")
        if len(stats['failed_files']) > 5:
            logger.warning(f"   ... и ещё {len(stats['failed_files']) - 5} ошибок")

    logger.info("="*50)

    return stats


def collect_tenders_data_to_jsonl(
    input_folder: str = DOWNLOAD_FOLDER,
    output_folder: str = EXTRACTED_TEXT_FOLDER,
    output_filename: str = "all_tenders_data.jsonl"
) -> Optional[str]:
    """
    Удобная обертка для сбора всех data.json из папки с извлеченными текстами
    в один JSONL файл.
    """
    output_path = os.path.join(output_folder, output_filename)

    try:
        stats = collect_json_to_jsonl(
            root_directory=input_folder,
            output_file=output_path,
            json_filename="data.json",
            recursive=True
        )

        if stats["total_files_found"] > 0:
            logger.info(f"✅ Данные тендеров успешно собраны в {output_path}")
            return output_path
        else:
            logger.warning(f"⚠️ Не найдено файлов data.json в {input_folder}")
            return None

    except Exception as e:
        logger.error(f"❌ Ошибка при сборе данных тендеров: {e}")
        return None


def cleanup_folder(folder_path: str, keep_jsonl: bool = False):
    """Очищает папку: удаляет всё содержимое, затем создаёт пустую папку заново."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        logger.info("Папка %s создана", folder_path)
        return

    if not keep_jsonl:
        try:
            shutil.rmtree(folder_path)
            logger.info("Папка %s удалена", folder_path)
        except Exception as e:
            logger.error("Ошибка при удалении папки %s: %s", folder_path, e)
        os.makedirs(folder_path, exist_ok=True)
        logger.info("Папка %s создана заново (пустая)", folder_path)
        return

    # Режим сохранения .jsonl файлов (только в корне)
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path):
            if not item.lower().endswith('.jsonl'):
                os.remove(item_path)
                logger.debug("Удалён файл: %s", item)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)
            logger.debug("Удалена папка: %s", item)

    logger.info("Папка %s очищена, все .jsonl файлы из корня сохранены", folder_path)


async def run_pipeline(
    headless=True,
    links_file=None,
    max_pages=None,
    log_file=LOG_FILE,
):
    """
    Запускает полный пайплайн: почта -> ссылки -> скачивание -> извлечение текста -> классификация -> deep RAG.
    """
    level = getattr(logging, str(LOG_LEVEL).upper(), logging.INFO)
    _setup_logging(level=level, log_file=log_file)
    links_file = links_file or LINKS_FILE

    logger.info("Старт пайплайна: почта -> ссылки -> скачивание -> извлечение текста")

    # 1. Получение ссылок из писем
    logger.info("Этап 1: получение ссылок на тендеры из почты")
    urls = fetch_links_from_emails(save_to_file=links_file)
    if not urls:
        logger.warning("Ссылок на тендеры не найдено. Пайплайн завершён.")
        return

    logger.info("Получено уникальных ссылок на тендеры: %s", len(urls))

    # 2. Скачивание метаданных по каждой ссылке
    os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)
    os.makedirs(EXTRACTED_TEXT_FOLDER, exist_ok=True)
    os.makedirs(COMPANY_CONTEXT_DIR, exist_ok=True)

    async with RostenderSession(AUTH_URL, headless=headless) as session:
        sem = asyncio.Semaphore(MAX_CONCURRENT_TENDERS)

        async def handle_tender(tender_id_int, url):
            tender_id = str(tender_id_int)
            async with sem:
                download_subfolder = os.path.join(DOWNLOAD_FOLDER, tender_id)
                logger.info("Обработка тендера %s: %s", tender_id, url)
                await session.get_tender_info(url, download_subfolder)

        tasks = [asyncio.create_task(handle_tender(i, url)) for i, url in enumerate(urls, start=1)]
        if tasks:
            await asyncio.gather(*tasks)

        # 3. Сбор всех data.json в один JSONL файл
        logger.info("Начинаем сбор данных из JSON файлов в JSONL формат...")
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        jsonl_filename = f"all_tenders_data_{timestamp_str}.jsonl"
        jsonl_path = collect_tenders_data_to_jsonl(
            input_folder=DOWNLOAD_FOLDER,
            output_folder=JSONL_OUTPUT_FOLDER,
            output_filename=jsonl_filename
        )
        cleanup_folder(DOWNLOAD_FOLDER)

        # 4. RAG-классификация (асинхронная, параллельная)
        if RAG_AVAILABLE and jsonl_path and os.path.exists(jsonl_path):
            logger.info("Этап 4: RAG-классификация тендеров (асинхронная, параллельная)")
            try:
                classifier = TenderClassifierRAG(
                    data_path=RAG_DATA_PATH,
                    embeddings_path=RAG_EMBEDDINGS_PATH,
                    index_path=RAG_INDEX_PATH,
                    k=RAG_K,
                    data_cache_path=RAG_DATA_CACHE_PATH,
                    faiss_threshold=RAG_FAISS_THRESHOLD,
                    use_inverse_freq=RAG_USE_INVERSE_FREQ,
                    inverse_freq_mode=RAG_INVERSE_FREQ_MODE,
                    llm_config=RAG_LLM_CONFIG,
                    company_context_dir=COMPANY_CONTEXT_DIR,
                    interest_threshold=RAG_INTEREST_THRESHOLD
                )
                # Используем асинхронный метод с параллельностью
                await classifier.process_file_async(jsonl_path, jsonl_path, concurrency=MAX_CONCURRENT_TENDERS)
            except Exception as e:
                logger.error(f"Ошибка при RAG-классификации: {e}", exc_info=True)
        else:
            logger.warning("RAG-классификация недоступна или файл JSONL отсутствует")

        # 5. Загружаем все записи из JSONL в память
        if not jsonl_path or not os.path.exists(jsonl_path):
            logger.error("Нет JSONL-файла для дальнейшей обработки")
            return

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            all_records = [json.loads(line) for line in f if line.strip()]

        # 6. Выделяем интересные тендеры
        interesting_records = [rec for rec in all_records if rec.get('is_interesting') is True]

        # 7. Если есть интересные – скачиваем файлы, извлекаем текст и выполняем deep RAG (параллельно)
        if interesting_records and DEEP_RAG_AVAILABLE:
            logger.info(f"Найдено {len(interesting_records)} интересных тендеров. Запуск глубокого анализа...")
            # Создаём временный JSONL для скачивания и извлечения текста
            temp_jsonl = os.path.join(JSONL_OUTPUT_FOLDER, "interesting_temp.jsonl")
            with open(temp_jsonl, 'w', encoding='utf-8') as f:
                for rec in interesting_records:
                    f.write(json.dumps(rec, ensure_ascii=False) + '\n')

            # Скачивание файлов (только для интересных) – асинхронное
            await download_tender_files_async(temp_jsonl, DOWNLOAD_FOLDER, max_concurrent_files=5)

            # Извлечение текста из скачанных файлов – асинхронное
            await process_folder(DOWNLOAD_FOLDER, EXTRACTED_TEXT_FOLDER, keywords=FILES_KEYWORDS)

            # ========== БЛОК DEEP RAG ==========
            logger.info("Запуск глубокого RAG анализа для интересных тендеров...")
            company_profile = COMPANY_PROFILE or DEFAULT_COMPANY_PROFILE

            # Создаём словарь для быстрого доступа по номеру тендера (из all_records)
            all_by_number = {rec.get("number"): rec for rec in all_records if rec.get("number")}

            # Анализируем каждый интересный тендер
            for record in interesting_records:
                tender_id = record.get("number")
                if not tender_id:
                    continue

                target_rec = all_by_number.get(tender_id)
                if target_rec is None:
                    logger.warning("Тендер %s не найден в общем списке, пропускаем", tender_id)
                    continue

                docs_folder = os.path.join(EXTRACTED_TEXT_FOLDER, str(tender_id))
                if not os.path.isdir(docs_folder):
                    logger.warning("Папка с документацией не найдена для тендера %s: %s", tender_id, docs_folder)
                    target_rec["deep_rag_score"] = 0.0
                    target_rec["deep_rag_decision"] = "Не подходит"
                    target_rec["deep_rag_reasoning"] = "Нет извлечённой документации."
                    continue

                try:
                    tender_category = target_rec.get("predicted_category")

                    def _run_deep_rag():
                        analyzer = TenderRAGAnalyzer(docs_folder)
                        suitability = analyzer.evaluate_suitability(company_profile, tender_category=tender_category)
                        qa = analyzer.analyze_all_questions(
                            questions=None,
                            tender_category=tender_category,
                        )
                        return suitability, qa

                    suitability, qa_answers = await asyncio.to_thread(_run_deep_rag)
                    target_rec["deep_rag_score"] = suitability["suitability_score"]
                    target_rec["deep_rag_decision"] = suitability["decision"]
                    target_rec["deep_rag_reasoning"] = suitability.get("reasoning", "")
                    target_rec["deep_rag_answers"] = qa_answers
                    logger.info(
                        "Тендер %s: %s (оценка %.2f)",
                        tender_id,
                        suitability["decision"],
                        suitability["suitability_score"],
                    )
                except Exception as e:
                    logger.exception("Ошибка deep RAG для тендера %s: %s", tender_id, e)
                    target_rec["deep_rag_score"] = 0.0
                    target_rec["deep_rag_decision"] = "На грани"
                    target_rec["deep_rag_reasoning"] = f"Ошибка анализа: {e}"

            # Обновлённые записи уже в all_records (через target_rec)
            logger.info("Deep RAG анализ завершён для %d интересных тендеров", len(interesting_records))
            # ========== КОНЕЦ БЛОКА DEEP RAG ==========

            # Удаляем временный JSONL
            os.remove(temp_jsonl)
            # ========== КОНЕЦ ДОБАВЛЕННОГО БЛОКА ==========

        elif interesting_records and not DEEP_RAG_AVAILABLE:
            logger.warning("Deep RAG недоступен, пропускаем глубокий анализ.")
        else:
            logger.info("Нет интересных тендеров для глубокого анализа.")

        # 8. Для всех записей (включая неинтересные) проставляем поля deep_rag_*, если их ещё нет
        for rec in all_records:
            if "deep_rag_score" not in rec:
                rec["deep_rag_score"] = 0.0
                rec["deep_rag_decision"] = "Не подходит"
                rec["deep_rag_reasoning"] = "Тендер не интересен, глубокий анализ не проводился."
                rec["deep_rag_answers"] = {}

        # 9. Перезаписываем общий JSONL с обновлёнными полями
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for rec in all_records:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        logger.info("Общий JSONL обновлён с полями deep_rag_*")

        # 10. Генерация отчётов из общего JSONL
        logger.info("Формируем HTML и Excel таблицы из общего JSONL")
        os.makedirs(REPORTS_FOLDER, exist_ok=True)
        html_table_path = os.path.join(REPORTS_FOLDER, f"tenders_summary_{timestamp_str}.html")
        xlsx_path = os.path.join(REPORTS_FOLDER, f"tenders_summary_{timestamp_str}.xlsx")
        generate_html_table(
            input_dir=JSONL_OUTPUT_FOLDER,
            output_html_path=html_table_path,
            jsonl_filename=jsonl_filename
        )
        generate_excel_table(
            input_dir=JSONL_OUTPUT_FOLDER,
            output_xlsx_path=xlsx_path,
            jsonl_filename=jsonl_filename
        )
        logger.info(f"📊 HTML-таблица создана: {html_table_path}")
        logger.info(f"📊 Excel-таблица создана: {xlsx_path}")

        # 11. Отправка файлов по email
        files_to_send = [html_table_path, xlsx_path]
        logger.info(f"📧 Отправка файлов по email: {files_to_send}")
        send_files_via_email(
            files_to_send,
            subject=f"Сводка тендеров {datetime.now().date().strftime('%d/%m/%Y')} (HTML + Excel)"
        )

        # 12. Очистка временных папок (опционально)
        cleanup_folder(DOWNLOAD_FOLDER)
        cleanup_folder(EXTRACTED_TEXT_FOLDER)

    logger.info("Пайплайн завершён. Тексты сохранены в: %s", os.path.abspath(EXTRACTED_TEXT_FOLDER))


def main():
    import sys
    headless = False
    asyncio.run(run_pipeline(headless=headless))


if __name__ == "__main__":
    main()