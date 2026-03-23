"""
RAG-анализ документации тендеров.
Для папки с .md файлами строит векторную БД, задаёт вопросы и получает ответы через Ollama.
"""
import os
import json
import logging
from typing import List, Dict, Any, Optional
from functools import lru_cache

try:
    import numpy as np
except ImportError:
    np = None

# Для работы с эмбеддингами и векторным поиском
try:
    import chromadb
    from chromadb.errors import NotFoundError
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

# Для разбиения текста (опционально langchain)
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# Для BM25
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

# Для прогресс-бара
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    tqdm = None
    TQDM_AVAILABLE = False

# Используем те же вспомогательные функции, что и в RAG-классификаторе
from .rag_classifier import ask_llama_sync, repair_json_fragment
from config import (
    ANALYSIS_MODEL,
    COMPANY_CONTEXT_DIR,
    RAG_MODEL_NAME,
    CATEGORY_QUESTIONS,
    QUERY_EXPANSION,
    HIGH_PRIORITY_KEYWORDS,
    MEDIUM_PRIORITY_KEYWORDS,
    LOW_PRIORITY_KEYWORDS
)

logger = logging.getLogger(__name__)

# Типовые вопросы (можно переопределить через конфиг)
DEFAULT_QUESTIONS = [
    "Дата начала срока подачи заявок?",
    "Какие требования предъявляются к участникам?",
    "Какие документы необходимо предоставить для участия?",
    "Каковы условия оплаты?",
    "Какие позиции включены в тендер (перечень товаров/работ)?"
]


class TenderRAGAnalyzer:
    """
    Класс для RAG-анализа одного тендера.
    """

    # Кэширование модели на уровне класса
    @staticmethod
    @lru_cache(maxsize=2)
    def _get_embedding_model(model_name: str):
        """Загружает модель SentenceTransformer и кэширует её."""
        logger.info(f"Загрузка модели эмбеддингов {model_name}...")
        model = SentenceTransformer(model_name)
        logger.info("Модель эмбеддингов загружена.")
        return model

    def __init__(
            self,
            docs_folder: str,
            embedding_model_name: str = RAG_MODEL_NAME,
            chunk_size: int = 1000,
            chunk_overlap: int = 200,
            top_k: int = 10,
            llm_model: str = ANALYSIS_MODEL,
            persist_directory: Optional[str] = None,  # по умолчанию in-memory (без хранения на диске)
            company_context_dir: Optional[str] = COMPANY_CONTEXT_DIR,
            use_stemmer: bool = True,
            query_expansion: bool = QUERY_EXPANSION,
            high_priority=HIGH_PRIORITY_KEYWORDS,
            medium_priority=MEDIUM_PRIORITY_KEYWORDS,
            low_priority=LOW_PRIORITY_KEYWORDS,
            show_progress: bool = False,  # флаг для отображения прогресс-бара
            max_chunks: int = 500          # ограничение на количество чанков
    ):
        """
        Параметры:
            docs_folder: путь к папке с .md файлами тендера
            embedding_model_name: имя модели sentence-transformers
            chunk_size: размер чанка в символах
            chunk_overlap: перекрытие чанков
            top_k: количество чанков, возвращаемых для контекста
            llm_model: модель Ollama для генерации ответов
            persist_directory: если указать, будет использована персистентная БД (иначе in-memory)
            company_context_dir: папка с дополнительными документами компании (правила и т.п.)
            use_stemmer: использовать ли стемминг для BM25
            query_expansion: расширять ли запрос через LLM
            high_priority, medium_priority, low_priority: списки ключевых слов для бустинга
            show_progress: показывать ли прогресс-бар при создании эмбеддингов
            max_chunks: максимальное количество чанков для индексации (для предотвращения перегрузки)
        """
        if not CHROMA_AVAILABLE:
            raise ImportError("chromadb не установлен. Установите: pip install chromadb")
        if not EMBEDDINGS_AVAILABLE:
            raise ImportError("sentence-transformers не установлен. Установите: pip install sentence-transformers")

        self.high_priority = high_priority
        self.medium_priority = medium_priority
        self.low_priority = low_priority
        self.query_expansion = query_expansion
        self.docs_folder = docs_folder
        self.embedding_model_name = embedding_model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.llm_model = llm_model
        self.company_context_dir = company_context_dir
        self.use_stemmer = use_stemmer
        self.show_progress = show_progress
        self.max_chunks = max_chunks

        # Инициализация модели через кэширующий метод
        self.embedding_model = self._get_embedding_model(embedding_model_name)

        # Инициализация стеммера, если запрошено
        if self.use_stemmer:
            try:
                from nltk.stem.snowball import SnowballStemmer
                self.stemmer = SnowballStemmer("russian")
            except ImportError:
                logger.warning("nltk не установлен, стемминг отключён. Установите: pip install nltk")
                self.use_stemmer = False

        # Создание клиента ChromaDB
        if persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.EphemeralClient()

        # Создание коллекции
        collection_name = f"tender_{os.path.basename(docs_folder)}"

        try:
            self.client.delete_collection(collection_name)
        except NotFoundError:
            pass

        self.collection = self.client.create_collection(
            name=collection_name,
            embedding_function=None
        )

        # Загружаем документы и наполняем коллекцию
        self.all_chunk_texts = []   # список текстов чанков
        self.all_chunk_ids = []     # список идентификаторов чанков (совпадают с id в Chroma)
        self._load_documents()

        # Строим BM25 индекс, если доступно
        self.bm25 = None
        if BM25_AVAILABLE:
            self._build_bm25_index()
        else:
            logger.warning("BM25 не доступен (установите rank-bm25). Используется только векторный поиск.")

    def _expand_query(self, question: str) -> str:
        """Расширяет/переформулирует вопрос с помощью LLM для улучшения поиска."""
        prompt = f"""Ты помогаешь улучшить поиск по тендерной документации. Переформулируй данный вопрос так, чтобы он лучше подходил для поиска релевантных фрагментов документов. Используй ключевые термины и синонимы, сохранив исходный смысл. Ответь только переформулированным вопросом, без пояснений.

Исходный вопрос: {question}

Переформулированный вопрос:"""
        response = ask_llama_sync(prompt, self.llm_model)
        if response and len(response) > 10:
            return response.strip()
        return question

    def _load_company_context(self) -> str:
        """Загружает дополнительный контекст о компании из отдельной папки."""
        if not self.company_context_dir:
            return ""
        if not os.path.isdir(self.company_context_dir):
            logger.warning(f"Папка контекста компании не найдена: {self.company_context_dir}")
            return ""

        parts: List[str] = []
        for root, _, files in os.walk(self.company_context_dir):
            for file in files:
                if not (file.lower().endswith(".md") or file.lower().endswith(".txt")):
                    continue
                path = os.path.join(root, file)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        parts.append(f.read())
                except Exception as e:
                    logger.warning(f"Не удалось прочитать файл контекста {path}: {e}")
                    continue

        return "\n\n---\n\n".join(parts)

    def _split_text(self, text: str) -> List[str]:
        """Разбивает текст на чанки с сохранением логики заголовков и абзацев."""
        if LANGCHAIN_AVAILABLE:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n## ", "\n### ", "\n#### ", "\n", ". ", "! ", "? ", " ", ""],
                keep_separator=True,
            )
            return splitter.split_text(text)
        else:
            # Простой сплиттер по абзацам и длине
            paragraphs = text.split("\n\n")
            chunks: List[str] = []
            current = ""
            for para in paragraphs:
                p = para.strip()
                if not p:
                    continue
                if p.lstrip().startswith("#"):
                    if current:
                        chunks.append(current)
                    current = p
                elif len(current) + len(p) < self.chunk_size:
                    current += ("\n\n" + p if current else p)
                else:
                    if current:
                        chunks.append(current)
                    current = p
            if current:
                chunks.append(current)
            return chunks

    def _load_documents(self):
        """Читает все .md файлы в папке, разбивает на чанки и добавляет в коллекцию."""
        if not os.path.isdir(self.docs_folder):
            raise NotADirectoryError(f"Папка не найдена: {self.docs_folder}")

        documents = []
        metadatas = []
        ids = []

        logger.info(f"Загрузка документов из {self.docs_folder}...")

        # Собираем все .md файлы
        md_files = []
        for root, _, files in os.walk(self.docs_folder):
            for file in files:
                if file.lower().endswith('.md'):
                    md_files.append(os.path.join(root, file))

        if not md_files:
            raise ValueError(f"В папке {self.docs_folder} не найдено .md файлов")

        # Если запрошен прогресс, используем tqdm
        if self.show_progress and TQDM_AVAILABLE:
            file_iterator = tqdm(md_files, desc="Чтение файлов", unit="файл")
        else:
            file_iterator = md_files
            if self.show_progress and not TQDM_AVAILABLE:
                logger.warning("tqdm не установлен, прогресс-бар не будет отображаться. Установите: pip install tqdm")

        for file_path in file_iterator:
            rel_path = os.path.relpath(file_path, self.docs_folder)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            except Exception as e:
                logger.warning(f"Не удалось прочитать {file_path}: {e}")
                continue

            chunks = self._split_text(text)
            for i, chunk in enumerate(chunks):
                doc_id = f"{rel_path}__chunk{i}"
                # Добавляем имя файла в начало чанка
                chunk_with_source = f"### Файл: {rel_path}\n\n{chunk}"
                documents.append(chunk_with_source)
                metadatas.append({"source": rel_path, "chunk": i})
                ids.append(doc_id)

        # Проверка лимита чанков
        if len(documents) > self.max_chunks:
            logger.warning(
                f"Количество чанков ({len(documents)}) превышает лимит {self.max_chunks}. "
                f"Будут использованы только первые {self.max_chunks} чанков."
            )
            documents = documents[:self.max_chunks]
            metadatas = metadatas[:self.max_chunks]
            ids = ids[:self.max_chunks]

        self.all_chunk_texts = documents
        self.all_chunk_ids = ids
        self.id_to_index = {id_: idx for idx, id_ in enumerate(ids)}

        logger.info(f"Создание эмбеддингов для {len(documents)} чанков...")
        # Получаем эмбеддинги
        texts_to_encode = documents
        if "e5" in (self.embedding_model_name or "").lower():
            texts_to_encode = [f"passage: {d}" for d in documents]

        # Используем прогресс-бар при создании эмбеддингов
        encode_kwargs = {
            "batch_size": 64,
            "normalize_embeddings": True,
            "show_progress_bar": self.show_progress and TQDM_AVAILABLE
        }

        embeddings = self.embedding_model.encode(texts_to_encode, **encode_kwargs).tolist()

        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        logger.info(f"Загружено {len(documents)} чанков из {self.docs_folder}")

    def _tokenize(self, text: str) -> List[str]:
        """Разбивает текст на токены, применяя стемминг если включён."""
        if self.use_stemmer:
            return [self.stemmer.stem(w) for w in text.split()]
        return text.split()

    def _build_bm25_index(self):
        """Строит BM25 индекс с учётом стемминга."""
        if not self.all_chunk_texts:
            logger.warning("Нет текстов для построения BM25 индекса")
            return
        tokenized_corpus = [self._tokenize(doc) for doc in self.all_chunk_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info("BM25 индекс построен (стемминг %s)", "включён" if self.use_stemmer else "отключён")

    def _hybrid_search(self, query: str, top_k: int) -> List[str]:
        """
        Гибридный поиск: объединяет результаты векторного поиска и BM25 с помощью RRF.
        После получения RRF-оценок применяется бустинг на основе приоритетных ключевых слов.
        Возвращает список текстов чанков (не более top_k).
        """
        candidate_multiplier = 2
        vec_n_results = top_k * candidate_multiplier

        # --- Векторный поиск ---
        query_text = query
        if "e5" in (self.embedding_model_name or "").lower():
            query_text = f"query: {query}"
        query_emb = self.embedding_model.encode([query_text], normalize_embeddings=True)[0]

        vec_results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=vec_n_results
        )
        vec_ids = vec_results['ids'][0] if vec_results['ids'] else []

        # --- BM25 поиск (если доступен) ---
        bm25_indices = []
        if self.bm25 is not None:
            tokenized_query = self._tokenize(query)
            bm25_scores = self.bm25.get_scores(tokenized_query)
            bm25_indices = np.argsort(bm25_scores)[-vec_n_results:][::-1].tolist()

        if self.bm25 is None:
            if not vec_ids:
                return []
            # Только векторные результаты (обрезаем до top_k)
            return [self.all_chunk_texts[self.id_to_index[id_]] for id_ in vec_ids[:top_k]]

        # --- RRF: ранговое слияние ---
        rrf_scores = {}
        k = 60  # параметр RRF

        # Векторные результаты
        for rank, id_ in enumerate(vec_ids):
            if id_ not in self.id_to_index:
                continue
            idx = self.id_to_index[id_]
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (k + rank + 1)

        # BM25 результаты
        for rank, idx in enumerate(bm25_indices):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (k + rank + 1)

        # --- Бустинг на основе приоритетных ключевых слов ---
        for idx in rrf_scores.keys():
            text_lower = self.all_chunk_texts[idx].lower()
            boost = 1.0

            # Высокий приоритет: +0.3 за любое совпадение (максимум один раз)
            for kw in self.high_priority:
                if kw.lower() in text_lower:
                    boost += 0.3
                    break

            # Средний приоритет: +0.15
            for kw in self.medium_priority:
                if kw.lower() in text_lower:
                    boost += 0.15
                    break

            # Низкий приоритет: +0.05
            for kw in self.low_priority:
                if kw.lower() in text_lower:
                    boost += 0.05
                    break

            # Применяем буст (аддитивно)
            rrf_scores[idx] *= boost

        # Сортируем по убыванию скорректированной RRF-оценки
        sorted_indices = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in sorted_indices[:top_k]]

        # Получаем тексты чанков
        result_texts = [self.all_chunk_texts[idx] for idx in top_indices]
        return result_texts

    def answer_question(self, question: str, retry: bool = False) -> str:
        # Возможно, расширяем запрос
        search_query = question
        if self.query_expansion and not retry:  # не расширяем при повторной попытке
            expanded = self._expand_query(question)
            if expanded != question:
                logger.debug(f"Расширенный запрос: {expanded}")
                search_query = expanded

        best_docs = self._hybrid_search(search_query, self.top_k)

        if not best_docs:
            return "Не удалось найти информацию в документации."

        context = "\n\n---\n\n".join(best_docs)

        prompt = f"""Ты — эксперт по тендерной документации. Используй приведённый ниже контекст, чтобы ответить на вопрос. Если в контексте нет информации, необходимой для ответа, напиши: "Информация отсутствует". 

Контекст:
{context}

Вопрос: {question}

Инструкция:
1. Найди в контексте фрагменты, относящиеся к вопросу.
2. Сформулируй ответ на основе этих фрагментов. Если возможно, процитируй ключевые части.
3. Объясни, почему ты делаешь такой вывод, ссылаясь на контекст.

Ответ:"""

        response = ask_llama_sync(prompt, self.llm_model)
        if response is None:
            return "Ошибка получения ответа от LLM."

        # Если ответ говорит об отсутствии информации, пробуем ещё раз с увеличенным top_k
        if "информация отсутствует" in response.lower() and not retry:
            logger.info("Ответ не найден, пробуем с увеличенным top_k")
            old_top_k = self.top_k
            self.top_k = old_top_k * 2
            response = self.answer_question(question, retry=True)
            self.top_k = old_top_k
            return response

        return response.strip()

    def evaluate_suitability(self, company_profile: str, tender_category: Optional[str] = None) -> Dict[str, Any]:
        """
        Оценивает пригодность тендера для компании на основе документации и правил компании.
        Возвращает словарь с полями:
          - suitability_score: число от 0 до 1
          - decision: строка ('Подходит', 'Не подходит', 'На грани')
          - reasoning: краткое текстовое пояснение
        """
        # Формируем улучшенный поисковый запрос, чтобы захватить чанки с технологиями и направлениями
        base_query = (
            "Найди в тендерной документации информацию о предмете закупки, типе работ, "
            "наличии приборов учета, интерфейсах передачи данных (LoRaWAN, GSM, RF, NB-IoT, PLC, RS-485 и др.), "
            "направлениях (вода, тепло, газ, электроэнергия, высоковольтные ПУ), а также об условиях исполнения. "
            f"Профиль компании: {company_profile}"
        )
        if tender_category:
            suitability_query = base_query + f" Особое внимание удели категории '{tender_category}'."
        else:
            suitability_query = base_query

        # Гибридный поиск с увеличенным top_k для лучшего покрытия
        best_docs = self._hybrid_search(suitability_query, max(self.top_k, 7))
        if not best_docs:
            return {
                "suitability_score": 0.0,
                "decision": "Не подходит",
                "reasoning": "В документации не удалось найти информацию для оценки пригодности."
            }

        context = "\n\n---\n\n".join(best_docs)

        # Загружаем дополнительный контекст компании (там должен лежать файл с правилами, например "Тендеры.md")
        company_extra_context = self._load_company_context()
        if not company_extra_context.strip():
            logger.warning("Контекст компании пуст. Убедитесь, что в папке company_context_dir есть файлы с правилами.")

        prompt = f"""Ты — эксперт по тендерной документации, который оценивает тендеры на соответствие критериям компании.

Ниже представлен **документ с критериями компании**, описывающий, какие тендеры интересны, а какие нет, а также приоритеты технологий. Используй ЭТОТ ДОКУМЕНТ как единственный источник правил для оценки.

ДОКУМЕНТ КОМПАНИИ:
{company_extra_context}

Категория тендера (предварительная, может уточняться): {tender_category}

КОНТЕКСТ ДОКУМЕНТАЦИИ ТЕНДЕРА (извлечён из файлов тендера):
{context}

На основе документа компании и контекста тендера выполни следующие шаги:

1. **Определи направление тендера** согласно классификации из документа компании (вода, тепло, газ, высоковольтные ПУ, работы, поставка ПУ и т.д.). Если направление не очевидно, укажи "не определено".

2. **Определи, какие технологии передачи данных упоминаются** в тендере (LoRaWAN, GSM, RF, NB-IoT, PLC, RS-485, Zigbee, Wi-Fi и др.). Сравни их с приоритетами из документа компании (высокий, средний, дополнительный мониторинг). Отметь, есть ли прямое упоминание LoRaWAN или комбинированных решений с LoRaWAN.

3. **Проверь особые правила для конкретных направлений:**
   - Если направление "Вода": проверь, есть ли поставка умных счетчиков (с интерфейсами передачи данных). Если да — тендер интересен.
   - Если "Тепло" или "Газ": определи, есть ли поставка приборов учета. Отметь наличие интерфейсов. Такие тендеры требуют аналитической сводки.
   - Если "Высоковольтные ПУ": оцени заказчика и объект, если информация доступна. Сравни с перечнем ключевых заказчиков (если он есть в документе компании).
   - Если "Работы": проверь, связаны ли работы с системами учета (АСКУЭ, АСУЭ, АИИС КУЭ, СМР, ПНР), есть ли интеллектуальные ПУ, каналы передачи данных.
   - Если "Поставка ПУ" (или "ПУ и работы"): убедись, что в спецификации есть интерфейсы передачи данных.

4. **Следуй алгоритму из документа компании (раздел 5):**
   - Шаг 1: проверь направление.
   - Шаг 2: проверь наличие технологий.
   - Шаг 3: определи тип тендера.
   - Шаг 4: проверь наличие умного учета.
   - Шаг 5: прими решение согласно правилам.

5. **Сформулируй итоговое решение** на основе критериев из документа:
   - "Подходит" — если тендер однозначно соответствует категориям "Однозначно интересны" (поставка ПУ с интерфейсами, умные счетчики воды, работы по АСКУЭ/АСУЭ, наличие LoRaWAN и т.д.).
   - "На грани" — если тендер относится к направлениям, требующим аналитической сводки (тепло, газ, высоковольтные ПУ без ключевого заказчика), или если есть смешанные сигналы.
   - "Не подходит" — если тендер не содержит ПУ, нет передачи данных, обычные счетчики без интеллектуальных функций, работы не связаны с системами учета.

6. **Выставь числовую оценку suitability_score** от 0 до 1:
   - 0.8–1.0 для "Подходит"
   - 0.4–0.7 для "На грани"
   - 0.0–0.3 для "Не подходит"

**Требования к ответу:**
- Ответ ДОЛЖЕН быть ТОЛЬКО в формате JSON без какого-либо лишнего текста до или после.
- В поле `reasoning` кратко поясни на русском языке, какие критерии из документа компании сработали (например, "Есть LoRaWAN, вода с умными счетчиками — соответствует высокому приоритету").

Формат JSON:
{{
  "suitability_score": число от 0 до 1,
  "decision": "Подходит" | "Не подходит" | "На грани",
  "reasoning": "пояснение"
}}

Ответ:"""

        raw_response = ask_llama_sync(prompt, self.llm_model)
        if raw_response is None:
            return {
                "suitability_score": 0.0,
                "decision": "Не подходит",
                "reasoning": "Ошибка LLM при попытке оценить пригодность тендера."
            }

        parsed = repair_json_fragment(raw_response)
        if not isinstance(parsed, dict):
            return {
                "suitability_score": 0.0,
                "decision": "Не подходит",
                "reasoning": "Не удалось корректно распарсить ответ LLM."
            }

        score = float(parsed.get("suitability_score", 0.0))
        score = max(0.0, min(1.0, score))

        decision = parsed.get("decision", "На грани")
        reasoning = parsed.get("reasoning", "").strip()

        # Дополнительная страховка: если score высокий, а decision "Не подходит" — корректируем
        if score >= 0.7 and decision == "Не подходит":
            decision = "На грани"
        if score <= 0.3 and decision == "Подходит":
            decision = "На грани"

        return {
            "suitability_score": score,
            "decision": decision,
            "reasoning": reasoning
        }

    def analyze_all_questions(self, questions: List[str] = None, tender_category: Optional[str] = None) -> Dict[str, str]:
        """
        Задаёт все вопросы и возвращает словарь {вопрос: ответ}.
        Если передана tender_category, формирует расширенный набор вопросов под конкретную категорию.
        """
        if questions is None:
            questions = build_category_questions(tender_category)
        answers: Dict[str, str] = {}
        for q in questions:
            logger.debug(f"Вопрос: {q}")
            ans = self.answer_question(q)
            answers[q] = ans
        return answers


def build_category_questions(tender_category: Optional[str]) -> List[str]:
    """
    Формирует список вопросов для deep RAG в зависимости от категории тендера.
    """
    base_questions = DEFAULT_QUESTIONS.copy()
    if not tender_category or tender_category == "Не профиль":
        return base_questions

    extra = CATEGORY_QUESTIONS.get(tender_category, [])
    return base_questions + extra


def analyze_tender_docs(
        docs_folder: str,
        questions: List[str] = None,
        embedding_model: str = "intfloat/multilingual-e5-large",
        llm_model: str = ANALYSIS_MODEL,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        top_k: int = 5,
        tender_category: Optional[str] = None,
        show_progress: bool = False,
        max_chunks: int = 500
) -> Dict[str, str]:
    """
    Упрощённая функция для анализа одного тендера.
    Если переданы questions — возвращает ответы на вопросы.
    Если questions is None — возвращает только оценку пригодности.
    """
    analyzer = TenderRAGAnalyzer(
        docs_folder=docs_folder,
        embedding_model_name=embedding_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
        llm_model=llm_model,
        show_progress=show_progress,
        max_chunks=max_chunks
    )
    if questions:
        return analyzer.analyze_all_questions(questions, tender_category=tender_category)
    else:
        default_profile = (
            "Компания занимается поставкой, проектированием, монтажом и обслуживанием систем учёта "
            "электроэнергии и воды, АСКУЭ, поверкой счётчиков, а также сопутствующим электротехническим оборудованием."
        )
        return analyzer.evaluate_suitability(default_profile, tender_category=tender_category)


# Для тестирования
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    folder = "extracted_texts/90349915"  # замените на реальный путь
    if os.path.isdir(folder):
        # Включим прогресс при тестировании
        answers = analyze_tender_docs(folder, show_progress=True)
        for q, a in answers.items():
            print(f"Q: {q}\nA: {a}\n")
    else:
        print(f"Папка {folder} не найдена")