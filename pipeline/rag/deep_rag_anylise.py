"""
RAG-анализ документации тендеров.
Для папки с .md файлами строит векторную БД, задаёт вопросы и получает ответы через Ollama.
"""
import os
import json
import torch
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



def _get_category_focus_block(tender_category: Optional[str]) -> str:
    """
    Возвращает категорийный блок инструкций для LLM, чтобы смещать фокус анализа
    в зависимости от predicted_category/tender_category.
    """
    if not tender_category:
        return (
            "КАТЕГОРИЙНЫЙ ФОКУС: категория не определена.\n"
            "- Приоритет: предмет закупки, наличие поставки ПУ, условия выполнения, сроки, требования к участнику.\n"
            "- Если явно есть технологии передачи данных, учитывай их как вторичный признак."
        )

    category_normalized = tender_category.strip().lower()

    if category_normalized == "технологии":
        return (
            "КАТЕГОРИЙНЫЙ ФОКУС: Технологии.\n"
            "- Приоритет №1: протоколы/каналы/интерфейсы передачи данных (LoRaWAN, NB-IoT, GSM, PLC, RF, RS-485, Modbus и т.д.).\n"
            "- Зафиксируй архитектурные признаки: шлюзы, базовые станции, концентраторы, телеметрия, АСКУЭ/АСУЭ/АИИС.\n"
            "- Если поставка ПУ не явная, но технологический контур и удалённый сбор данных есть — это релевантный сигнал.\n"
            "- Отдельно отметь, какие технологии прямо названы, а какие отсутствуют."
        )

    if category_normalized == "работы":
        return (
            "КАТЕГОРИЙНЫЙ ФОКУС: Работы.\n"
            "- Приоритет №1: тип работ (монтаж/демонтаж/пусконаладка/ремонт/обслуживание), объём и состав работ.\n"
            "- Проверяй необходимость СРО/допусков/лицензий, а также требования к персоналу и технике.\n"
            "- Отдельно оцени, это только сервис/проектирование или есть поставка и внедрение оборудования.\n"
            "- В reasoning делай акцент на контрактных условиях исполнения: сроки, гарантия, ответственность."
        )

    if (
        category_normalized in ("вода. простые пу", "ээ.простые пу", "пу и работы")
        or "счетчик" in category_normalized
        or "счётчик" in category_normalized
        or "пу" in category_normalized
    ):
        return (
            "КАТЕГОРИЙНЫЙ ФОКУС: Счётчики/ПУ.\n"
            "- Приоритет №1: конкретные требования к приборам учёта (тип, класс точности, Ду/DN, межповерочный интервал, комплектация).\n"
            "- Явно фиксируй, есть ли поставка/замена ПУ или только поверка/обслуживание существующих.\n"
            "- Проверяй наличие технических характеристик счётчиков, требований к сертификации и метрологии.\n"
            "- Технологии связи учитывай как дополнительный фактор после базовых требований к ПУ."
        )

    return (
        f"КАТЕГОРИЙНЫЙ ФОКУС: {tender_category}.\n"
        "- Приоритет: предмет закупки, поставка ПУ, условия исполнения и релевантные технические требования.\n"
        "- Если есть прямые указания на технологии связи/АСКУЭ — учитывай их как усиливающий фактор.\n"
        "- Избегай домыслов: опирайся только на факты документации."
    )


class TenderRAGAnalyzer:
    """
    Класс для RAG-анализа одного тендера.
    """

    # Кэширование модели на уровне класса
    @staticmethod
    @lru_cache(maxsize=2)
    def _get_embedding_model(model_name: str, device: str):
        logger.info(f"Загрузка модели эмбеддингов {model_name} на устройство {device}...")
        model = SentenceTransformer(model_name, device=device)
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
            max_chunks: int = 500,         # ограничение на количество чанков
            device: str = "auto"            # устройство для эмбеддингов
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

        if device == "auto":
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
                logger.info("Используется MPS (Metal Performance Shaders)")
            elif torch.cuda.is_available():
                self.device = "cuda"
                logger.info("Используется CUDA")
            else:
                self.device = "cpu"
                logger.info("Используется CPU")
        else:
            self.device = device
            logger.info(f"Используется явно заданное устройство: {self.device}")

        # Инициализация модели через кэширующий метод
        self.embedding_model = self._get_embedding_model(embedding_model_name, self.device)

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

    def _generate_multi_queries(self, question: str) -> List[str]:
        """
        Генерирует несколько перефразировок вопроса для расширенного поиска (multi-query RAG).
        Возвращает список из оригинального вопроса + 2-3 вариантов.
        """
        prompt = f"""Ты помогаешь улучшить поиск по тендерной документации.
Сгенерируй 3 различные перефразировки следующего вопроса для поиска релевантных фрагментов.
Используй синонимы, ключевые слова, краткие формулировки, профессиональные термины.
Каждый вариант — на отдельной строке. Только перефразировки, без нумерации и пояснений.

Вопрос: {question}

Перефразировки:"""
        response = ask_llama_sync(prompt, self.llm_model)
        queries = [question]  # Оригинальный вопрос всегда первым
        if response:
            for line in response.strip().splitlines():
                line = line.strip().lstrip("-•1234567890.). ")
                if line and len(line) > 5 and line not in queries:
                    queries.append(line)
                if len(queries) >= 4:  # Оригинал + 3 перефразировки максимум
                    break
        return queries

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

    def _hybrid_search(self, query: str, top_k: int, extra_queries: Optional[List[str]] = None) -> List[str]:
        """
        Гибридный поиск: объединяет результаты векторного поиска и BM25 с помощью RRF.
        Поддерживает multi-query: если передан extra_queries, результаты по всем запросам
        объединяются перед итоговым RRF-ранжированием.
        После получения RRF-оценок применяется бустинг на основе приоритетных ключевых слов.
        Возвращает список текстов чанков (не более top_k).
        """
        candidate_multiplier = 3  # увеличено с 2 для лучшего охвата
        vec_n_results = min(top_k * candidate_multiplier, len(self.all_chunk_texts))

        all_queries = [query]
        if extra_queries:
            all_queries += [q for q in extra_queries if q and q != query]

        rrf_scores: Dict[int, float] = {}
        k = 60  # параметр RRF

        for q_idx, q in enumerate(all_queries):
            # --- Векторный поиск ---
            query_text = q
            if "e5" in (self.embedding_model_name or "").lower():
                query_text = f"query: {q}"
            query_emb = self.embedding_model.encode([query_text], normalize_embeddings=True)[0]

            vec_results = self.collection.query(
                query_embeddings=[query_emb],
                n_results=vec_n_results
            )
            vec_ids = vec_results['ids'][0] if vec_results['ids'] else []

            # Векторные результаты → RRF
            for rank, id_ in enumerate(vec_ids):
                if id_ not in self.id_to_index:
                    continue
                idx = self.id_to_index[id_]
                rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (k + rank + 1)

            # --- BM25 поиск (если доступен) ---
            if self.bm25 is not None:
                tokenized_query = self._tokenize(q)
                bm25_scores = self.bm25.get_scores(tokenized_query)
                bm25_indices = np.argsort(bm25_scores)[-vec_n_results:][::-1].tolist()
                for rank, idx in enumerate(bm25_indices):
                    rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (k + rank + 1)

        if not rrf_scores:
            return []

        if self.bm25 is None and len(all_queries) == 1:
            # Только векторные результаты без BM25 и multi-query
            vec_ids_fallback = []
            query_text = query
            if "e5" in (self.embedding_model_name or "").lower():
                query_text = f"query: {query}"
            query_emb = self.embedding_model.encode([query_text], normalize_embeddings=True)[0]
            vec_results = self.collection.query(query_embeddings=[query_emb], n_results=top_k)
            vec_ids_fallback = vec_results['ids'][0] if vec_results['ids'] else []
            return [self.all_chunk_texts[self.id_to_index[id_]] for id_ in vec_ids_fallback if id_ in self.id_to_index]

        # --- Бустинг на основе приоритетных ключевых слов ---
        # Также учитываем слова из самого вопроса (question-aware boost)
        question_words = set(query.lower().split())
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

            # Question-aware boost: если чанк содержит ≥2 слова из вопроса — лёгкий буст
            question_hits = sum(1 for w in question_words if len(w) > 3 and w in text_lower)
            if question_hits >= 3:
                boost += 0.1
            elif question_hits >= 2:
                boost += 0.05

            rrf_scores[idx] *= boost

        # Сортируем по убыванию скорректированной RRF-оценки
        sorted_indices = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in sorted_indices[:top_k]]

        # Получаем тексты чанков
        result_texts = [self.all_chunk_texts[idx] for idx in top_indices]
        return result_texts

    def answer_question(self, question: str, retry: bool = False, tender_category: Optional[str] = None) -> str:
        # --- Multi-query: генерируем перефразировки для лучшего охвата ---
        extra_queries: List[str] = []
        if not retry:
            if self.query_expansion:
                multi = self._generate_multi_queries(question)
                extra_queries = multi[1:] if len(multi) > 1 else []
                logger.debug(f"Multi-query LLM ({len(extra_queries)+1} запросов): {multi}")
            # Дополняем статическими search hints по категории (бесплатно, без LLM)
            if tender_category:
                hints = get_search_hints_for_question(question, tender_category)
                for h in hints:
                    if h not in extra_queries:
                        extra_queries.append(h)
                logger.debug(f"Добавлены статические hints для '{tender_category}': {hints}")

        best_docs = self._hybrid_search(question, self.top_k, extra_queries=extra_queries if not retry else None)

        if not best_docs:
            return "Не удалось найти информацию в документации."

        context = "\n\n---\n\n".join(best_docs)

        prompt = f"""Ты — аналитик тендерной документации. Твоя задача — найти ответ на вопрос в приведённом контексте.

Контекст:
{context}

Вопрос: {question}

Инструкция:
1. Если ответ ЯВНО есть в контексте — ответь одним-двумя предложениями, процитировав или пересказав ключевую фразу.
2. Если ответ КОСВЕННО следует из контекста (упомянуты связанные требования, оборудование, условия) — ответь с пометкой «(косвенно)» и поясни, что именно это подразумевает.
3. Только если информации нет совсем — ответь: "Не упомянуто в документации."
4. НЕ додумывай факты, которых нет даже косвенно.

Ответ:"""

        response = ask_llama_sync(prompt, self.llm_model)
        if response is None:
            return "Ошибка получения ответа от LLM."

        # Если ответ говорит об отсутствии информации, пробуем ещё раз с увеличенным top_k
        if ("не упомянуто" in response.lower() or "информация отсутствует" in response.lower()) and not retry:
            logger.info("Ответ не найден, пробуем с увеличенным top_k")
            old_top_k = self.top_k
            self.top_k = min(old_top_k * 3, len(self.all_chunk_texts))  # утроенный top_k
            response = self.answer_question(question, retry=True, tender_category=tender_category)
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
        category_focus_block = _get_category_focus_block(tender_category)

        if tender_category:
            suitability_query = (
                base_query
                + f" Особое внимание удели категории '{tender_category}'. "
                + category_focus_block.replace("\n", " ")
            )
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

        prompt = f"""Ты — строгий классификатор тендеров. Твоя задача — извлечь факты из документации и принять решение по жёстким правилам. Никаких предположений, домыслов, попыток «улучшить» тендер.

ПРАВИЛА КОМПАНИИ:
{company_extra_context}

КАТЕГОРИЯ ТЕНДЕРА (предварительная): {tender_category}

{category_focus_block}

ФРАГМЕНТЫ ДОКУМЕНТАЦИИ ТЕНДЕРА:
{context}

═══════════════════════════════════════
ИНСТРУКЦИЯ: выполни шаги строго по порядку.

ШАГ 1. ИЗВЛЕЧЕНИЕ ФАКТОВ (только то, что явно написано в документации)
Заполни поля:
- direction: направление тендера — одно из: "Вода", "Тепло", "Газ", "Электроэнергия", "Высоковольтные ПУ", "Работы", "Освещение", "не определено"
- has_pu_supply: есть ли явное требование к ПОСТАВКЕ приборов учёта — true/false
- found_technologies: список технологий передачи данных, ДОСЛОВНО упомянутых в тексте. Если ни одной — пустой список []. Не додумывай — только то, что написано.
- has_smart_metering: есть ли явные слова "умный счетчик", "интеллектуальный ПУ", "АСКУЭ", "АСУЭ", "АИИС КУЭ", "автоматизированный сбор", "удаленный съем" — true/false
- only_service: тендер ТОЛЬКО на обслуживание/поверку/ремонт существующих ПУ без поставки новых — true/false
- only_design: тендер ТОЛЬКО на проектирование/изыскания без поставки и монтажа ПУ — true/false
- customer: заказчик (строка или null)
- key_facts: список из 2–5 дословных коротких цитат или точных фактов из документации, которые определили решение (например: ["поставка счётчиков воды Ду15", "технология LoRaWAN не упомянута", "только поверка приборов учёта"])

ШАГ 2. ПРИНЯТИЕ РЕШЕНИЯ (по жёстким правилам, без исключений)

ПРАВИЛО C (проверяй ПЕРВЫМ) → "Не подходит" (score 0.05–0.15):
  - only_service == true  (только обслуживание/поверка/ремонт — нет поставки/замены)
  - only_design == true   (только проектирование без поставки и монтажа ПУ)
  - direction == "Освещение"
  - direction == "не определено" И has_pu_supply == false И found_technologies пустой
  - direction == "Работы" И has_smart_metering == false И found_technologies пустой И has_pu_supply == false
  - has_pu_supply == false И found_technologies пустой И has_smart_metering == false И direction не в ["Вода","Тепло","Газ","Электроэнергия","Высоковольтные ПУ"]

ПРАВИЛО A → "Подходит" (score 0.82–0.92):
  Применяется ТОЛЬКО если правило C не сработало. Хотя бы одно из:
  - found_technologies содержит "LoRaWAN"
  - direction == "Вода" И has_pu_supply == true И (found_technologies непустой ИЛИ has_smart_metering == true)
  - has_smart_metering == true И has_pu_supply == true
  - direction == "Электроэнергия" И found_technologies содержит хотя бы одно из [GSM, NB-IoT, PLC, RF, RS-485] И has_pu_supply == true
  - КАТЕГОРИЯ ТЕНДЕРА == "Технологии" И direction == "Электроэнергия" И found_technologies содержит хотя бы одно из [GSM, NB-IoT, PLC, RF, RS-485, LoRaWAN] И has_smart_metering == true

ПРАВИЛО B → "На грани" (score 0.42–0.58):
  Применяется ТОЛЬКО если не сработали ни C, ни A. Хотя бы одно из:
  - direction в ["Тепло", "Газ"] И has_pu_supply == true
  - direction == "Высоковольтные ПУ" И has_pu_supply == true
  - found_technologies непустой И has_pu_supply == true И direction не попадает под A
  - КАТЕГОРИЯ ТЕНДЕРА == "Технологии" И found_technologies непустой И has_smart_metering == true (даже без явной поставки ПУ)

  ВАЖНО: "На грани" запрещено если:
  - has_pu_supply == false И found_technologies пустой И has_smart_metering == false → это всегда "Не подходит"
  - direction в ["Тепло","Газ"] И has_pu_supply == false → "Не подходит"

ШАГ 3. ФОРМИРОВАНИЕ reasoning
Подробное обоснование — 3–5 предложений. Обязательно включи:
1. Что конкретно найдено в документации (предмет закупки, объект, условия — дословно или близко к тексту).
2. Какие технологии упомянуты или явно отсутствуют.
3. Есть ли требование к поставке ПУ и умному учёту — с опорой на текст.
4. Какое правило (A / B / C) применено и почему именно оно.
5. Что именно не позволяет поставить более высокую оценку (для B и C).

Формат reasoning: связный текст, не список. Ссылайся на конкретные факты из key_facts.
НЕ ПИШИ общих фраз вроде "тендер не профильный" без обоснования.

═══════════════════════════════════════
Ответ ТОЛЬКО в формате JSON без лишнего текста:
{{
  "direction": "...",
  "has_pu_supply": true/false,
  "found_technologies": ["...", "..."],
  "has_smart_metering": true/false,
  "only_service": true/false,
  "only_design": true/false,
  "customer": "..." или null,
  "key_facts": ["...", "..."],
  "suitability_score": число,
  "decision": "Подходит" | "На грани" | "Не подходит",
  "reasoning": "..."
}}

JSON:"""

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
                "reasoning": "Не удалось корректно распарсить ответ LLM.",
                "direction": "не определено",
                "found_technologies": [],
                "has_pu_supply": False,
                "has_smart_metering": False,
                "customer": None,
            }

        score = float(parsed.get("suitability_score", 0.0))
        score = max(0.0, min(1.0, score))

        decision = parsed.get("decision", "На грани")
        reasoning = parsed.get("reasoning", "").strip()
        direction = parsed.get("direction", "не определено")
        found_technologies = parsed.get("found_technologies", [])
        has_pu_supply = bool(parsed.get("has_pu_supply", False))
        has_smart_metering = bool(parsed.get("has_smart_metering", False))
        only_service = bool(parsed.get("only_service", False))
        only_design = bool(parsed.get("only_design", False))
        key_facts = parsed.get("key_facts", [])

        # ── ДЕТЕРМИНИРОВАННЫЕ ПРАВИЛА ПОСТОБРАБОТКИ ─────────────────────────
        # Применяются поверх ответа LLM — они не могут быть «домыслены» моделью.

        # Флаг: классификатор определил категорию "Технологии" (GSM, LoRaWAN и т.п.)
        is_tech_category = tender_category == "Технологии"

        # Правило C1: чисто сервисные / поверочные тендеры — всегда "Не подходит"
        # Исключение: если классификатор дал категорию "Технологии" И есть технологии/АСКУЭ —
        # тендер на обслуживание системы передачи данных релевантен, снижаем только до "На грани"
        if only_service or only_design:
            if is_tech_category and (found_technologies or has_smart_metering):
                # Смягчённое правило для категории "Технологии": сервисный тендер с технологиями — "На грани"
                if decision == "Не подходит":
                    decision = "На грани"
                    score = max(score, 0.45)
                score = min(score, 0.62)
                tag = "обслуживание/поверку" if only_service else "проектирование"
                reasoning += (
                    f" [Правило C1-Тех: тендер только на {tag}, но классификатор дал категорию 'Технологии' "
                    f"и найдены технологии {found_technologies or ['АСКУЭ']} — решение смягчено до 'На грани'.]"
                )
            else:
                decision = "Не подходит"
                score = min(score, 0.12)
                if only_service:
                    reasoning += " [Правило C1: тендер только на обслуживание/поверку без поставки новых ПУ — Не подходит.]"
                else:
                    reasoning += " [Правило C1: тендер только на проектирование без поставки/монтажа ПУ — Не подходит.]"

        # Правило C2: нет ни поставки ПУ, ни технологий, ни умного учёта
        elif not has_pu_supply and not found_technologies and not has_smart_metering:
            decision = "Не подходит"
            score = min(score, 0.12)
            reasoning += " [Правило C2: в документации нет ни поставки ПУ, ни технологий передачи данных, ни АСКУЭ — Не подходит.]"

        # Правило A-Тех: категория "Технологии" + Электроэнергия + технологии + АСКУЭ → "Подходит"
        # (применяется независимо от has_pu_supply, т.к. сервис/поверка систем передачи данных — наш профиль)
        elif (
            is_tech_category
            and direction == "Электроэнергия"
            and found_technologies
            and has_smart_metering
        ):
            decision = "Подходит"
            score = max(score, 0.82)
            reasoning += (
                f" [Правило A-Тех: классификатор дал категорию 'Технологии', направление Электроэнергия, "
                f"технологии {found_technologies} и АСКУЭ подтверждены — решение 'Подходит'.]"
            )

        # Правило B-Тех: категория "Технологии" + есть технологии + АСКУЭ (без поставки ПУ) → "На грани"
        elif is_tech_category and found_technologies and has_smart_metering:
            if decision == "Не подходит":
                decision = "На грани"
                score = max(score, 0.48)
            score = min(score, 0.65)
            reasoning += (
                f" [Правило B-Тех: классификатор дал категорию 'Технологии', найдены технологии {found_technologies} "
                f"и АСКУЭ — решение повышено до 'На грани'.]"
            )

        # Правило B-Тех-слабый: категория "Технологии" + есть технологии (без АСКУЭ) → минимум "На грани"
        elif is_tech_category and found_technologies and decision == "Не подходит":
            decision = "На грани"
            score = max(score, 0.42)
            score = min(score, 0.55)
            reasoning += (
                f" [Правило B-Тех-слабый: классификатор дал категорию 'Технологии' и найдены технологии "
                f"{found_technologies} — решение повышено до 'На грани'.]"
            )

        # Правило C3: Тепло/Газ без поставки ПУ — "Не подходит" (не "На грани")
        elif direction in ("Тепло", "Газ") and not has_pu_supply:
            decision = "Не подходит"
            score = min(score, 0.15)
            reasoning += f" [Правило C3: направление {direction} без явного требования к поставке ПУ — Не подходит.]"

        # Правило C4: "На грани" без поставки ПУ и без технологий → "Не подходит"
        elif decision == "На грани" and not has_pu_supply and not found_technologies:
            decision = "Не подходит"
            score = min(score, 0.18)
            reasoning += " [Правило C4: решение 'На грани' без поставки ПУ и без технологий не обосновано — понижено до Не подходит.]"

        # Правило B-страховка: Тепло/Газ с поставкой ПУ — максимум "На грани"
        elif direction in ("Тепло", "Газ") and decision == "Подходит":
            decision = "На грани"
            score = min(score, 0.62)
            reasoning += f" [Правило B-страховка: направление {direction} — не выше 'На грани' по политике компании.]"

        # Страховка: синхронизация score и decision
        if score >= 0.7 and decision == "Не подходит":
            decision = "На грани"
            reasoning += " [Корректировка: score высокий, решение повышено до 'На грани'.]"
        if score <= 0.25 and decision == "Подходит":
            decision = "На грани"
            reasoning += " [Корректировка: score низкий, решение понижено до 'На грани'.]"

        # Обогащаем reasoning структурированным резюме если оно слишком короткое
        if len(reasoning) < 80:
            tech_str = ", ".join(found_technologies) if found_technologies else "не упомянуты"
            facts_str = "; ".join(key_facts) if key_facts else "не извлечены"
            reasoning = (
                f"Направление: {direction}. Поставка ПУ: {'да' if has_pu_supply else 'нет'}. "
                f"Технологии: {tech_str}. Умный учёт: {'да' if has_smart_metering else 'нет'}. "
                f"Ключевые факты: {facts_str}. Решение: {decision}."
            )

        return {
            "suitability_score": score,
            "decision": decision,
            "reasoning": reasoning,
            # Структурированные факты
            "direction": direction,
            "found_technologies": found_technologies,
            "has_pu_supply": has_pu_supply,
            "has_smart_metering": has_smart_metering,
            "only_service": only_service,
            "only_design": only_design,
            "key_facts": key_facts,
            "customer": parsed.get("customer"),
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
            ans = self.answer_question(q, tender_category=tender_category)
            answers[q] = ans
        return answers


def build_category_questions(tender_category: Optional[str]) -> List[str]:
    """
    Формирует список вопросов для deep RAG в зависимости от категории тендера.
    Каждый вопрос дополняется ключевыми синонимами для повышения recall поиска.
    """
    base_questions = CATEGORY_QUESTIONS.get(tender_category, [])
    return base_questions


# Синонимичные перефразировки для каждой категории — используются при multi-query поиске
# без вызова LLM (статические, дешевле и быстрее)
CATEGORY_SEARCH_HINTS: Dict[str, List[str]] = {
    "Работы": [
        "монтаж демонтаж строительные работы установка",
        "СРО допуск лицензия строительство",
        "пусконаладка наладка испытания",
        "проектная документация проект схема",
        "поставка оборудование спецификация",
    ],
    "Освещение": [
        "счетчик электроэнергии освещение свет",
        "АСКУЭ автоматизация управление освещением",
        "светильник лампа LED энергоэффективный",
        "датчик движения фотодатчик реле",
        "трансформатор тока щит учета",
    ],
    "ПУ и работы": [
        "прибор учета поставка монтаж вода электроэнергия тепло газ",
        "поверка метрологическая аттестация",
        "гарантия обслуживание сервис",
        "проектирование система учета АСКУЭ",
        "интеграция система сбора данных",
        "сертификат соответствие тип средство измерения",
    ],
    "Вода. Простые ПУ": [
        "счетчик воды холодная горячая ХВС ГВС",
        "импульсный выход дистанционный съем",
        "поверка счетчик вода",
        "антимагнитная защита пломба",
        "диаметр условный проход Ду DN",
        "радиомодуль RF NB-IoT LoRa беспроводной",
    ],
    "ЭЭ.Простые ПУ": [
        "счетчик электроэнергии однофазный трехфазный",
        "класс точности 0.5 1 2",
        "RS-485 Modbus импульсный выход",
        "многотарифный день ночь тариф",
        "трансформатор тока ТТ подключение",
        "напряжение ток номинальный",
    ],
    "Тепло": [
        "теплосчетчик вычислитель расходомер термометр",
        "поверка тепловая энергия",
        "архив передача данных GSM модем",
        "погрешность измерение температура",
        "диаметр расходомер Ду материал латунь сталь",
        "датчик температуры термопреобразователь",
    ],
    "Газ": [
        "газовый счетчик мембранный ротационный ультразвуковой",
        "диапазон расхода Qmin Qmax",
        "поверка газ первичная периодическая",
        "импульсный выход модуль связи газ",
        "взрывозащита Ex взрывоопасная зона",
        "корректор объема давление температура компенсация",
        "фланец резьба присоединение корпус",
    ],
    "Технологии": [
        "LoRaWAN GSM NB-IoT PLC RF Zigbee Wi-Fi канал связи",
        "RS-485 Modbus протокол интерфейс",
        "концентратор базовая станция модем шлюз",
        "АСКУЭ АСУЭ АИИС автоматизированный сбор данных",
        "резервный канал связи резервное питание",
        "передача данных удаленный мониторинг телеметрия",
        "частота 433 868 МГц диапазон радиочастота",
    ],
}


def get_search_hints_for_question(question: str, tender_category: Optional[str]) -> List[str]:
    """
    Возвращает список дополнительных поисковых подсказок для данного вопроса
    на основе статических синонимов категории.
    Используется в multi-query поиске как дешёвая альтернатива LLM-расширению.
    """
    hints = CATEGORY_SEARCH_HINTS.get(tender_category, [])
    # Выбираем те подсказки, у которых есть пересечение слов с вопросом
    question_words = set(question.lower().split())
    scored: List[tuple] = []
    for hint in hints:
        hint_words = set(hint.lower().split())
        overlap = len(question_words & hint_words)
        scored.append((overlap, hint))
    scored.sort(key=lambda x: x[0], reverse=True)
    # Берём топ-2 наиболее релевантные подсказки
    return [h for _, h in scored[:2] if _ >= 0]


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