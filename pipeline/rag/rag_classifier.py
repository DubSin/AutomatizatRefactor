import json
import re
import aiohttp
import asyncio
import logging
from sentence_transformers import SentenceTransformer
import urllib.request
import urllib.error
import faiss
import pandas as pd
import numpy as np
import pickle
import os
import torch
from typing import List, Dict, Any, Optional, Tuple

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def ask_llama(prompt, model, url="http://localhost:11434/api/generate"):
    """Асинхронная версия ask_llama."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "num_predict": 100000,
                    "top_p": 0.9,
                    "top_k": 40
                }
            }, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                data = await resp.json()
                return data.get("response")
    except Exception as e:
        logger.error(f"Ошибка при запросе к Ollama: {e}")
        return None

def ask_llama_sync(prompt, model, url="http://localhost:11434/api/generate"):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 100000,
            "top_p": 0.9,
            "top_k": 40
        }
    }
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            result = json.loads(response.read().decode('utf-8'))
            return result.get("response")
    except Exception as e:
        logger.error(f"Ошибка при синхронном запросе к Ollama: {e}")
        return None

def repair_json_fragment(text):
    """Восстанавливает повреждённый JSON (из вашего кода)"""
    text = text.strip()
    if not text.startswith('{'):
        text = '{' + text
    if not text.endswith('}'):
        if text[-1] != '"' and text[-1] != '}':
            last_quote = text.rfind('"')
            if last_quote > text.rfind(':') and last_quote != len(text)-1:
                text += '"'
        text += '}'
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pattern = r'"([^"]+)"\s*:\s*("[^"]*"|\d+(?:\.\d+)?|true|false|null)'
        matches = re.findall(pattern, text)
        obj = {}
        for key, val in matches:
            if val.startswith('"') and val.endswith('"'):
                val = val[1:-1]
            else:
                try:
                    if '.' in val:
                        val = float(val)
                    else:
                        val = int(val)
                except ValueError:
                    if val == 'true':
                        val = True
                    elif val == 'false':
                        val = False
                    elif val == 'null':
                        val = None
            obj[key] = val
        return obj

class TenderClassifierRAG:
    """
    Класс для классификации тендеров с использованием RAG (поиск похожих + LLM).
    """

    def __init__(
        self,
        data_path: Any,                 # путь(-и) к Excel-файлам с обучающей базой
        embeddings_path: str,            # путь для кэша эмбеддингов
        index_path: str,                 # путь для кэша индекса FAISS
        data_cache_path: str = 'data_cache.pkl',  # путь для кэша DataFrame
        model_name: str = "intfloat/multilingual-e5-large",
        k: int = 10,                     # число ближайших соседей
        faiss_threshold: float = 0.15,    # порог для поиска (если ниже – сразу "Не профиль")
        use_inverse_freq: bool = True,
        inverse_freq_mode: str = "log",
        llm_config: Optional[Dict] = None,  # конфигурация LLM
        company_context_dir: Optional[str] = None,
        interest_threshold: float = 0.5,     # порог для преобразования interest_score в is_interesting
        device: str = "auto"
    ):
        """
        Инициализация: загрузка базы (из Excel или кэша), вычисление/загрузка эмбеддингов, построение индекса.
        """
        if device == "auto":
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                dev = "mps"
            elif torch.cuda.is_available():
                dev = "cuda"
            else:
                dev = "cpu"
        else:
            dev = device
        logger.info(f"Используется устройство для эмбеддингов: {dev}")
        # Поддерживаем как один путь (str), так и список путей
        if isinstance(data_path, (list, tuple)):
            self.data_paths = list(data_path)
        else:
            self.data_paths = [data_path]
        self.company_context_dir = company_context_dir
        self.embeddings_path = embeddings_path
        self.index_path = index_path
        self.data_cache_path = data_cache_path
        self.model_name = model_name
        self.k = k
        self.faiss_threshold = faiss_threshold
        self.use_inverse_freq = use_inverse_freq
        self.inverse_freq_mode = inverse_freq_mode
        self.llm_config = llm_config or {}
        self.interest_threshold = interest_threshold

        # Загружаем данные: либо из кэша, либо из Excel
        self.df = self._load_or_create_data()

        logger.info("Загружено %d тендеров, категорий: %d", len(self.df), self.df['category'].nunique())

        # Инициализируем модель эмбеддингов
        logger.info("Инициализация модели эмбеддингов %s", model_name)
        self.embedder = SentenceTransformer(model_name, device=dev)

        # Получаем эмбеддинги (с кэшированием)
        self.embeddings = self._get_embeddings()
        self.index = self._get_faiss_index()
        self.centroids = self._compute_centroids()   

        # Вычисляем веса категорий, если нужно
        if self.use_inverse_freq:
            self.category_weights = self._compute_category_weights()
            logger.info("Веса категорий вычислены (режим %s)", inverse_freq_mode)
        else:
            self.category_weights = {}

        # Список всех категорий для LLM
        self.all_cat = sorted(self.df['category'].unique())
        self.all_categories = []
        seen = set()
        for item in self.all_cat:
            item_lower = item.lower() 
            if item_lower not in seen:
                self.all_categories.append(item) 
                seen.add(item_lower)

        # Множество категорий, которые будут заменены на "технологии"
        self.tech_categories = {
            'GSM', 'LoRaWAN', 'Любая с LoRaWAN', 'PLC', 'RF', '485', 'RS232',
            'RF-433', 'NB-FI', 'NB-IOT', 'GSM+NB-Iot', 'Zigbee', 'Wi-fi',
            'RF/PLC', 'Группа тех-гий, вкл. LoRaWAN', 'Группа тех-гий без LoRaWAN'
        }

    def _ensure_parent_dir(self, path: str) -> None:
        """Создаёт директорию-родитель для файлов кэша (embeddings/index/cache)."""
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
            
    def _load_company_context(self) -> str:
        """Загружает содержимое всех .txt и .md файлов из папки company_context_dir."""
        if not self.company_context_dir or not os.path.isdir(self.company_context_dir):
            return ""
        parts = []
        for root, _, files in os.walk(self.company_context_dir):
            for file in files:
                if file.lower().endswith(('.txt', '.md')):
                    path = os.path.join(root, file)
                    try:
                        with open(path, 'r', encoding='utf-8') as f:
                            parts.append(f.read())
                    except Exception as e:
                        logger.warning(f"Не удалось прочитать {path}: {e}")
        return "\n\n---\n\n".join(parts)

    def _load_or_create_data(self) -> pd.DataFrame:
        """
        Загружает данные из кэша (pickle), если все три файла (кэш данных, эмбеддинги, индекс) существуют.
        Иначе читает Excel, создаёт DataFrame и сохраняет в кэш.
        """
        # Проверяем наличие всех трёх кэшей
        all_caches_exist = (
            os.path.exists(self.embeddings_path) and
            os.path.exists(self.index_path) and
            os.path.exists(self.data_cache_path)
        )
        if all_caches_exist:
            logger.info("Загрузка данных из кэша %s", self.data_cache_path)
            with open(self.data_cache_path, 'rb') as f:
                df = pickle.load(f)
            logger.info("Данные загружены из кэша: %d записей", len(df))
            return df

        logger.info("Кэш данных не найден или неполный, загружаем из Excel...")
        df = self._load_and_prepare_data()

        # Сохраняем DataFrame в кэш
        self._ensure_parent_dir(self.data_cache_path)
        with open(self.data_cache_path, 'wb') as f:
            pickle.dump(df, f)
        logger.info("Данные сохранены в кэш %s", self.data_cache_path)
        return df

    def _prepare_single_df(self, df: pd.DataFrame, source_name: str) -> pd.DataFrame:
        """Готовит один датафрейм: собирает текст и категорию с учётом возможных различий в названиях колонок."""
        # Возможные варианты названий для основных текстовых полей
        title_cols = ['Предмет тендера', 'Наименование закупки', 'Наименование лота']
        comment_cols = ['Комментарий', 'Описание', 'Примечание']
        customer_cols = ['Заказчик', 'Организация', 'Заказчик/организация']

        def first_existing(col_candidates):
            for c in col_candidates:
                if c in df.columns:
                    return c
            return None

        title_col = first_existing(title_cols)
        comment_col = first_existing(comment_cols)
        customer_col = first_existing(customer_cols)

        parts = []
        if title_col:
            parts.append(df[title_col].fillna(''))
        if comment_col:
            parts.append(df[comment_col].fillna(''))
        if customer_col:
            parts.append(df[customer_col].fillna(''))

        if not parts:
            raise ValueError(f"Не удалось найти текстовые колонки в файле {source_name}. Найденные столбцы: {df.columns.tolist()}")

        df['text'] = (
            (parts[0] if len(parts) > 0 else '') + ' ' +
            (parts[1] if len(parts) > 1 else '') + ' ' +
            (parts[2] if len(parts) > 2 else '')
        )

        df['text'] = df['text'].astype(str).str.strip()
        df = df[df['text'] != ''].reset_index(drop=True)

        # Категории: поддерживаем несколько возможных имён колонок
        category_cols = ['Работы', 'Категория', 'Category', 'D']
        cat_col = first_existing(category_cols)
        if not cat_col:
            logger.warning(f"В файле {source_name} не найдена колонка с категорией. Все объекты будут помечены как 'Не профиль'.")
            df['category'] = 'Не профиль'
        else:
            df['category'] = df[cat_col].fillna('Не профиль')

        df['source_file'] = source_name

        logger.info("Файл %s: столбцы=%s", source_name, df.columns.tolist())
        logger.info("Файл %s: примеры категорий=%s", source_name, df['category'].unique()[:10])
        return df

    def _load_and_prepare_data(self) -> pd.DataFrame:
        """Загружает один или несколько Excel и формирует общую обучающую выборку."""
        all_dfs = []
        for path in self.data_paths:
            if not os.path.exists(path):
                logger.warning("Файл %s не найден, пропускаем его при обучении RAG.", path)
                continue
            df_raw = pd.read_excel(path)
            prepared = self._prepare_single_df(df_raw, os.path.basename(path))
            all_dfs.append(prepared)

        if not all_dfs:
            raise ValueError(f"Не удалось загрузить ни одного Excel-файла из путей: {self.data_paths}")

        df = pd.concat(all_dfs, ignore_index=True)
        logger.info("Общая обучающая выборка: %d записей, %d категорий", len(df), df['category'].nunique())
        return df

    def _get_embeddings(self) -> np.ndarray:
        """Загружает или вычисляет эмбеддинги для всех текстов."""
        if os.path.exists(self.embeddings_path):
            with open(self.embeddings_path, 'rb') as f:
                embeddings = pickle.load(f)
            logger.info("Эмбеддинги загружены из кэша")
            return embeddings

        logger.info("Вычисление эмбеддингов для %d текстов...", len(self.df))
        texts = self.df['text'].tolist()
        # Для моделей семейства E5 важно использовать специальные префиксы
        if "e5" in self.model_name.lower():
            texts_to_encode = [f"passage: {t}" for t in texts]
        else:
            texts_to_encode = texts

        embeddings = self.embedder.encode(
            texts_to_encode,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        self._ensure_parent_dir(self.embeddings_path)
        with open(self.embeddings_path, 'wb') as f:
            pickle.dump(embeddings, f)
        logger.info("Эмбеддинги вычислены и сохранены")
        return embeddings

    def _get_faiss_index(self) -> faiss.Index:
        """Загружает или создаёт индекс FAISS."""
        if os.path.exists(self.index_path):
            index = faiss.read_index(self.index_path)
            logger.info("Индекс FAISS загружен из файла")
            return index

        logger.info("Создание индекса FAISS...")
        dimension = self.embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # косинусное сходство (после нормализации)
        index.add(self.embeddings)
        self._ensure_parent_dir(self.index_path)
        faiss.write_index(index, self.index_path)
        logger.info("Индекс FAISS создан и сохранён")
        return index

    def _compute_category_weights(self) -> Dict[str, float]:
        """Вычисляет веса категорий на основе частоты."""
        freq = self.df['category'].value_counts()
        N = len(self.df)
        if self.inverse_freq_mode == 'raw':
            return {cat: 1 / f for cat, f in freq.items()}
        elif self.inverse_freq_mode == 'log':
            return {cat: np.log(N / f) for cat, f in freq.items()}
        else:
            raise ValueError("inverse_freq_mode должен быть 'raw' или 'log'")

    def _compute_centroids(self) -> Dict[str, np.ndarray]:
        """Вычисляет нормализованные центроиды для каждой категории."""
        centroids = {}
        unique_categories = self.df['category'].unique()
        for cat in unique_categories:
            indices = self.df[self.df['category'] == cat].index.tolist()
            cat_embeddings = self.embeddings[indices]  # (n_cat, dim)
            centroid = np.mean(cat_embeddings, axis=0)
            # Нормализуем центроид (после усреднения он может быть не единичной длины)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
            centroids[cat] = centroid
        return centroids

    def _find_similar(self, text: str) -> List[Dict]:
        """Находит k похожих тендеров через FAISS и возвращает список с текстом, категорией, сходством и индексом."""
        # Для моделей семейства E5 используем префикс "query:" для запросов
        if "e5" in self.model_name.lower():
            query_text = f"query: {text}"
        else:
            query_text = text

        query_emb = self.embedder.encode([query_text], normalize_embeddings=True)
        similarities, indices = self.index.search(query_emb, self.k)
        similarities = similarities[0]
        indices = indices[0]

        similar = []
        for sim, idx in zip(similarities, indices):
            similar.append({
                'text': self.df.iloc[idx]['text'][:500],
                'category': self.df.iloc[idx]['category'],
                'similarity': float(sim),
                'index': int(idx)                       # добавили индекс
            })
        return similar

    def _fallback_vote(self, similar: List[Dict]) -> Tuple[str, float]:
        """Взвешенное голосование с учётом близости к центроиду и весов категорий."""
        # Если все сходства очень низкие, возвращаем "Не профиль"
        if not similar or max(item['similarity'] for item in similar) < 0.2:
            return "Не профиль", 0.0

        cat_weights = {}
        for item in similar:
            cat = item['category']
            sim = item['similarity']
            idx = item['index']

            # Близость соседа к центроиду своей категории
            centroid = self.centroids.get(cat)
            if centroid is not None:
                centroid_sim = np.dot(self.embeddings[idx], centroid)
                centroid_sim = max(centroid_sim, 0.0)   # защита от отрицательных
            else:
                centroid_sim = 1.0

            # Вес категории (обратная частота)
            cat_weight = self.category_weights.get(cat, 1.0) if self.use_inverse_freq else 1.0

            weight = sim * centroid_sim * cat_weight
            cat_weights[cat] = cat_weights.get(cat, 0) + weight

        if not cat_weights:
            return "Не профиль", 0.0

        # Сортируем по весу
        sorted_cats = sorted(cat_weights.items(), key=lambda x: x[1], reverse=True)
        top_cat, top_weight = sorted_cats[0]
        total_weight = sum(cat_weights.values())
        confidence = top_weight / total_weight if total_weight > 0 else 0.0

        # Если разрыв между первой и второй небольшой, снижаем уверенность
        if len(sorted_cats) > 1:
            second_weight = sorted_cats[1][1]
            if top_weight - second_weight < 0.1 * total_weight:
                confidence *= 0.8   # штраф за неопределённость

        return top_cat, confidence

    def _extract_text_from_record(self, record: dict) -> str:
        """
        Извлекает текст тендера из записи JSON для последующей классификации.
        Ожидается структура с полями title, description, customer, positions.
        """
        parts = []

        # Основные поля
        for field in ['title', 'description', 'customer', 'delivery_place']:
            if field in record and record[field]:
                parts.append(str(record[field]).strip())

        # Позиции (если есть)
        if 'positions' in record and isinstance(record['positions'], list):
            pos_names = []
            for pos in record['positions']:
                if isinstance(pos, dict) and 'name' in pos and pos['name']:
                    pos_names.append(str(pos['name']).strip())
            if pos_names:
                parts.append(' '.join(pos_names))

        # Если ничего не собрано, используем любое строковое поле (на всякий случай)
        if not parts:
            for v in record.values():
                if isinstance(v, str) and v:
                    parts.append(v)
                    break

        # Обрезаем до 512 токенов (приблизительно 1500 символов, но лучше через токенизатор)
        full_text = ' '.join(parts)
        tokens = self.embedder.tokenizer.tokenize(full_text)
        if len(tokens) > 512:
            tokens = tokens[:512]
            full_text = self.embedder.tokenizer.convert_tokens_to_string(tokens)
        return full_text

    async def _call_llm_async(self, tender_text: str, similar: List[Dict], max_retries: int = 2):
        """Асинхронная версия вызова LLM."""
        company_context = self._load_company_context()
        company_context_block = f"""
    Дополнительная информация о компании (наши приоритеты и правила отбора):
    {company_context}
    """ if company_context else ""

        # Формируем примеры из похожих тендеров
        examples = []
        for i, t in enumerate(similar, 1):
            examples.append(
                f"Пример {i} (сходство: {t['similarity']:.3f}):\n"
                f"Текст: {t['text']}\n"
                f"Категория: {t['category']}"
            )
        examples_text = "\n\n".join(examples)

        categories_str = ", ".join(self.all_categories)

        base_prompt = f"""Ты — эксперт по классификации тендеров. Твоя задача — строго по заданному списку категорий определить категорию нового тендера, а также оценить интерес тендера для нашей компании по шкале от 0 до 1. Используй информацию о компании и правила отбора, приведённые ниже. Не выдумывай категории, используй только приведенные из списка.

    {company_context_block}

    Список допустимых категорий (только эти, никаких других):
    {categories_str}

    Если ни одна из категорий не подходит, выбери "Не профиль". Это важно: "Не профиль" — это тоже допустимая категория.

    Ниже приведены несколько похожих тендеров из базы (отсортированы по убыванию сходства). Используй их как примеры для анализа.

    {examples_text}

    Новый тендер для классификации:
    {tender_text}

    На основе информации о компании и правил отбора определи:
    - К какой категории относится тендер? (выбери строго из списка)
    - Насколько ты уверен в категории (число от 0 до 1)?
    - Краткое пояснение, почему выбрана эта категория.
    - Оцени интерес тендера для нашей компании числом от 0 до 1...
    - Краткое пояснение, почему поставлена такая оценка интереса...

    Формат ответа строго JSON:
    {{
        "category": "точное название категории из списка",
        "confidence": число от 0 до 1,
        "reasoning": "краткое объяснение на русском",
        "interest_score": число от 0 до 1,
        "interest_reasoning": "подробное объяснение оценки интереса"
    }}

    Твой ответ:"""

        strict_prompt = base_prompt + "\n\nВАЖНО: Ты ОБЯЗАН выбрать категорию строго из приведённого списка. Недопустимо использовать категории, которых нет в списке. Если сомневаешься, выбирай 'Не профиль'."

        model_name = self.llm_config.get('model', 'llama3.2')
        url = self.llm_config.get('url', 'http://localhost:11434/api/generate')

        for attempt in range(max_retries):
            prompt = base_prompt if attempt == 0 else strict_prompt
            response = await ask_llama(prompt, model_name, url)
            if not response:
                continue
            result_obj = repair_json_fragment(response)
            if not isinstance(result_obj, dict):
                continue
            category = result_obj.get('category', 'Не профиль').strip()
            confidence = float(result_obj.get('confidence', 0.5))
            reasoning = result_obj.get('reasoning', '')
            interest_score = float(result_obj.get('interest_score', 0.0))
            interest_score = max(0.0, min(1.0, interest_score))
            interest_reasoning = result_obj.get('interest_reasoning', '')
            if category in self.all_categories:
                return category, confidence, reasoning, interest_score >= self.interest_threshold, interest_score, interest_reasoning
        # Если не удалось
        return "Не профиль(ошибка_llm)", 0.0, "LLM не выбрала категорию", False, 0.0, "Ошибка"

    def _combine_predictions(
        self,
        fb_cat: str,
        fb_conf: float,
        llm_cat: str,
        llm_conf: float,
        llm_reason: str,
        llm_interesting: bool,
        llm_interest_score: float,
        llm_interest_reason: str,
        similar: List[Dict]
    ) -> Tuple[str, float, str, bool, float, str]:
        """
        Комбинирует предсказания fallback и LLM.
        Возвращает (категория, уверенность, объяснение, интересен (bool), interest_score, пояснение интереса).
        """
        # Если LLM вернул явную ошибку или недопустимую категорию, используем fallback для категории,
        # но интересность оставляем от LLM (она может быть осмысленной)
        if llm_cat == "Не профиль(ошибка_llm)":
            return ("Не профиль(ошибка_llm)", 0.0, "Ошибка LLM при классификации",
                llm_interesting, llm_interest_score, llm_interest_reason)

        if llm_cat not in self.all_categories:
            logger.info(f"LLM вернула недопустимую категорию '{llm_cat}', используем fallback: '{fb_cat}'")
            return (fb_cat, fb_conf,
                    f"Fallback (LLM error: {llm_reason})",
                    llm_interesting, llm_interest_score, llm_interest_reason)

        # Если обе категории совпадают – объединяем уверенность
        if fb_cat == llm_cat:
            combined_conf = max(fb_conf, llm_conf)
            return llm_cat, combined_conf, llm_reason, llm_interesting, llm_interest_score, llm_interest_reason

        # Категории различаются – финальное решение за LLM, но уверенность можно скорректировать
        adjusted_conf = llm_conf
        # Лёгкая корректировка уверенности, если fallback очень уверен
        fb_neighbors_sim = np.mean([
            item['similarity'] for item in similar if item['category'] == fb_cat
        ]) if any(item['category'] == fb_cat for item in similar) else 0.0

        if fb_conf >= 0.6 and fb_neighbors_sim >= 0.3:
            adjusted_conf = min(1.0, llm_conf + 0.1)

        explanation = f"{llm_reason} (fallback предлагал категорию '{fb_cat}' с уверенностью {fb_conf:.2f})"
        return llm_cat, adjusted_conf, explanation, llm_interesting, llm_interest_score, llm_interest_reason

    async def classify(self, tender_text: str) -> Dict[str, Any]:
        similar = self._find_similar(tender_text)
        max_sim = max([item['similarity'] for item in similar]) if similar else 0.0

        if max_sim < self.faiss_threshold:
            return {
                'text': tender_text,
                'predicted_category': 'Не профиль',
                'confidence': 0.0,
                'reasoning': f'Максимальное сходство с базой {max_sim:.3f} ниже порога {self.faiss_threshold}',
                'neighbors': similar,
                'max_similarity': max_sim,
                'is_interesting': False,
                'interest_score': 0.0,
                'interest_reasoning': 'Низкое сходство с обучающей базой.',
            }

        fb_cat, fb_conf = self._fallback_vote(similar)
        llm_cat, llm_conf, llm_reason, llm_interesting, llm_interest_score, llm_interest_reason = await self._call_llm_async(tender_text, similar)

        final_cat, final_conf, final_reason, final_interesting, final_interest_score, final_interest_reason = self._combine_predictions(
            fb_cat, fb_conf,
            llm_cat, llm_conf, llm_reason,
            llm_interesting, llm_interest_score, llm_interest_reason,
            similar
        )

        if final_cat in self.tech_categories:
            final_reason += f" (категория обобщена до 'Технологии', исходно: {final_cat})"
            final_cat = "Технологии"

        return {
            'text': tender_text,
            'predicted_category': final_cat,
            'confidence': final_conf,
            'reasoning': final_reason,
            'neighbors': similar,
            'max_similarity': max_sim,
            'fallback_prediction': fb_cat,
            'fallback_confidence': fb_conf,
            'llm_prediction': llm_cat,
            'llm_confidence': llm_conf,
            'llm_reasoning': llm_reason,
            'interest_score': final_interest_score,
            'is_interesting': final_interesting,
            'interest_reasoning': final_interest_reason,
        }

    async def process_file_async(self, input_jsonl: str, output_jsonl: str, concurrency: int = 5):
        records = []
        with open(input_jsonl, 'r', encoding='utf-8') as f_in:
            for line_num, line in enumerate(f_in, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    records.append((line_num, record))
                except json.JSONDecodeError as e:
                    logger.error(f"Строка {line_num}: ошибка парсинга JSON: {e}")
                    continue

        sem = asyncio.Semaphore(concurrency)

        async def process_one(line_num, record):
            async with sem:
                text = self._extract_text_from_record(record)
                if not text:
                    logger.warning(f"Строка {line_num} не содержит текста, сохраняем исходную запись")
                    return record
                try:
                    result = await self.classify(text)
                    record.update({
                        'predicted_category': result['predicted_category'],
                        'confidence': result['confidence'],
                        'reasoning': result['reasoning'],
                        'interest_score': result['interest_score'],
                        'is_interesting': result['is_interesting'],
                        'interest_reasoning': result['interest_reasoning']
                    })
                except Exception as e:
                    logger.error(f"Строка {line_num}: ошибка классификации: {e}")
                return record

        tasks = [asyncio.create_task(process_one(num, rec)) for num, rec in records]
        processed_records = await asyncio.gather(*tasks)

        with open(output_jsonl, 'w', encoding='utf-8') as f_out:
            for rec in processed_records:
                f_out.write(json.dumps(rec, ensure_ascii=False) + '\n')
        logger.info(f"Асинхронная обработка завершена. Записано {len(processed_records)} записей.")


# ========== Пример использования ==========
if __name__ == "__main__":
    # Конфигурация LLM для Ollama (LLaMA)
    llm_config = {
        'provider': 'ollama',
        'model': 'qwen3:4b-instruct',  
        'url': 'http://localhost:11434/api/generate'
    }

    # Инициализация классификатора (один раз)
    classifier = TenderClassifierRAG(
        data_path=['2401_Закупки.xlsx', '2601_Закупки.xlsx'],
        embeddings_path='embeddings.pkl',
        index_path='faiss.index',
        data_cache_path='data_cache.pkl',
        k=10,
        faiss_threshold=0.15,
        use_inverse_freq=True,
        inverse_freq_mode='log',
        llm_config=llm_config,
        company_context_dir='company_context',  # папка с .txt и .md файлами о компании
        interest_threshold=0.5                   # порог для is_interesting
    )

    # Обработка файла
    classifier.process_file('extracted_texts/all_tenders_data_20260310_115514.jsonl', 'output_results.jsonl')