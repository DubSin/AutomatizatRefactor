import pandas as pd
import numpy as np
import random
import pickle
import os
import faiss
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import TripletEvaluator
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity
import logging
import re
from pipeline.config import TRAIN_FOLDER
from logging.handlers import RotatingFileHandler

from pipeline.config import LOG_DIR, LOG_MAX_BYTES, LOG_BACKUP_COUNT

os.makedirs(LOG_DIR, exist_ok=True)
_train_log_path = os.path.join(LOG_DIR, "train.log")

_handlers = [
    logging.StreamHandler(),
    RotatingFileHandler(
        _train_log_path,
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    ),
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=_handlers,
    force=True,
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# 1. Загрузка и подготовка данных
# ------------------------------------------------------------
def load_and_prepare_data(file_path: str) -> pd.DataFrame:
    """Загружает Excel, формирует текстовое поле, чистит, удаляет пустые строки."""
    df = pd.read_excel(file_path, header=0, dtype=str)
    item_col = 'Предмет тендера'
    customer_col = 'Заказчик'
    category_col = 'D'

    df = df.dropna(subset=[category_col, item_col])
    df[category_col] = df[category_col].astype(str).str.strip()
    df[item_col] = df[item_col].astype(str).str.strip()
    df[customer_col] = df[customer_col].fillna('').astype(str).str.strip()

    def get_customer_short(cust):
        if not cust:
            return ''
        words = cust.split()[:5]
        return ' '.join(words)

    df['text'] = df.apply(
        lambda row: (row[item_col] + ' ' + get_customer_short(row[customer_col])).strip(),
        axis=1
    )

    df = df.drop_duplicates(subset=['text'])
    df['text'] = df['text'].apply(lambda x: re.sub(r'\s+', ' ', x).strip())
    
    # Сбрасываем индекс, чтобы он стал последовательным
    df = df.reset_index(drop=True)

    logger.info(f"Загружено {len(df)} записей, категорий: {df[category_col].nunique()}")
    return df

# ------------------------------------------------------------
# 2. Обрезка текста до max_length токенов
# ------------------------------------------------------------
def truncate_texts(texts, tokenizer, max_length=512):
    truncated = []
    for t in texts:
        tokens = tokenizer.tokenize(t)
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
            t = tokenizer.convert_tokens_to_string(tokens)
        truncated.append(t)
    return truncated

# ------------------------------------------------------------
# 3. Генерация триплетов с hard negative mining
# ------------------------------------------------------------
def generate_hard_triplets(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    category_col: str = 'D',
    text_col: str = 'text',
    k_negs: int = 10,
    max_triplets_per_category: int = 2000
):
    triplets = []
    categories = df[category_col].unique()
    emb_len = len(embeddings)

    for cat in categories:
        cat_indices = df[df[category_col] == cat].index.tolist()
        # Фильтруем индексы, выходящие за пределы массива эмбеддингов
        cat_indices = [i for i in cat_indices if i < emb_len]
        if len(cat_indices) < 2:
            continue

        other_indices = df[df[category_col] != cat].index.tolist()
        other_indices = [i for i in other_indices if i < emb_len]
        if not other_indices:
            continue

        other_embs = embeddings[other_indices]  # теперь безопасно

        n_triplets = min(len(cat_indices) * 2, max_triplets_per_category)
        for _ in range(n_triplets):
            anchor_idx = random.choice(cat_indices)
            anchor_text = df.loc[anchor_idx, text_col]
            anchor_emb = embeddings[anchor_idx].reshape(1, -1)

            pos_candidates = [i for i in cat_indices if i != anchor_idx]
            if not pos_candidates:
                continue
            pos_idx = random.choice(pos_candidates)
            positive_text = df.loc[pos_idx, text_col]

            sims = cosine_similarity(anchor_emb, other_embs)[0]
            top_k_indices = np.argsort(sims)[-k_negs:]
            # Преобразуем индексы из other_embs обратно в глобальные индексы
            selected_neg_indices = [other_indices[idx] for idx in top_k_indices]
            neg_idx = random.choice(selected_neg_indices)
            negative_text = df.loc[neg_idx, text_col]

            triplets.append((anchor_text, positive_text, negative_text))

    logger.info(f"Создано {len(triplets)} триплетов")
    return triplets

# ------------------------------------------------------------
# 4. Основной скрипт дообучения
# ------------------------------------------------------------
def main():
    # Параметры
    excel_path = "2601_Закупки.xlsx"
    os.makedirs(TRAIN_FOLDER, exist_ok=True)

    cache_emb_path = os.path.join(TRAIN_FOLDER, "embeddings_cache.pkl")
    model_name = "intfloat/multilingual-e5-large"
    output_model_path = os.path.join(TRAIN_FOLDER, "fine_tuned_e5_tender")
    batch_size = 16
    epochs = 3
    learning_rate = 2e-5
    max_triplets_per_category = 2000
    k_negs = 10

    # Загрузка и подготовка данных
    df = load_and_prepare_data(excel_path)
    texts = df['text'].tolist()
    categories = df['D'].tolist()

    base_model = SentenceTransformer(model_name)
    tokenizer = base_model.tokenizer
    texts_trunc = truncate_texts(texts, tokenizer, max_length=512)

    # Эмбеддинги для hard negative mining
    if os.path.exists(cache_emb_path):
        with open(cache_emb_path, 'rb') as f:
            embeddings = pickle.load(f)
        logger.info("Эмбеддинги загружены из кэша")
    else:
        logger.info("Вычисляем эмбеддинги...")
        embeddings = base_model.encode(texts_trunc, normalize_embeddings=True, show_progress_bar=True)
        with open(cache_emb_path, 'wb') as f:
            pickle.dump(embeddings, f)
        logger.info("Эмбеддинги сохранены в кэш")

    # Генерация триплетов
    triplets = generate_hard_triplets(
        df,
        embeddings,
        category_col='D',
        text_col='text',
        k_negs=k_negs,
        max_triplets_per_category=max_triplets_per_category
    )

    logger.info(triplets)
    # Разделение
    train_triplets, val_triplets = train_test_split(triplets, test_size=0.1, random_state=42)
    logger.info(f"Train: {len(train_triplets)}, Val: {len(val_triplets)}")

    # Подготовка данных для обучения
    train_examples = [InputExample(texts=[a, p, n]) for a, p, n in train_triplets]
    val_examples = [InputExample(texts=[a, p, n]) for a, p, n in val_triplets]

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)

    # Модель
    model = SentenceTransformer(model_name)
    train_loss = losses.TripletLoss(model)

    # Валидационный evaluator
    val_anchors = [ex.texts[0] for ex in val_examples]
    val_positives = [ex.texts[1] for ex in val_examples]
    val_negatives = [ex.texts[2] for ex in val_examples]
    if len(val_anchors) > 1000:
        indices = random.sample(range(len(val_anchors)), 1000)
        val_anchors = [val_anchors[i] for i in indices]
        val_positives = [val_positives[i] for i in indices]
        val_negatives = [val_negatives[i] for i in indices]

    evaluator = TripletEvaluator(val_anchors, val_positives, val_negatives, name='val_triplet')

    # Обучение
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=epochs,
        warmup_steps=100,
        optimizer_params={'lr': learning_rate},
        output_path=output_model_path,
        save_best_model=True,
        show_progress_bar=True
    )

    logger.info(f"Обучение завершено. Модель сохранена в {output_model_path}")

    # --------------------------------------------------------
    # Дополнительно: вычисляем эмбеддинги для всей базы с дообученной моделью и строим FAISS-индекс
    # --------------------------------------------------------
    logger.info("Вычисление эмбеддингов для всей базы с дообученной моделью...")
    final_model = SentenceTransformer(output_model_path)
    final_texts_trunc = truncate_texts(texts, final_model.tokenizer, max_length=512)
    final_embeddings = final_model.encode(final_texts_trunc, normalize_embeddings=True, show_progress_bar=True)

    # Сохраняем эмбеддинги
    final_emb_path = os.path.join(TRAIN_FOLDER, "embeddings_final.pkl")
    with open(final_emb_path, 'wb') as f:
        pickle.dump(final_embeddings, f)
    logger.info(f"Финальные эмбеддинги сохранены в {final_emb_path}")

    # Строим и сохраняем FAISS-индекс
    dimension = final_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(final_embeddings)
    index_path = os.path.join(TRAIN_FOLDER, "faiss_final.index")
    faiss.write_index(index, index_path)
    logger.info(f"Индекс FAISS сохранён в {index_path}")


if __name__ == "__main__":
    main()