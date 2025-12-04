"""
Модуль для підготовки датасету для класифікації (передбачення Gender користувача).
Оптимізована версія з використанням Pandas для швидшої обробки.

Author: Bohdan Osmuk
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, Optional

# Налаштування логування
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# КОНСТАНТИ
# ============================================================================

TOP_GENRES = [
    "Action", "Adventure", "Comedy", "Drama", "Fantasy", "Romance",
    "Sci-Fi", "Slice of Life", "Sports", "Supernatural", "Mystery",
    "Horror", "Psychological", "Thriller", "Mecha", "Music", "School",
    "Shounen", "Shoujo", "Seinen", "Josei", "Ecchi", "Harem"
]

ANIME_TYPES = ["TV", "Movie", "OVA", "ONA", "Special"]


# ============================================================================
# ДОПОМІЖНІ ФУНКЦІЇ
# ============================================================================

def save_metadata(metadata: Dict, output_path: str, filename: str) -> None:
    """Зберігає metadata у JSON файл."""
    os.makedirs(output_path, exist_ok=True)
    file_path = os.path.join(output_path, filename)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info(f"Metadata збережено: {file_path}")


def normalize_column(series: pd.Series) -> pd.Series:
    """Min-max нормалізація."""
    min_val = series.min()
    max_val = series.max()
    if max_val != min_val:
        return (series - min_val) / (max_val - min_val)
    return series * 0


# ============================================================================
# ГОЛОВНІ ФУНКЦІЇ
# ============================================================================

def load_and_filter_users(users_path: str, sample_size: int = 20000) -> pd.DataFrame:
    """
    Завантажує користувачів та фільтрує за статтю.
    """
    logger.info(f"Завантаження користувачів з {users_path}...")
    
    users_df = pd.read_csv(users_path)
    logger.info(f"Завантажено {len(users_df)} користувачів")
    
    # Фільтруємо за статтю
    users_df = users_df[users_df['Gender'].isin(['Male', 'Female'])].copy()
    logger.info(f"Користувачів з відомою статтю: {len(users_df)}")
    
    # Sampling
    if len(users_df) > sample_size:
        users_df = users_df.sample(n=sample_size, random_state=42)
        logger.info(f"Відібрано {sample_size} користувачів")
    
    return users_df


def load_ratings_for_users(ratings_path: str, user_ids: set, chunksize: int = 500000) -> pd.DataFrame:
    """
    Завантажує ratings тільки для вибраних користувачів (читає частинами).
    """
    logger.info(f"Завантаження ratings для {len(user_ids)} користувачів...")
    
    ratings_list = []
    total_rows = 0
    filtered_rows = 0
    
    for chunk in pd.read_csv(ratings_path, chunksize=chunksize):
        total_rows += len(chunk)
        filtered_chunk = chunk[chunk['user_id'].isin(user_ids)]
        filtered_rows += len(filtered_chunk)
        if len(filtered_chunk) > 0:
            ratings_list.append(filtered_chunk)
        
        if total_rows % 5000000 == 0:
            logger.info(f"  Оброблено {total_rows:,} рядків, знайдено {filtered_rows:,} ratings")
    
    ratings_df = pd.concat(ratings_list, ignore_index=True)
    logger.info(f"Завантажено {len(ratings_df):,} ratings")
    
    return ratings_df


def aggregate_user_features(
    ratings_df: pd.DataFrame,
    users_df: pd.DataFrame,
    anime_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Агрегує features користувача.
    """
    logger.info("Агрегування features користувачів...")
    
    # Об'єднуємо ratings з anime info
    ratings_with_anime = ratings_df.merge(
        anime_df[['anime_id', 'Genres', 'Type', 'Score']],
        on='anime_id',
        how='left'
    )
    
    # Базова статистика по користувачу
    logger.info("  Базова статистика...")
    user_stats = ratings_with_anime.groupby('user_id').agg({
        'rating': ['count', 'mean', 'std', 'min', 'max'],
        'anime_id': 'nunique'
    }).reset_index()
    
    user_stats.columns = [
        'user_id', 'total_ratings', 'avg_rating', 'rating_std',
        'min_rating', 'max_rating', 'unique_anime_count'
    ]
    
    # Агрегація жанрів
    logger.info("  Агрегація жанрів...")
    for genre in TOP_GENRES:
        col_name = f"genre_count_{genre.lower().replace(' ', '_').replace('-', '_')}"
        ratings_with_anime[col_name] = ratings_with_anime['Genres'].str.contains(
            genre, na=False, case=False
        ).astype(int)
    
    genre_cols = [f"genre_count_{g.lower().replace(' ', '_').replace('-', '_')}" for g in TOP_GENRES]
    genre_agg = ratings_with_anime.groupby('user_id')[genre_cols].sum().reset_index()
    user_stats = user_stats.merge(genre_agg, on='user_id', how='left')
    
    # Агрегація типів
    logger.info("  Агрегація типів...")
    for anime_type in ANIME_TYPES:
        col_name = f"type_count_{anime_type.lower()}"
        ratings_with_anime[col_name] = (ratings_with_anime['Type'] == anime_type).astype(int)
    
    type_cols = [f"type_count_{t.lower()}" for t in ANIME_TYPES]
    type_agg = ratings_with_anime.groupby('user_id')[type_cols].sum().reset_index()
    user_stats = user_stats.merge(type_agg, on='user_id', how='left')
    
    # Середній рейтинг за типом
    logger.info("  Середній рейтинг за типом...")
    for anime_type in ANIME_TYPES:
        col_name = f"avg_rating_{anime_type.lower()}"
        type_ratings = ratings_with_anime[ratings_with_anime['Type'] == anime_type]
        type_avg = type_ratings.groupby('user_id')['rating'].mean().reset_index()
        type_avg.columns = ['user_id', col_name]
        user_stats = user_stats.merge(type_avg, on='user_id', how='left')
    
    # Додаємо статистику користувача
    logger.info("  Додавання демографічних даних...")
    users_cols = users_df[['Mal ID', 'Gender', 'Days Watched', 'Mean Score', 
                           'Completed', 'Dropped', 'Plan to Watch', 
                           'Total Entries', 'Rewatched', 'Episodes Watched']].copy()
    users_cols.columns = ['user_id', 'Gender', 'days_watched', 'user_mean_score',
                          'user_completed', 'user_dropped', 'user_plan_to_watch',
                          'user_total_entries', 'user_rewatched', 'user_episodes_watched']
    
    user_features = user_stats.merge(users_cols, on='user_id', how='inner')
    
    return user_features


def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Створює додаткові features.
    """
    logger.info("Створення derived features...")
    
    # Genre diversity
    genre_cols = [c for c in df.columns if c.startswith('genre_count_')]
    df['genre_diversity'] = (df[genre_cols] > 0).sum(axis=1)
    
    # Rating consistency
    df['rating_consistency'] = df['rating_std'].fillna(0) / df['avg_rating'].replace(0, 1)
    
    # Preference intensity
    df['preference_intensity'] = (df['max_rating'] - df['min_rating']) / 9.0
    
    # Drop rate
    df['drop_rate'] = df['user_dropped'] / df['user_total_entries'].replace(0, 1)
    
    # Gender encoding
    df['gender_encoded'] = (df['Gender'] == 'Female').astype(int)
    
    # Genre percentages
    for col in genre_cols:
        pct_col = col.replace('_count_', '_pct_')
        df[pct_col] = df[col] / df['total_ratings'].replace(0, 1)
    
    # Нормалізація
    cols_to_normalize = ['total_ratings', 'avg_rating', 'days_watched',
                         'user_mean_score', 'user_completed', 'user_episodes_watched']
    for col in cols_to_normalize:
        if col in df.columns:
            df[f'{col}_normalized'] = normalize_column(df[col].fillna(0))
    
    # Заповнення NaN
    df = df.fillna(0)
    
    return df


def split_and_save(
    df: pd.DataFrame,
    output_path: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15
):
    """
    Stratified split та збереження.
    """
    logger.info("Розділення на train/val/test...")
    
    os.makedirs(output_path, exist_ok=True)
    
    # Stratified split
    from sklearn.model_selection import train_test_split
    
    train_df, temp_df = train_test_split(
        df, test_size=(1 - train_ratio), 
        stratify=df['gender_encoded'], random_state=42
    )
    
    val_size = val_ratio / (1 - train_ratio)
    val_df, test_df = train_test_split(
        temp_df, test_size=(1 - val_size),
        stratify=temp_df['gender_encoded'], random_state=42
    )
    
    logger.info(f"Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    logger.info(f"Validation: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    logger.info(f"Test: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
    
    # Збереження CSV
    train_df.to_csv(os.path.join(output_path, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_path, "validation.csv"), index=False)
    test_df.to_csv(os.path.join(output_path, "test.csv"), index=False)
    
    # Збереження Parquet
    train_df.to_parquet(os.path.join(output_path, "train.parquet"), index=False)
    val_df.to_parquet(os.path.join(output_path, "validation.parquet"), index=False)
    test_df.to_parquet(os.path.join(output_path, "test.parquet"), index=False)
    
    logger.info(f"Дані збережено в {output_path}")
    
    # Split info
    split_info = {
        "train_size": len(train_df),
        "validation_size": len(val_df),
        "test_size": len(test_df),
        "stratified": True
    }
    save_metadata(split_info, output_path, "split_info.json")
    
    return train_df, val_df, test_df


def main():
    """
    Головна функція.
    """
    logger.info("=" * 60)
    logger.info("ПІДГОТОВКА ДАТАСЕТУ ДЛЯ КЛАСИФІКАЦІЇ")
    logger.info("=" * 60)
    
    # Шляхи
    data_path = "data"
    output_path = os.path.join(data_path, "ml_datasets", "classification")
    
    # Параметри
    SAMPLE_USERS = 20000
    MIN_RATINGS = 10
    
    try:
        # 1. Завантаження користувачів
        users_df = load_and_filter_users(
            os.path.join(data_path, "users-details-2023.csv"),
            sample_size=SAMPLE_USERS
        )
        user_ids = set(users_df['Mal ID'].values)
        
        # 2. Завантаження ratings (тільки для вибраних користувачів)
        ratings_df = load_ratings_for_users(
            os.path.join(data_path, "users-score-2023.csv"),
            user_ids
        )
        
        # 3. Завантаження anime
        logger.info("Завантаження anime...")
        anime_df = pd.read_csv(os.path.join(data_path, "anime-filtered.csv"))
        logger.info(f"Завантажено {len(anime_df)} аніме")
        
        # 4. Агрегація features
        user_features = aggregate_user_features(ratings_df, users_df, anime_df)
        logger.info(f"Агреговано features для {len(user_features)} користувачів")
        
        # 5. Фільтрація за мінімальною кількістю оцінок
        user_features = user_features[user_features['total_ratings'] >= MIN_RATINGS]
        logger.info(f"Після фільтрації (>= {MIN_RATINGS} оцінок): {len(user_features)}")
        
        # 6. Derived features
        user_features = create_derived_features(user_features)
        
        # 7. Баланс класів
        class_balance = user_features['Gender'].value_counts().to_dict()
        logger.info(f"Баланс класів: {class_balance}")
        
        # 8. Metadata
        feature_cols = [c for c in user_features.columns 
                        if c not in ['user_id', 'Gender', 'gender_encoded']]
        metadata = {
            "task": "classification",
            "target": "gender_encoded",
            "target_labels": {"Male": 0, "Female": 1},
            "feature_count": len(feature_cols),
            "features": feature_cols,
            "total_samples": len(user_features),
            "class_balance": class_balance,
            "min_ratings_threshold": MIN_RATINGS
        }
        save_metadata(metadata, output_path, "preprocessing_info.json")
        
        # 9. Split та збереження
        split_and_save(user_features, output_path)
        
        logger.info("=" * 60)
        logger.info("ГОТОВО!")
        logger.info(f"Результати: {output_path}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Помилка: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
