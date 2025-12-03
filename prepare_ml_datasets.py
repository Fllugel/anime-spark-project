"""
Модуль для підготовки датасетів для Machine Learning моделей.

Цей модуль створює два окремих датасети:
1. Регресія - передбачення Score аніме за features
2. Класифікація - передбачення Gender користувача за його anime preferences

Author: Bohdan Osmuk
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Optional

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, count, sum as spark_sum, avg, when, lit, split, explode,
    array_contains, size, stddev, min as spark_min, max as spark_max,
    regexp_replace, trim, lower, coalesce, rand, row_number,
    collect_list, concat_ws, expr, countDistinct
)
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, FloatType, DoubleType
)
from pyspark.sql.window import Window
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml import Pipeline

# Налаштування логування
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# КОНСТАНТИ
# ============================================================================

# Топ жанри для one-hot encoding
TOP_GENRES = [
    "Action", "Adventure", "Comedy", "Drama", "Fantasy", "Romance",
    "Sci-Fi", "Slice of Life", "Sports", "Supernatural", "Mystery",
    "Horror", "Psychological", "Thriller", "Mecha", "Music", "School",
    "Shounen", "Shoujo", "Seinen", "Josei", "Ecchi", "Harem",
    "Isekai", "Military", "Historical", "Demons", "Magic", "Super Power",
    "Vampire"
]

# Типи аніме для encoding
ANIME_TYPES = ["TV", "Movie", "OVA", "ONA", "Special", "Music"]

# Джерела аніме
ANIME_SOURCES = [
    "Manga", "Original", "Light novel", "Visual novel", "Game",
    "Novel", "Web manga", "4-koma manga", "Other"
]

# Рейтинги аніме
ANIME_RATINGS = [
    "G - All Ages", "PG - Children", "PG-13 - Teens 13 or older",
    "R - 17+ (violence & profanity)", "R+ - Mild Nudity", "Rx - Hentai"
]

# Топ студії для feature extraction
TOP_STUDIOS = [
    "Toei Animation", "Madhouse", "Sunrise", "J.C.Staff", "A-1 Pictures",
    "Bones", "Production I.G", "Studio Pierrot", "MAPPA", "Kyoto Animation",
    "ufotable", "Wit Studio", "CloverWorks", "Studio Ghibli", "Shaft",
    "TMS Entertainment", "OLM", "Brain's Base", "Doga Kobo", "P.A. Works"
]


# ============================================================================
# ДОПОМІЖНІ ФУНКЦІЇ
# ============================================================================

def get_spark_session(app_name: str = "ML_Dataset_Preparation") -> SparkSession:
    """
    Створює або отримує SparkSession.
    
    Args:
        app_name: Назва Spark application
        
    Returns:
        SparkSession instance
    """
    return SparkSession.builder \
        .appName(app_name) \
        .config("spark.sql.shuffle.partitions", "200") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()


def load_csv_data(spark: SparkSession, file_path: str) -> DataFrame:
    """
    Завантажує CSV файл в DataFrame.
    
    Args:
        spark: SparkSession
        file_path: Шлях до CSV файлу
        
    Returns:
        DataFrame з даними
    """
    logger.info(f"Завантаження даних з {file_path}")
    return spark.read.csv(file_path, header=True, inferSchema=True)


def save_dataset(
    df: DataFrame,
    output_path: str,
    format_type: str = "both"
) -> None:
    """
    Зберігає DataFrame у CSV та/або Parquet форматі.
    
    Args:
        df: DataFrame для збереження
        output_path: Базовий шлях (без розширення)
        format_type: "csv", "parquet", або "both"
    """
    if format_type in ["csv", "both"]:
        csv_path = f"{output_path}.csv"
        logger.info(f"Збереження CSV: {csv_path}")
        df.coalesce(1).write.mode("overwrite").option("header", "true").csv(csv_path)
    
    if format_type in ["parquet", "both"]:
        parquet_path = f"{output_path}.parquet"
        logger.info(f"Збереження Parquet: {parquet_path}")
        df.write.mode("overwrite").parquet(parquet_path)


def save_metadata(
    metadata: Dict,
    output_path: str,
    filename: str
) -> None:
    """
    Зберігає metadata у JSON файл.
    
    Args:
        metadata: Словник з метаданими
        output_path: Шлях до директорії
        filename: Назва файлу
    """
    os.makedirs(output_path, exist_ok=True)
    file_path = os.path.join(output_path, filename)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info(f"Metadata збережено: {file_path}")


# ============================================================================
# ENCODING ФУНКЦІЇ
# ============================================================================

def encode_genres(df: DataFrame, genres_column: str = "Genres") -> DataFrame:
    """
    Кодує жанри як binary columns (one-hot encoding для multi-label).
    
    Args:
        df: DataFrame з колонкою жанрів
        genres_column: Назва колонки з жанрами
        
    Returns:
        DataFrame з додатковими binary колонками для кожного жанру
    """
    logger.info("Кодування жанрів...")
    
    result_df = df
    
    for genre in TOP_GENRES:
        col_name = f"genre_{genre.lower().replace(' ', '_').replace('-', '_')}"
        result_df = result_df.withColumn(
            col_name,
            when(col(genres_column).contains(genre), 1).otherwise(0)
        )
    
    logger.info(f"Додано {len(TOP_GENRES)} колонок для жанрів")
    return result_df


def encode_categorical(
    df: DataFrame,
    column: str,
    categories: List[str],
    prefix: str
) -> DataFrame:
    """
    Кодує categorical змінну як one-hot encoding.
    
    Args:
        df: DataFrame
        column: Назва колонки для кодування
        categories: Список можливих категорій
        prefix: Префікс для нових колонок
        
    Returns:
        DataFrame з додатковими binary колонками
    """
    result_df = df
    
    for category in categories:
        col_name = f"{prefix}_{category.lower().replace(' ', '_').replace('-', '_').replace('+', 'plus')}"
        result_df = result_df.withColumn(
            col_name,
            when(col(column) == category, 1).otherwise(0)
        )
    
    return result_df


def extract_duration_minutes(df: DataFrame, duration_column: str = "Duration") -> DataFrame:
    """
    Витягує тривалість в хвилинах з текстового поля.
    
    Args:
        df: DataFrame з колонкою тривалості
        duration_column: Назва колонки
        
    Returns:
        DataFrame з числовою колонкою duration_minutes
    """
    return df.withColumn(
        "duration_minutes",
        when(
            col(duration_column).contains("hr"),
            regexp_replace(
                regexp_replace(col(duration_column), r"(\d+)\s*hr.*?(\d+)\s*min.*", "$1"),
                r"[^\d]", ""
            ).cast("int") * 60 +
            coalesce(
                regexp_replace(
                    regexp_replace(col(duration_column), r".*?(\d+)\s*min.*", "$1"),
                    r"[^\d]", ""
                ).cast("int"),
                lit(0)
            )
        ).when(
            col(duration_column).contains("min"),
            regexp_replace(
                regexp_replace(col(duration_column), r"(\d+)\s*min.*", "$1"),
                r"[^\d]", ""
            ).cast("int")
        ).otherwise(lit(None))
    )


def extract_year(df: DataFrame, aired_column: str = "Aired") -> DataFrame:
    """
    Витягує рік випуску з поля Aired.
    
    Args:
        df: DataFrame
        aired_column: Назва колонки з датами
        
    Returns:
        DataFrame з колонкою release_year
    """
    return df.withColumn(
        "release_year",
        regexp_replace(
            regexp_replace(col(aired_column), r".*(\d{4}).*", "$1"),
            r"[^\d]", ""
        ).cast("int")
    )


# ============================================================================
# ПІДГОТОВКА ДАТАСЕТУ ДЛЯ РЕГРЕСІЇ
# ============================================================================

def prepare_regression_dataset(
    spark: SparkSession,
    anime_path: str,
    output_path: str
) -> DataFrame:
    """
    Підготовлює датасет для регресії (передбачення Score аніме).
    
    Args:
        spark: SparkSession
        anime_path: Шлях до файлу з даними про аніме
        output_path: Шлях для збереження результату
        
    Returns:
        Підготовлений DataFrame
    """
    logger.info("=" * 60)
    logger.info("ПІДГОТОВКА ДАТАСЕТУ ДЛЯ РЕГРЕСІЇ")
    logger.info("=" * 60)
    
    # Завантаження даних
    df = load_csv_data(spark, anime_path)
    initial_count = df.count()
    logger.info(f"Завантажено {initial_count} записів")
    
    # Фільтрація записів без Score
    df = df.filter(col("Score").isNotNull() & (col("Score") > 0))
    after_filter_count = df.count()
    logger.info(f"Після фільтрації Score: {after_filter_count} записів")
    
    # Витягування числових features
    df = extract_duration_minutes(df)
    df = extract_year(df)
    
    # Кодування жанрів
    df = encode_genres(df, "Genres")
    
    # Кодування Type
    df = encode_categorical(df, "Type", ANIME_TYPES, "type")
    
    # Кодування Source
    df = encode_categorical(df, "Source", ANIME_SOURCES, "source")
    
    # Кодування Rating
    df = encode_categorical(df, "Rating", ANIME_RATINGS, "rating")
    
    # Кодування топ студій
    for studio in TOP_STUDIOS:
        col_name = f"studio_{studio.lower().replace(' ', '_').replace('.', '').replace('-', '_')}"
        df = df.withColumn(
            col_name,
            when(col("Studios").contains(studio), 1).otherwise(0)
        )
    
    # Обробка Episodes (заміна Unknown на null)
    df = df.withColumn(
        "episodes_clean",
        when(col("Episodes") == "Unknown", lit(None))
        .otherwise(col("Episodes").cast("int"))
    )
    
    # Приведення числових колонок до правильних типів
    numeric_columns_to_cast = [
        "Members", "Favorites", "Popularity", "Watching", 
        "Completed", "On-Hold", "Dropped", "Ranked"
    ]
    
    for col_name in numeric_columns_to_cast:
        if col_name in df.columns:
            df = df.withColumn(col_name, col(col_name).cast("double"))
    
    # Числові features
    numeric_features = [
        "episodes_clean", "duration_minutes", "release_year",
        "Members", "Favorites", "Popularity", "Watching", 
        "Completed", "On-Hold", "Dropped"
    ]
    
    # Заповнення пропущених значень медіаною для числових колонок
    for col_name in numeric_features:
        if col_name in df.columns:
            # Обчислюємо медіану
            try:
                median_val = df.filter(col(col_name).isNotNull()) \
                    .approxQuantile(col_name, [0.5], 0.01)
                if median_val:
                    df = df.withColumn(
                        col_name,
                        coalesce(col(col_name), lit(median_val[0]))
                    )
            except Exception as e:
                logger.warning(f"Не вдалося обчислити медіану для {col_name}: {e}")
    
    # Нормалізація числових features (min-max scaling)
    for col_name in ["Members", "Favorites", "Popularity", "Watching", "Completed"]:
        if col_name in df.columns:
            try:
                stats = df.filter(col(col_name).isNotNull()).agg(
                    spark_min(col(col_name).cast("double")).alias("min_val"),
                    spark_max(col(col_name).cast("double")).alias("max_val")
                ).collect()[0]
                min_val, max_val = stats["min_val"], stats["max_val"]
                if max_val is not None and min_val is not None and max_val != min_val:
                    df = df.withColumn(
                        f"{col_name.lower()}_normalized",
                        (col(col_name).cast("double") - lit(min_val)) / lit(max_val - min_val)
                    )
            except Exception as e:
                logger.warning(f"Не вдалося нормалізувати {col_name}: {e}")
    
    # Вибір фінальних колонок
    genre_cols = [f"genre_{g.lower().replace(' ', '_').replace('-', '_')}" for g in TOP_GENRES]
    type_cols = [f"type_{t.lower().replace(' ', '_').replace('-', '_')}" for t in ANIME_TYPES]
    source_cols = [f"source_{s.lower().replace(' ', '_').replace('-', '_')}" for s in ANIME_SOURCES]
    rating_cols = [f"rating_{r.lower().replace(' ', '_').replace('-', '_').replace('+', 'plus')}" for r in ANIME_RATINGS]
    studio_cols = [f"studio_{s.lower().replace(' ', '_').replace('.', '').replace('-', '_')}" for s in TOP_STUDIOS]
    
    normalized_cols = [
        "members_normalized", "favorites_normalized", "popularity_normalized",
        "watching_normalized", "completed_normalized"
    ]
    
    # Фінальні колонки
    final_columns = (
        ["anime_id", "Name", "Score"] +  # ID та target
        ["episodes_clean", "duration_minutes", "release_year"] +  # Числові
        normalized_cols +  # Нормалізовані
        genre_cols + type_cols + source_cols + rating_cols + studio_cols  # Encoded
    )
    
    # Фільтруємо тільки існуючі колонки
    existing_columns = [c for c in final_columns if c in df.columns]
    df_final = df.select(existing_columns)
    
    # Видалення записів з критичними пропусками
    df_final = df_final.filter(col("Score").isNotNull())
    
    final_count = df_final.count()
    logger.info(f"Фінальний датасет: {final_count} записів")
    
    # Збереження metadata
    feature_names = [c for c in existing_columns if c not in ["anime_id", "Name", "Score"]]
    metadata = {
        "task": "regression",
        "target": "Score",
        "feature_count": len(feature_names),
        "features": feature_names,
        "total_samples": final_count,
        "genres_encoded": TOP_GENRES,
        "types_encoded": ANIME_TYPES,
        "sources_encoded": ANIME_SOURCES,
        "ratings_encoded": ANIME_RATINGS,
        "studios_encoded": TOP_STUDIOS
    }
    save_metadata(metadata, output_path, "preprocessing_info.json")
    
    return df_final


# ============================================================================
# ПІДГОТОВКА ДАТАСЕТУ ДЛЯ КЛАСИФІКАЦІЇ
# ============================================================================

def aggregate_user_features(
    spark: SparkSession,
    ratings_df: DataFrame,
    users_df: DataFrame,
    anime_df: DataFrame,
    rating_column: str = "rating"
) -> DataFrame:
    """
    Агрегує features користувача на основі його переглянутих аніме.
    
    Args:
        spark: SparkSession
        ratings_df: DataFrame з оцінками користувачів
        users_df: DataFrame з деталями користувачів
        anime_df: DataFrame з інформацією про аніме
        rating_column: Назва колонки з оцінками (rating або my_score)
        
    Returns:
        DataFrame з агрегованими features для кожного користувача
    """
    logger.info("Агрегування features користувачів...")
    
    # Вибираємо тільки потрібні колонки з ratings_df без дублікатів
    ratings_clean = ratings_df.select(
        col("user_id"),
        col("anime_id"),
        col(rating_column).alias("user_score")
    )
    
    # Об'єднуємо ratings з anime info
    ratings_with_anime = ratings_clean.join(
        anime_df.select(
            col("anime_id").alias("a_id"),
            col("Genres").alias("anime_genres"),
            col("Type").alias("anime_type"),
            col("Source").alias("anime_source"),
            col("Score").alias("anime_score")
        ),
        ratings_clean.anime_id == col("a_id"),
        "left"
    ).drop("a_id")
    
    # Агрегація по користувачу
    user_stats = ratings_with_anime.groupBy("user_id").agg(
        count("*").alias("total_ratings"),
        avg("user_score").alias("avg_rating"),
        stddev("user_score").alias("rating_std"),
        spark_min("user_score").alias("min_rating"),
        spark_max("user_score").alias("max_rating"),
        countDistinct("anime_id").alias("unique_anime_count")
    )
    
    # Агрегація жанрів для кожного користувача
    for genre in TOP_GENRES:
        col_name = f"genre_count_{genre.lower().replace(' ', '_').replace('-', '_')}"
        genre_agg = ratings_with_anime.groupBy("user_id").agg(
            spark_sum(
                when(col("anime_genres").contains(genre), 1).otherwise(0)
            ).alias(col_name)
        )
        user_stats = user_stats.join(genre_agg, on="user_id", how="left")
    
    # Агрегація за типами аніме
    for anime_type in ANIME_TYPES:
        col_name = f"type_count_{anime_type.lower()}"
        type_agg = ratings_with_anime.groupBy("user_id").agg(
            spark_sum(
                when(col("anime_type") == anime_type, 1).otherwise(0)
            ).alias(col_name)
        )
        user_stats = user_stats.join(type_agg, on="user_id", how="left")
    
    # Середній рейтинг за типом
    for anime_type in ANIME_TYPES:
        col_name = f"avg_rating_{anime_type.lower()}"
        type_avg = ratings_with_anime.filter(col("anime_type") == anime_type) \
            .groupBy("user_id") \
            .agg(avg("user_score").alias(col_name))
        user_stats = user_stats.join(type_avg, on="user_id", how="left")
    
    # Додаємо статистику користувача з users_df
    users_stats_cols = users_df.select(
        col("Mal ID").alias("uid"),
        "Gender",
        col("Days Watched").alias("days_watched"),
        col("Mean Score").alias("user_mean_score"),
        col("Completed").alias("user_completed"),
        col("Dropped").alias("user_dropped"),
        col("Plan to Watch").alias("user_plan_to_watch"),
        col("Total Entries").alias("user_total_entries"),
        col("Rewatched").alias("user_rewatched"),
        col("Episodes Watched").alias("user_episodes_watched")
    )
    
    user_features = user_stats.join(
        users_stats_cols,
        user_stats.user_id == users_stats_cols.uid,
        "inner"
    ).drop("uid")
    
    return user_features


def prepare_classification_dataset(
    spark: SparkSession,
    ratings_path: str,
    users_path: str,
    anime_path: str,
    output_path: str,
    min_ratings: int = 10,
    sample_users: Optional[int] = None
) -> DataFrame:
    """
    Підготовлює датасет для класифікації (передбачення Gender).
    
    Args:
        spark: SparkSession
        ratings_path: Шлях до файлу з оцінками користувачів
        users_path: Шлях до даних користувачів
        anime_path: Шлях до даних аніме
        output_path: Шлях для збереження результату
        min_ratings: Мінімальна кількість оцінок для включення користувача
        sample_users: Опціонально - обмежити кількість користувачів для sampling
        
    Returns:
        Підготовлений DataFrame
    """
    logger.info("=" * 60)
    logger.info("ПІДГОТОВКА ДАТАСЕТУ ДЛЯ КЛАСИФІКАЦІЇ")
    logger.info("=" * 60)
    
    # Завантаження даних
    users_df = load_csv_data(spark, users_path)
    anime_df = load_csv_data(spark, anime_path)
    
    # Фільтруємо користувачів з відомою статтю ПЕРЕД завантаженням ratings
    users_with_gender = users_df.filter(
        (col("Gender") == "Male") | (col("Gender") == "Female")
    )
    logger.info(f"Користувачі з відомою статтю: {users_with_gender.count()}")
    
    # Sampling якщо потрібно
    if sample_users:
        users_with_gender = users_with_gender.orderBy(rand()).limit(sample_users)
        logger.info(f"Відібрано {sample_users} користувачів для sampling")
    
    # Отримуємо ID користувачів з відомою статтю
    user_ids = users_with_gender.select(col("Mal ID").alias("uid")).distinct()
    
    # Завантаження ratings
    ratings_df = load_csv_data(spark, ratings_path)
    
    # Фільтруємо ratings тільки для користувачів з відомою статтю
    ratings_df = ratings_df.join(
        user_ids,
        ratings_df.user_id == user_ids.uid,
        "inner"
    ).drop("uid")
    
    logger.info(f"Завантажено ratings (після фільтрації): {ratings_df.count()}")
    logger.info(f"Завантажено anime: {anime_df.count()}")
    
    # Агрегуємо features (rating - колонка з users-score-2023.csv)
    user_features = aggregate_user_features(
        spark, ratings_df, users_with_gender, anime_df,
        rating_column="rating"
    )
    
    # Фільтруємо за мінімальною кількістю оцінок
    user_features = user_features.filter(col("total_ratings") >= min_ratings)
    logger.info(f"Після фільтрації (>= {min_ratings} оцінок): {user_features.count()}")
    
    # Створюємо додаткові derived features
    
    # Genre diversity score (кількість різних жанрів)
    genre_count_cols = [
        f"genre_count_{g.lower().replace(' ', '_').replace('-', '_')}" 
        for g in TOP_GENRES
    ]
    existing_genre_cols = [c for c in genre_count_cols if c in user_features.columns]
    
    if existing_genre_cols:
        # Підраховуємо скільки жанрів має хоча б 1 перегляд
        user_features = user_features.withColumn(
            "genre_diversity",
            sum([when(col(c) > 0, 1).otherwise(0) for c in existing_genre_cols])
        )
    
    # Rating consistency (стандартне відхилення / mean)
    user_features = user_features.withColumn(
        "rating_consistency",
        when(
            (col("avg_rating").isNotNull()) & (col("avg_rating") > 0),
            coalesce(col("rating_std"), lit(0)) / col("avg_rating")
        ).otherwise(lit(0))
    )
    
    # Preference intensity (схильність до крайніх оцінок)
    user_features = user_features.withColumn(
        "preference_intensity",
        (col("max_rating") - col("min_rating")) / lit(9.0)  # Нормалізовано на шкалу 1-10
    )
    
    # Drop rate (відсоток dropped)
    user_features = user_features.withColumn(
        "drop_rate",
        when(
            col("user_total_entries") > 0,
            col("user_dropped") / col("user_total_entries")
        ).otherwise(lit(0))
    )
    
    # Encoding Gender як числове значення (для класифікації)
    user_features = user_features.withColumn(
        "gender_encoded",
        when(col("Gender") == "Male", 0).otherwise(1)
    )
    
    # Перетворюємо genre counts на відсотки
    for col_name in existing_genre_cols:
        pct_col = col_name.replace("_count_", "_pct_")
        user_features = user_features.withColumn(
            pct_col,
            when(
                col("total_ratings") > 0,
                col(col_name) / col("total_ratings")
            ).otherwise(lit(0))
        )
    
    # Нормалізація числових features
    numeric_to_normalize = [
        "total_ratings", "avg_rating", "days_watched", 
        "user_mean_score", "user_completed", "user_episodes_watched"
    ]
    
    for col_name in numeric_to_normalize:
        if col_name in user_features.columns:
            stats = user_features.filter(col(col_name).isNotNull()).agg(
                spark_min(col_name).alias("min_val"),
                spark_max(col_name).alias("max_val")
            ).collect()[0]
            min_val, max_val = stats["min_val"], stats["max_val"]
            if max_val and min_val and max_val != min_val:
                user_features = user_features.withColumn(
                    f"{col_name}_normalized",
                    (col(col_name) - lit(min_val)) / lit(max_val - min_val)
                )
    
    # Заповнення null значень
    user_features = user_features.fillna(0)
    
    # Вибір фінальних колонок
    final_columns = ["user_id", "Gender", "gender_encoded"]
    
    # Додаємо всі feature колонки
    feature_columns = [c for c in user_features.columns 
                       if c not in ["user_id", "Gender", "gender_encoded", "uid"]]
    final_columns.extend(feature_columns)
    
    existing_final = [c for c in final_columns if c in user_features.columns]
    df_final = user_features.select(existing_final)
    
    final_count = df_final.count()
    logger.info(f"Фінальний датасет: {final_count} записів")
    
    # Підрахунок балансу класів
    class_counts = df_final.groupBy("Gender").count().collect()
    class_balance = {row["Gender"]: row["count"] for row in class_counts}
    logger.info(f"Баланс класів: {class_balance}")
    
    # Збереження metadata
    features = [c for c in existing_final if c not in ["user_id", "Gender", "gender_encoded"]]
    metadata = {
        "task": "classification",
        "target": "gender_encoded",
        "target_labels": {"Male": 0, "Female": 1},
        "feature_count": len(features),
        "features": features,
        "total_samples": final_count,
        "class_balance": class_balance,
        "min_ratings_threshold": min_ratings,
        "genres_used": TOP_GENRES,
        "types_used": ANIME_TYPES
    }
    save_metadata(metadata, output_path, "preprocessing_info.json")
    
    return df_final


# ============================================================================
# SPLIT ТА ЗБЕРЕЖЕННЯ
# ============================================================================

def split_and_save(
    df: DataFrame,
    output_path: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    stratify_column: Optional[str] = None,
    format_type: str = "both"
) -> Tuple[DataFrame, DataFrame, DataFrame]:
    """
    Розділяє датасет на train/validation/test та зберігає.
    
    Args:
        df: DataFrame для розділення
        output_path: Шлях для збереження
        train_ratio: Частка для train
        val_ratio: Частка для validation
        test_ratio: Частка для test
        stratify_column: Колонка для stratified split (опціонально)
        format_type: Формат збереження ("csv", "parquet", "both")
        
    Returns:
        Tuple з (train_df, val_df, test_df)
    """
    logger.info("Розділення датасету на train/validation/test...")
    
    os.makedirs(output_path, exist_ok=True)
    
    if stratify_column and stratify_column in df.columns:
        # Stratified split
        logger.info(f"Використовується stratified split за колонкою: {stratify_column}")
        
        # Додаємо random колонку
        df = df.withColumn("_rand", rand())
        
        # Window для stratified sampling
        window = Window.partitionBy(stratify_column).orderBy("_rand")
        df = df.withColumn("_row_num", row_number().over(window))
        
        # Підраховуємо кількість в кожному класі
        class_counts = df.groupBy(stratify_column).count()
        
        train_dfs = []
        val_dfs = []
        test_dfs = []
        
        for row in class_counts.collect():
            class_val = row[stratify_column]
            class_count = row["count"]
            
            train_end = int(class_count * train_ratio)
            val_end = int(class_count * (train_ratio + val_ratio))
            
            class_df = df.filter(col(stratify_column) == class_val)
            
            train_dfs.append(class_df.filter(col("_row_num") <= train_end))
            val_dfs.append(class_df.filter(
                (col("_row_num") > train_end) & (col("_row_num") <= val_end)
            ))
            test_dfs.append(class_df.filter(col("_row_num") > val_end))
        
        # Об'єднуємо
        train_df = train_dfs[0]
        val_df = val_dfs[0]
        test_df = test_dfs[0]
        
        for i in range(1, len(train_dfs)):
            train_df = train_df.union(train_dfs[i])
            val_df = val_df.union(val_dfs[i])
            test_df = test_df.union(test_dfs[i])
        
        # Видаляємо службові колонки
        train_df = train_df.drop("_rand", "_row_num")
        val_df = val_df.drop("_rand", "_row_num")
        test_df = test_df.drop("_rand", "_row_num")
        
    else:
        # Random split
        logger.info("Використовується random split")
        splits = df.randomSplit([train_ratio, val_ratio, test_ratio], seed=42)
        train_df, val_df, test_df = splits
    
    # Логування розмірів
    train_count = train_df.count()
    val_count = val_df.count()
    test_count = test_df.count()
    
    logger.info(f"Train: {train_count} ({train_count/(train_count+val_count+test_count)*100:.1f}%)")
    logger.info(f"Validation: {val_count} ({val_count/(train_count+val_count+test_count)*100:.1f}%)")
    logger.info(f"Test: {test_count} ({test_count/(train_count+val_count+test_count)*100:.1f}%)")
    
    # Збереження
    save_dataset(train_df, os.path.join(output_path, "train"), format_type)
    save_dataset(val_df, os.path.join(output_path, "validation"), format_type)
    save_dataset(test_df, os.path.join(output_path, "test"), format_type)
    
    # Збереження інформації про split
    split_info = {
        "train_size": train_count,
        "validation_size": val_count,
        "test_size": test_count,
        "train_ratio": train_ratio,
        "validation_ratio": val_ratio,
        "test_ratio": test_ratio,
        "stratified": stratify_column is not None,
        "stratify_column": stratify_column
    }
    save_metadata(split_info, output_path, "split_info.json")
    
    return train_df, val_df, test_df


# ============================================================================
# ГОЛОВНА ФУНКЦІЯ
# ============================================================================

def main():
    """
    Головна функція для підготовки обох датасетів.
    """
    logger.info("=" * 60)
    logger.info("ЗАПУСК ПІДГОТОВКИ ML ДАТАСЕТІВ")
    logger.info("=" * 60)
    
    # Шляхи до даних
    data_path = "data"
    output_base = os.path.join(data_path, "ml_datasets")
    
    # Ініціалізація Spark
    spark = get_spark_session()
    
    try:
        # ================================================================
        # 1. РЕГРЕСІЯ
        # ================================================================
        regression_output = os.path.join(output_base, "regression")
        os.makedirs(regression_output, exist_ok=True)
        
        regression_df = prepare_regression_dataset(
            spark,
            anime_path=os.path.join(data_path, "anime-filtered.csv"),
            output_path=regression_output
        )
        
        # Split та збереження
        split_and_save(
            regression_df,
            regression_output,
            stratify_column=None,  # Для регресії без stratification
            format_type="both"
        )
        
        logger.info("Датасет для регресії підготовлено!")
        
        # ================================================================
        # 2. КЛАСИФІКАЦІЯ
        # ================================================================
        classification_output = os.path.join(output_base, "classification")
        os.makedirs(classification_output, exist_ok=True)
        
        classification_df = prepare_classification_dataset(
            spark,
            ratings_path=os.path.join(data_path, "users-score-2023.csv"),
            users_path=os.path.join(data_path, "users-details-2023.csv"),
            anime_path=os.path.join(data_path, "anime-filtered.csv"),
            output_path=classification_output,
            min_ratings=10,
            sample_users=50000  # Обмежуємо для швидшої обробки
        )
        
        # Split та збереження з stratification
        split_and_save(
            classification_df,
            classification_output,
            stratify_column="gender_encoded",
            format_type="both"
        )
        
        logger.info("Датасет для класифікації підготовлено!")
        
        # ================================================================
        # ФІНАЛЬНИЙ SUMMARY
        # ================================================================
        logger.info("=" * 60)
        logger.info("ПІДГОТОВКА ЗАВЕРШЕНА!")
        logger.info("=" * 60)
        logger.info(f"Регресія: {regression_output}")
        logger.info(f"Класифікація: {classification_output}")
        
    except Exception as e:
        logger.error(f"Помилка: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

