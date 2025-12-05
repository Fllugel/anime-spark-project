"""
Підготовка стабільного датасету для класифікації статі користувача.
Версія: покращена, fault-tolerant, оптимізована.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from typing import List, Optional, Dict

from sklearn.model_selection import train_test_split

# ----------------------------------------------------------
# ЛОГІНГ
# ----------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("classification-prep")


# ----------------------------------------------------------
# КОНСТАНТИ
# ----------------------------------------------------------
TOP_GENRES = [
    "Action", "Adventure", "Comedy", "Drama", "Fantasy", "Romance",
    "Sci-Fi", "Slice of Life", "Sports", "Supernatural", "Mystery",
    "Horror", "Psychological", "Thriller", "Mecha", "Music", "School",
    "Shounen", "Shoujo", "Seinen", "Josei", "Ecchi", "Harem"
]

ANIME_TYPES = ["TV", "Movie", "OVA", "ONA", "Special"]


# ----------------------------------------------------------
# ДОПОМІЖНІ ФУНКЦІЇ
# ----------------------------------------------------------

def autodetect_column(df: pd.DataFrame, candidates: List[str]) -> str:
    """Автоматично знаходить назву колонки серед можливих варіантів."""
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    raise ValueError(f"Не знайдено жодної з колонок: {candidates}")


def normalize(series: pd.Series) -> pd.Series:
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(np.zeros(len(series)))
    return (series - mn) / (mx - mn)


def save_metadata(path: str, meta: Dict):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "preprocessing_info.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


# ----------------------------------------------------------
# ЗАВАНТАЖЕННЯ USERS
# ----------------------------------------------------------

def load_users(path: str, sample: int = 20000) -> pd.DataFrame:
    log.info(f"Завантаження users: {path}")
    df = pd.read_csv(path)

    # Автовизначення колонок
    gender_col = autodetect_column(df, ["Gender"])
    id_col = autodetect_column(df, ["Mal ID", "user_id", "uid", "id"])

    df = df[df[gender_col].isin(["Male", "Female"])]
    log.info(f"Users з відомою статтю: {len(df)}")

    if len(df) > sample:
        df = df.sample(sample, random_state=42)

    df = df.rename(columns={id_col: "user_id", gender_col: "Gender"})
    return df


# ----------------------------------------------------------
# ЗАВАНТАЖЕННЯ RATINGS ЧАНКАМИ
# ----------------------------------------------------------

def load_ratings(path: str, user_ids: set) -> pd.DataFrame:
    log.info(f"Завантаження ratings: {path}")

    chunks = []
    rows_total = 0
    rows_kept = 0

    for chunk in pd.read_csv(path, chunksize=400_000):
        rows_total += len(chunk)

        # Автовизначення колонок (1 раз)
        if "user_id" not in chunk.columns:
            id_col = autodetect_column(chunk, ["user_id", "UID", "Mal ID"])
            chunk = chunk.rename(columns={id_col: "user_id"})

        if "anime_id" not in chunk.columns:
            anime_col = autodetect_column(chunk, ["anime_id", "Anime ID", "id"])
            chunk = chunk.rename(columns={anime_col: "anime_id"})

        part = chunk[chunk["user_id"].isin(user_ids)]
        rows_kept += len(part)

        if len(part) > 0:
            chunks.append(part)

        if rows_total % 2_000_000 < 400_000:
            log.info(f"  Оброблено {rows_total:,}, відібрано {rows_kept:,}")

    if len(chunks) == 0:
        raise ValueError("ratings_df пустий — жодного рейтингу для цих users")

    df = pd.concat(chunks, ignore_index=True)
    log.info(f"Завантажено ratings: {len(df):,}")
    return df


# ----------------------------------------------------------
# ФІЧІ КОРИСТУВАЧІВ
# ----------------------------------------------------------

def build_features(ratings: pd.DataFrame, users: pd.DataFrame, anime: pd.DataFrame) -> pd.DataFrame:
    log.info("Формування фіч...")

    # Автовизначення колонок
    anime_id_col = autodetect_column(anime, ["anime_id", "id"])
    genre_col = autodetect_column(anime, ["Genres", "genres"])
    type_col = autodetect_column(anime, ["Type", "type"])
    score_col = autodetect_column(anime, ["Score"])

    anime = anime.rename(columns={
        anime_id_col: "anime_id",
        genre_col: "Genres",
        type_col: "Type",
        score_col: "AnimeScore"
    })

    ratings = ratings.merge(
        anime[["anime_id", "Genres", "Type", "AnimeScore"]],
        on="anime_id",
        how="left"
    )

    # Базова статистика
    stats = ratings.groupby("user_id").agg(
        total_ratings=("rating", "count"),
        avg_rating=("rating", "mean"),
        std_rating=("rating", "std"),
        min_rating=("rating", "min"),
        max_rating=("rating", "max"),
        unique_anime=("anime_id", "nunique")
    ).reset_index()

    # Жанри
    for g in TOP_GENRES:
        col = f"genre_{g.lower().replace(' ', '_')}"
        ratings[col] = ratings["Genres"].str.contains(g, case=False, na=False).astype(int)

    genre_cols = [c for c in ratings.columns if c.startswith("genre_")]
    genre_sum = ratings.groupby("user_id")[genre_cols].sum().reset_index()

    stats = stats.merge(genre_sum, on="user_id", how="left")

    # Типи аніме
    for t in ANIME_TYPES:
        col = f"type_{t.lower()}"
        ratings[col] = (ratings["Type"] == t).astype(int)

    type_cols = [c for c in ratings.columns if c.startswith("type_")]
    type_sum = ratings.groupby("user_id")[type_cols].sum().reset_index()
    stats = stats.merge(type_sum, on="user_id", how="left")

    # Демографічні дані користувача
    demo_cols = ["Days Watched", "Mean Score", "Completed", "Dropped",
                 "Plan to Watch", "Total Entries", "Rewatched", "Episodes Watched"]

    for c in demo_cols:
        if c not in users.columns:
            users[c] = 0

    users = users.rename(columns={
        c: c.lower().replace(" ", "_") for c in demo_cols
    })

    stats = stats.merge(users[["user_id", "Gender"] + [c.lower().replace(" ", "_") for c in demo_cols]],
                        on="user_id", how="left")

    # Derived features
    stats["std_rating"] = stats["std_rating"].fillna(0)
    stats["genre_diversity"] = (stats[genre_cols] > 0).sum(axis=1)
    stats["rating_stability"] = stats["std_rating"] / stats["avg_rating"].replace(0, 1)
    stats["preference_intensity"] = (stats["max_rating"] - stats["min_rating"]) / 9
    stats["gender_encoded"] = (stats["Gender"] == "Female").astype(int)

    # Нормалізація
    for c in ["total_ratings", "avg_rating", "unique_anime",
              "days_watched", "mean_score", "completed", "episodes_watched"]:
        if c in stats.columns:
            stats[c + "_norm"] = normalize(stats[c].fillna(0))

    stats = stats.fillna(0)
    return stats


# ----------------------------------------------------------
# SPLIT + SAVE
# ----------------------------------------------------------

def split_and_save(df: pd.DataFrame, output_path: str):
    os.makedirs(output_path, exist_ok=True)

    # Перевірка балансів
    cnt = df["gender_encoded"].value_counts().to_dict()
    log.info(f"Баланс статей: {cnt}")

    # Якщо мало жінок → stratify OFF
    stratify_col = df["gender_encoded"] if min(cnt.values()) >= 2 else None

    train, temp = train_test_split(
        df, test_size=0.30, random_state=42, stratify=stratify_col
    )
    val, test = train_test_split(
        temp, test_size=0.50, random_state=42,
        stratify=temp["gender_encoded"] if stratify_col is not None else None
    )

    train.to_parquet(os.path.join(output_path, "train.parquet"), index=False)
    val.to_parquet(os.path.join(output_path, "validation.parquet"), index=False)
    test.to_parquet(os.path.join(output_path, "test.parquet"), index=False)

    log.info(f"Збережено train/val/test у {output_path}")


# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------

def main():
    data_dir = "data"
    out_dir = os.path.join(data_dir, "ml_datasets", "classification")

    users = load_users(os.path.join(data_dir, "users-details-2023.csv"))
    ratings = load_ratings(os.path.join(data_dir, "users-score-2023.csv"),
                           set(users["user_id"].tolist()))
    anime = pd.read_csv(os.path.join(data_dir, "anime-filtered.csv"))

    features = build_features(ratings, users, anime)

    # Фільтрація мінімального числа оцінок
    features = features[features["total_ratings"] >= 10]

    split_and_save(features, out_dir)

    save_metadata(out_dir, {
        "task": "classification",
        "target": "gender_encoded",
        "num_samples": len(features),
        "num_features": len(features.columns)
    })

    log.info("ГОТОВО — датасет класифікації створено!")


if __name__ == "__main__":
    main()
