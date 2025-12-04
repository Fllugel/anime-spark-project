"""
Заглушка для стадії ML-класифікації.

Працює з уже підготовленими датасетами у `data/ml_datasets/classification`.
У майбутньому тут можна реалізувати повний пайплайн тренування / оцінки моделей.
"""

import os
from typing import Optional

import pandas as pd


def _load_split(
    base_path: str, split_name: str = "train", prefer_parquet: bool = True
) -> Optional[pd.DataFrame]:
    """
    Завантажує один з датасетів (train / validation / test), якщо він існує.
    """
    parquet_path = os.path.join(base_path, f"{split_name}.parquet")
    csv_path = os.path.join(base_path, f"{split_name}.csv")

    if prefer_parquet and os.path.exists(parquet_path):
        return pd.read_parquet(parquet_path)
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    if os.path.exists(parquet_path):
        return pd.read_parquet(parquet_path)

    return None


def run_classification_modeling(data_path: str = "data") -> None:
    """
    Заглушка для класифікаційного ML-пайплайну.

    На даному етапі:
    - перевіряє наявність `data/ml_datasets/classification`
    - завантажує train split
    - виводить базову інформацію про датасет
    """
    base_path = os.path.join(data_path, "ml_datasets", "classification")

    if not os.path.exists(base_path):
        print(
            "⚠️  Датасет класифікації не знайдено. "
            "Спочатку підготуйте його через меню (варіант 2)."
        )
        return

    df_train = _load_split(base_path, "train")
    if df_train is None:
        print(
            "⚠️  Не вдалося знайти train-спліт для класифікаційного датасету. "
            "Очікуються файли train.parquet або train.csv."
        )
        return

    print("\n" + "=" * 60)
    print("🧪 ML КЛАСИФІКАЦІЯ (заглушка)")
    print("=" * 60)
    print(f"Форма train-датасету: {df_train.shape}")

    target_col = "gender_encoded"
    feature_cols = [c for c in df_train.columns if c not in ["user_id", "Gender", target_col]]

    print(f"Кількість features: {len(feature_cols)}")
    print(f"Target колонка: {target_col}")

    class_counts = df_train[target_col].value_counts(normalize=True) * 100
    print("\nРозподіл цільового класу (у %):")
    for cls, pct in class_counts.sort_index().items():
        label = "Female" if cls == 1 else "Male"
        print(f"  {cls} ({label}): {pct:.2f}%")

    print(
        "\nНа цьому етапі модель ще не тренується. "
        "Тут можна додати пайплайн sklearn / PySpark ML у майбутньому."
    )


