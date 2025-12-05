import os
import time
import warnings
from typing import Optional, Tuple, Any, List
import joblib

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report
)

from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)


# ============================================================
# Завантаження train/test
# ============================================================
def _load_split(base_path: str, split_name: str = "train", prefer_parquet: bool = True) -> Optional[pd.DataFrame]:
    parquet_path = os.path.join(base_path, f"{split_name}.parquet")
    csv_path = os.path.join(base_path, f"{split_name}.csv")

    if prefer_parquet and os.path.exists(parquet_path):
        return pd.read_parquet(parquet_path)
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    if os.path.exists(parquet_path):
        return pd.read_parquet(parquet_path)

    return None


# ============================================================
# Три моделі: базова → середня → найкраща
# ============================================================
def get_models() -> List[Tuple[str, Any]]:
    return [
        # 1. БАЗОВА МОДЕЛЬ
        ("Logistic Regression", LogisticRegression(
            max_iter=2000,
            n_jobs=-1
        )),

        # 2. СЕРЕДНЯ МОДЕЛЬ
        ("Random Forest (n=200)", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )),

        # 3. НАЙКРАЩА МОДЕЛЬ (бустинг)
        ("Gradient Boosting", GradientBoostingClassifier(
            random_state=42
        )),
    ]


# ============================================================
# Оцінювання моделей
# ============================================================
def evaluate_models(X_train, y_train, X_test, y_test, models_dir: str) -> pd.DataFrame:
    os.makedirs(models_dir, exist_ok=True)

    results = []
    models = get_models()

    print("\nПочинаємо оцінювання моделей...")
    print("-" * 80)
    print(f"{'Model':<30} | {'Acc':<8} | {'F1':<8} | {'Time (s)':<8} | Status")
    print("-" * 80)

    for name, model in models:
        start = time.time()

        safe_name = (
            name.replace(" ", "_")
                .replace("(", "")
                .replace(")", "")
                .replace("=", "_")
                .replace(",", "")
        )

        model_path = os.path.join(models_dir, f"{safe_name}.joblib")
        status = "Trained"

        try:
            # Якщо вже є — завантажуємо
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                status = "Loaded"
            else:
                model.fit(X_train, y_train)
                joblib.dump(model, model_path)

            # Прогноз
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            print(f"{name:<30} | {acc:<8.4f} | {f1:<8.4f} | {time.time() - start:<8.2f} | {status}")

            results.append({
                "Model": name,
                "Accuracy": acc,
                "F1": f1,
                "Status": status
            })

        except Exception as e:
            print(f"{name:<30} | ERROR: {str(e)}")

    print("-" * 80)
    return pd.DataFrame(results).sort_values("F1", ascending=False)


# ============================================================
# Головна функція
# ============================================================
def run_classification_modeling(data_path: str = "data") -> None:
    base_path = os.path.join(data_path, "ml_datasets", "classification")
    models_path = os.path.join(data_path, "models", "classification")

    print("\n" + "=" * 60)
    print("🧪 ML КЛАСИФІКАЦІЯ: ТРЕНУВАННЯ")
    print("=" * 60)

    df_train = _load_split(base_path, "train")
    df_test = _load_split(base_path, "test")

    if df_train is None or df_test is None:
        print("❌ Не знайдено датасетів train/test.")
        return

    target_col = "gender_encoded"
    ignore = ["user_id", "Gender", target_col]
    feature_cols = [c for c in df_train.columns if c not in ignore]

    X_train = df_train[feature_cols]
    y_train = df_train[target_col]

    X_test = df_test[feature_cols]
    y_test = df_test[target_col]

    # Чистимо пропущені значення
    if X_train.isnull().sum().sum() > 0:
        X_train = X_train.fillna(0)
        X_test = X_test.fillna(0)

    results = evaluate_models(X_train, y_train, X_test, y_test, models_path)

    print("\n🏆 Найкращі моделі:")
    print(results.head().to_string(index=False))

    # Детальний звіт найкращої моделі
    print("\n📊 Детальний classification report для найкращої моделі:")
    best_model_name = results.iloc[0]["Model"]
    best_model = joblib.load(os.path.join(
        models_path,
        best_model_name.replace(" ", "_").replace("(", "").replace(")", "") + ".joblib"
    ))

    y_pred = best_model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=["Male", "Female"]))
