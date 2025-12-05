"""
Модуль для тренування та оцінки регресійних моделей.
"""

import os
import time
import warnings
from typing import Optional, Dict, List, Tuple, Any
import joblib

import pandas as pd
import numpy as np

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    BaggingRegressor,
)
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.exceptions import ConvergenceWarning

# Ігноруємо попередження про конвергенцію для чистих результатів
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def _load_split(
    base_path: str, split_name: str = "train", prefer_parquet: bool = True
) -> Optional[pd.DataFrame]:
    """
    Завантажує один з датасетів (train / validation / test).
    """
    parquet_path = os.path.join(base_path, f"{split_name}.parquet")
    csv_path = os.path.join(base_path, f"{split_name}.csv")

    if prefer_parquet and os.path.exists(parquet_path):
        return pd.read_parquet(parquet_path)
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    
    # Fallback for parquet if preferred but checked CSV second
    if os.path.exists(parquet_path):
        return pd.read_parquet(parquet_path)

    return None


def get_models() -> List[Tuple[str, Any]]:
    """
    Повертає список з ТОП-3 моделей для регресії.
    Відібрано за результатами експериментів (найкращі RMSE/R2).
    """
    models = [
        # 1. Найкраща модель (RMSE ~0.448, R2 ~0.696)
        ("Random Forest (n=100)", RandomForestRegressor(
            n_estimators=100, 
            random_state=42, 
            n_jobs=-1
        )),
        
        # 2. Друга найкраща модель (RMSE ~0.459, R2 ~0.680)
        ("Extra Trees", ExtraTreesRegressor(
            n_estimators=100, 
            random_state=42, 
            n_jobs=-1
        )),

        # 3. Третя найкраща модель (RMSE ~0.469, R2 ~0.666)
        ("Bagging Regressor", BaggingRegressor(
            random_state=42, 
            n_jobs=-1
        )),
    ]
    return models


def evaluate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    models_dir: str
) -> pd.DataFrame:
    """
    Тренує моделі (або завантажує існуючі) та повертає DataFrame з метриками.
    """
    results = []
    models = get_models()
    
    os.makedirs(models_dir, exist_ok=True)

    print(f"\nПочаток оцінки {len(models)} моделей...")
    print(f"Папка для моделей: {models_dir}")
    print("-" * 90)
    print(f"{'Model Name':<30} | {'RMSE':<10} | {'MAE':<10} | {'R2':<10} | {'Time (s)':<10} | {'Status':<10}")
    print("-" * 90)

    for name, model in models:
        start_time = time.time()
        # Створюємо безпечне ім'я файлу
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "_").replace(",", "")
        model_path = os.path.join(models_dir, f"{safe_name}.joblib")
        
        status = "Trained"
        
        try:
            # Check if model exists
            if os.path.exists(model_path):
                # Load model
                model = joblib.load(model_path)
                status = "Loaded"
            else:
                # Train model
                model.fit(X_train, y_train)
                # Save model
                joblib.dump(model, model_path)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            duration = time.time() - start_time
            
            print(f"{name:<30} | {rmse:<10.4f} | {mae:<10.4f} | {r2:<10.4f} | {duration:<10.2f} | {status:<10}")
            
            results.append({
                "Model": name,
                "RMSE": rmse,
                "MAE": mae,
                "R2": r2,
                "Time": duration,
                "Status": status
            })
            
        except Exception as e:
            print(f"{name:<30} | ERROR: {str(e)}")

    print("-" * 90)
    return pd.DataFrame(results).sort_values(by="RMSE", ascending=True)


def run_regression_modeling(data_path: str = "data") -> None:
    """
    Запускає процес тренування та оцінки регресійних моделей.
    """
    base_path = os.path.join(data_path, "ml_datasets", "regression")
    models_path = os.path.join(data_path, "models", "regression")
    
    print("\n" + "=" * 60)
    print("🧪 ML РЕГРЕСІЯ: ЕКСПЕРИМЕНТ")
    print("=" * 60)

    # 1. Load Data
    print("Завантаження даних...")
    df_train = _load_split(base_path, "train")
    df_test = _load_split(base_path, "test") # Використовуємо test для фінальної оцінки в цьому експерименті
    
    if df_train is None or df_test is None:
        print("❌ Помилка: Не знайдено train або test датасети.")
        return

    # 2. Prepare Features and Target
    target_col = "Score"
    ignore_cols = ["anime_id", "Name", "s_score", target_col]
    
    # Фільтруємо колонки, які є в датасеті
    feature_cols = [c for c in df_train.columns if c not in ignore_cols]
    
    X_train = df_train[feature_cols]
    y_train = df_train[target_col]
    
    X_test = df_test[feature_cols]
    y_test = df_test[target_col]
    
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape:  {X_test.shape}")
    print(f"Features:    {len(feature_cols)}")
    
    # Handling NaNs if any (MLP and some others don't like NaNs)
    if X_train.isnull().sum().sum() > 0:
        print("⚠️ Увага: Знайдено пропущені значення. Заповнюємо 0...")
        X_train = X_train.fillna(0)
        X_test = X_test.fillna(0)

    # 3. Run Evaluation
    results_df = evaluate_models(X_train, y_train, X_test, y_test, models_dir=models_path)
    
    # 4. Show Final Leaderboard
    print("\n🏆 ТОП-3 МОДЕЛЕЙ (за RMSE):")
    print(results_df.head(3).to_string(index=False))
    
    # 5. Save results (optional)
    results_path = os.path.join(data_path, "results", "regression_leaderboard.csv")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    results_df.to_csv(results_path, index=False)
    print(f"\n📄 Повні результати збережено в: {results_path}")
