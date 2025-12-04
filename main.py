import os
from typing import Tuple

from transformation.data_extraction import (
    create_star_schema,
    save_star_schema_to_parquet,
    load_star_schema_from_parquet,
)
from transformation.business_questions import (
    run_artem_questions,
    run_bohdan_questions,
    run_oskar_questions,
    run_arii_extended_questions,
)
from transformation.dataset_info import run_dataset_info_analysis
from transformation.numeric_statistics import run_numeric_statistics_analysis
from transformation.raw_data_extraction import run_raw_data_extraction
from data_analysis.prepare_ml_datasets import (
    prepare_regression_dataset,
    prepare_classification_dataset,
    split_and_save,
)
from data_analysis.classification_modeling import run_classification_modeling
from data_analysis.regression_modeling import run_regression_modeling

# Перевірка доступності PySpark
try:
    from pyspark.sql import SparkSession, DataFrame  # type: ignore

    SPARK_AVAILABLE = True
except ImportError:
    SparkSession = None  # type: ignore
    DataFrame = None  # type: ignore
    SPARK_AVAILABLE = False


DATA_PATH = "data"


def create_spark_session() -> "SparkSession":
    """
    Створює SparkSession з оптимальними налаштуваннями для локального запуску.
    """
    assert SPARK_AVAILABLE, "PySpark недоступний"
    spark = (
        SparkSession.builder.appName("AnimeSparkApp")
        .config(
            "spark.driver.extraJavaOptions",
            "--add-opens=java.base/java.nio=ALL-UNNAMED "
            "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
        )
        .config(
            "spark.executor.extraJavaOptions",
            "--add-opens=java.base/java.nio=ALL-UNNAMED "
            "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
        )
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
        .config("spark.sql.parquet.datetimeRebaseModeInWrite", "LEGACY")
        .config("spark.driver.memory", "2g")
        .config("spark.executor.memory", "2g")
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", "1000")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")
    return spark


# ======================================================================
# Перевірки та підготовка стадій пайплайну
# ======================================================================


def ensure_raw_data(data_path: str = DATA_PATH) -> None:
    """
    Перевіряє наявність сирих CSV файлів.
    Якщо якихось не вистачає – запускає заглушку raw data extraction.
    """
    required_files = [
        "anime-dataset-2023.csv",
        "users-details-2023.csv",
        "users-score-2023.csv",
    ]
    missing = [
        f for f in required_files if not os.path.exists(os.path.join(data_path, f))
    ]

    if not missing:
        print("✅ Сирі дані вже присутні (CSV файли знайдені).")
        return

    print("⚠️ Не вистачає сирих даних:")
    for name in missing:
        print(f"   - {name}")
    print("▶️ Запускаю заглушку стадії 'data extraction'...")
    run_raw_data_extraction(data_path=data_path, missing_files=missing)


def _star_schema_parquet_exists(data_path: str = DATA_PATH) -> bool:
    base = os.path.join(data_path, "star_schema")
    required_dirs = [
        "dim_user",
        "dim_anime",
        "dim_date",
        "fact_user_ratings",
    ]
    return all(os.path.exists(os.path.join(base, d)) for d in required_dirs)


def ensure_star_schema(
    spark: "SparkSession", data_path: str = DATA_PATH
) -> Tuple["DataFrame", "DataFrame", "DataFrame", "DataFrame"]:
    """
    Гарантує наявність зірчастої схеми:
    - якщо є Parquet – завантажує
    - інакше створює з нуля та зберігає в Parquet
    """
    parquet_path = os.path.join(data_path, "star_schema")

    if _star_schema_parquet_exists(data_path):
        print("\n📂 Зірчаста схема вже існує в Parquet – завантажую...")
        return load_star_schema_from_parquet(spark, parquet_path=parquet_path)

    print("\n🌟 Зірчаста схема ще не створена – створюю з нуля...")
    dim_user, dim_anime, dim_date, fact_ratings = create_star_schema(
        spark, data_path=data_path
    )
    save_star_schema_to_parquet(
        dim_user, dim_anime, dim_date, fact_ratings, output_path=parquet_path
    )
    return dim_user, dim_anime, dim_date, fact_ratings


def _ml_dataset_meta_path(kind: str, data_path: str = DATA_PATH) -> str:
    return os.path.join(data_path, "ml_datasets", kind, "preprocessing_info.json")


def ensure_regression_dataset(spark: "SparkSession", data_path: str = DATA_PATH) -> None:
    """
    Гарантує наявність ML датасету для регресії.
    Використовує PySpark-пайплайн з data_analysis.prepare_ml_datasets.
    """
    meta_path = _ml_dataset_meta_path("regression", data_path)
    output_dir = os.path.dirname(meta_path)

    if os.path.exists(meta_path):
        print("✅ ML датасет регресії вже підготовлений.")
        return

    print("\n📦 Підготовка ML датасету для регресії...")
    os.makedirs(output_dir, exist_ok=True)

    regression_df = prepare_regression_dataset(
        spark,
        anime_path=os.path.join(data_path, "anime-filtered.csv"),
        output_path=output_dir,
    )

    split_and_save(
        regression_df,
        output_dir,
        stratify_column=None,
        format_type="both",
    )
    print("✅ ML датасет для регресії підготовлено.")


def ensure_classification_dataset(
    spark: "SparkSession", data_path: str = DATA_PATH
) -> None:
    """
    Гарантує наявність ML датасету для класифікації.
    Використовує PySpark-пайплайн з data_analysis.prepare_ml_datasets.
    """
    meta_path = _ml_dataset_meta_path("classification", data_path)
    output_dir = os.path.dirname(meta_path)

    if os.path.exists(meta_path):
        print("✅ ML датасет класифікації вже підготовлений.")
        return

    print("\n📦 Підготовка ML датасету для класифікації...")
    os.makedirs(output_dir, exist_ok=True)

    classification_df = prepare_classification_dataset(
        spark,
        ratings_path=os.path.join(data_path, "users-score-2023.csv"),
        users_path=os.path.join(data_path, "users-details-2023.csv"),
        anime_path=os.path.join(data_path, "anime-filtered.csv"),
        output_path=output_dir,
        min_ratings=10,
        sample_users=50000,
    )

    split_and_save(
        classification_df,
        output_dir,
        stratify_column="gender_encoded",
        format_type="both",
    )
    print("✅ ML датасет для класифікації підготовлено.")


# ======================================================================
# Сценарії / стадії, які можна запускати з меню
# ======================================================================


def run_business_questions_flow(spark: "SparkSession", data_path: str = DATA_PATH) -> None:
    """
    Повний пайплайн для бізнес-питань:
    - перевірка сирих даних
    - створення / завантаження зірчастої схеми
    - базовий аналіз датасету (dataset_info + numeric_statistics)
    - запуск усіх бізнес-питань (Artem, Bohdan, Oskar)
    """
    ensure_raw_data(data_path)
    dim_user, dim_anime, dim_date, fact_ratings = ensure_star_schema(
        spark, data_path=data_path
    )

    print("\n" + "=" * 60)
    print("🔄 ТРАНСФОРМАЦІЯ ТА БАЗОВИЙ АНАЛІЗ ДАНИХ")
    print("=" * 60)

    anime_dataset_path = os.path.join(data_path, "anime-dataset-2023.csv")
    df_anime_original = spark.read.csv(anime_dataset_path, header=True, inferSchema=True)

    run_dataset_info_analysis(
        df_anime_original,
        output_dir=os.path.join(data_path, "results"),
    )

    run_numeric_statistics_analysis(
        df_anime_original,
        output_dir=os.path.join(data_path, "results"),
    )

    print("\n" + "=" * 60)
    print("❓ БІЗНЕС-ПИТАННЯ")
    print("=" * 60)

    results_path = os.path.join(data_path, "results")

    run_artem_questions(fact_ratings, dim_user, dim_anime, dim_date, results_path)
    run_bohdan_questions(fact_ratings, dim_user, dim_anime, dim_date, results_path)
    run_oskar_questions(fact_ratings, dim_user, dim_anime, dim_date, results_path)
    run_arii_extended_questions(fact_ratings, dim_user, dim_anime, dim_date, results_path)

    print("\n✅ Усі бізнес-питання виконано.")


def run_prepare_all_ml_datasets(spark: "SparkSession", data_path: str = DATA_PATH) -> None:
    """
    Окремий сценарій: примусово перебудувати обидва ML датасети.
    """
    ensure_raw_data(data_path)
    ensure_regression_dataset(spark, data_path=data_path)
    ensure_classification_dataset(spark, data_path=data_path)


def run_regression_step(spark: "SparkSession", data_path: str = DATA_PATH) -> None:
    """
    Стадія: підготовка датасетів (якщо потрібно) + заглушка для регресійної моделі.
    
    Не потребує зірчастої схеми - працює напряму з сирими CSV.
    """
    ensure_raw_data(data_path)
    ensure_regression_dataset(spark, data_path=data_path)
    run_regression_modeling(data_path=data_path)


def run_classification_step(spark: "SparkSession", data_path: str = DATA_PATH) -> None:
    """
    Стадія: підготовка датасетів (якщо потрібно) + заглушка для класифікаційної моделі.
    
    Не потребує зірчастої схеми - працює напряму з сирими CSV.
    """
    ensure_raw_data(data_path)
    ensure_classification_dataset(spark, data_path=data_path)
    run_classification_modeling(data_path=data_path)


# ======================================================================
# CLI-меню
# ======================================================================


def print_menu() -> None:
    print("\n" + "=" * 60)
    print("🎛  Anime Spark – головне меню")
    print("=" * 60)
    print("1) Бізнес-питання (потребує: сирі дані → зірчаста схема)")
    print("   └─ Створює зірчасту схему, запускає аналіз та бізнес-питання")
    print("")
    print("2) Регресія ML (потребує: сирі дані → ML датасет регресії)")
    print("   └─ Підготовка датасету (якщо потрібно) + запуск моделі")
    print("")
    print("3) Класифікація ML (потребує: сирі дані → ML датасет класифікації)")
    print("   └─ Підготовка датасету (якщо потрібно) + запуск моделі")
    print("")
    print("0) Вихід")


def main() -> None:
    """
    Головна точка входу застосунку.

    Дозволяє обрати стадію пайплайну:
    - бізнес-питання
    - регресія (ML, заглушка)
    - класифікація (ML, заглушка)
    """
    if not SPARK_AVAILABLE:
        print("❌ PySpark недоступний. Будь ласка, використовуйте Docker для запуску.")
        print('   Запустіть: docker run -v "$(pwd)/data:/app/data" my-spark-img')
        return

    print("🚀 Запуск Anime Spark App (PySpark)")

    spark = create_spark_session()

    try:
        while True:
            print_menu()
            choice = input("Оберіть дію: ").strip()

            if choice == "1":
                run_business_questions_flow(spark, data_path=DATA_PATH)
            elif choice == "2":
                run_regression_step(spark, data_path=DATA_PATH)
            elif choice == "3":
                run_classification_step(spark, data_path=DATA_PATH)
            elif choice == "0":
                print("👋 Вихід з застосунку.")
                break
            else:
                print("⚠️ Невірний вибір. Спробуйте ще раз.")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()