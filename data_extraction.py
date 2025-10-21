from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType, ArrayType
from pyspark.sql.functions import col, count, when, isnull

def create_anime_schema():
    """
    Створює схему для аніме датасету
    """
    return StructType([
        StructField("anime_id", IntegerType(), True),
        StructField("name", StringType(), True),
        StructField("genre", StringType(), True),  # або ArrayType(StringType()) для списку жанрів
        StructField("type", StringType(), True),
        StructField("episodes", IntegerType(), True),
        StructField("rating", DoubleType(), True),
        StructField("members", IntegerType(), True),
        StructField("synopsis", StringType(), True),
        StructField("studios", StringType(), True),
        StructField("source", StringType(), True),
        StructField("duration", StringType(), True),
        StructField("aired", StringType(), True),
        StructField("rank", DoubleType(), True),
        StructField("popularity", IntegerType(), True),
        StructField("favorites", IntegerType(), True),
        StructField("scored_by", IntegerType(), True),
        StructField("status", StringType(), True),
        StructField("premiered", StringType(), True),
        StructField("broadcast", StringType(), True),
        StructField("related", StringType(), True)
    ])

def create_anime_dataframe(spark: SparkSession, file_path: str):
    """
    Створює DataFrame для аніме датасету з заданою схемою

    Args:
        spark: SparkSession
        file_path: Шлях до CSV файлу

    Returns:
        DataFrame з аніме даними
    """
    # Створюємо схему
    schema = create_anime_schema()

    # Зчитуємо CSV файл з інференцією схеми або з заданою схемою
    try:
        # Спочатку спробуємо з інференцією схеми
        df = spark.read.csv(file_path, header=True, inferSchema=True)
        print(f"✅ Файл зчитано з інференцією схеми. Кількість колонок: {len(df.columns)}")
        print(f"Назви колонок: {df.columns}")
        return df
    except Exception as e:
        print(f"⚠️  Помилка з інференцією схеми: {e}")
        print("🔄 Спроба зчитати з заданою схемою...")

        # Якщо інференція не працює, використовуємо задану схему
        df = spark.read.csv(file_path, header=True, schema=schema)
        print(f"✅ Файл зчитано з заданою схемою. Кількість колонок: {len(df.columns)}")
        return df

def validate_dataframe(df):
    """
    Перевіряє чи коректно зчитався DataFrame

    Args:
        df: DataFrame для валідації
    """
    print("\n📊 Інформація про DataFrame:")
    print(f"Кількість рядків: {df.count()}")
    print(f"Кількість колонок: {len(df.columns)}")
    print(f"Назви колонок: {df.columns}")

    # Показуємо перші 5 рядків
    print("\n🔍 Перші 5 рядків:")
    df.show(5, truncate=True)

    # Показуємо статистику по колонках
    print("\n📈 Статистика по колонках:")
    df.printSchema()

    # Перевіряємо на null значення в ключових колонках
    print("\n🔍 Перевірка на null значення в ключових колонках:")
    key_columns = ['anime_id', 'name', 'rating']
    for col_name in key_columns:
        if col_name in df.columns:
            null_count = df.filter(isnull(col(col_name))).count()
            print(f"  Колонка '{col_name}': {null_count} null значень")

    # Показуємо статистику по числовим колонкам
    print("\n📊 Статистика по числовим колонкам:")
    numeric_columns = ['episodes', 'rating', 'members', 'popularity', 'favorites']
    for col_name in numeric_columns:
        if col_name in df.columns:
            try:
                df.select(col_name).describe().show()
            except:
                print(f"  Не вдалося отримати статистику для колонки '{col_name}'")

    print("\n✅ Валідацію завершено!")

def get_dataframe_info(df):
    """
    Допоміжна функція для отримання інформації про DataFrame

    Args:
        df: DataFrame

    Returns:
        Dict з інформацією про DataFrame
    """
    return {
        'row_count': df.count(),
        'column_count': len(df.columns),
        'columns': df.columns,
        'schema': df.schema
    }
