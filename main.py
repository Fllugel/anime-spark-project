try:
    from pyspark.sql import SparkSession
    from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType, ArrayType
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False
    print("⚠️  PySpark недоступний локально, використовую pandas як альтернативу")

import pandas as pd
from data_extraction import create_anime_dataframe, validate_dataframe

def main():
    if SPARK_AVAILABLE:
        print("🚀 Використовую PySpark")
        # Створіть SparkSession
        spark = SparkSession.builder \
            .appName("AnimeDataExtraction") \
            .getOrCreate()

        try:
            # Крок 1: Створіть відповідні схеми для набору даних
            print("Крок 1: Створення схем для набору даних")

            # Крок 2: Використовуючи створені схеми, створіть відповідні DataFrame
            print("Крок 2: Створення DataFrame з CSV файлу")
            anime_df = create_anime_dataframe(spark, "data/final_animedataset.csv")

            # Крок 3: Перевірте чи коректно все зчиталось
            print("Крок 3: Валідація DataFrame")
            validate_dataframe(anime_df)

            print("\n✅ Всі кроки виконано успішно!")

        except Exception as e:
            print(f"❌ Помилка: {str(e)}")
        finally:
            spark.stop()
    else:
        print("🚀 Використовую pandas (альтернатива PySpark)")
        try:
            # Крок 1: Створіть відповідні схеми для набору даних
            print("Крок 1: Створення схем для набору даних")

            # Крок 2: Створіть DataFrame з CSV файлу
            print("Крок 2: Створення DataFrame з CSV файлу")
            anime_df = create_anime_dataframe(None, "data/final_animedataset.csv")

            # Крок 3: Перевірте чи коректно все зчиталось
            print("Крок 3: Валідація DataFrame")
            validate_dataframe(anime_df)

            print("\n✅ Всі кроки виконано успішно!")

        except Exception as e:
            print(f"❌ Помилка: {str(e)}")

if __name__ == "__main__":
    main()
