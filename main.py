"""
Головний файл для створення зірчастої схеми даних та виконання бізнес-питань.
"""

try:
    from pyspark.sql import SparkSession
    from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False
    print("⚠️  PySpark недоступний локально, використовую pandas як альтернативу")

from data_extraction import (
    create_star_schema,
    save_star_schema_to_parquet,
    load_star_schema_from_parquet
)


def main():
    """
    Головна функція для створення зірчастої схеми та виконання бізнес-питань.
    """
    if SPARK_AVAILABLE:
        print("🚀 Використовую PySpark")
        
        # Створюємо SparkSession з JVM аргументами для Java 11/17
        spark = SparkSession.builder \
            .appName("AnimeStarSchemaAnalysis") \
            .config("spark.driver.extraJavaOptions", "--add-opens=java.base/java.nio=ALL-UNNAMED --add-opens=java.base/sun.nio.ch=ALL-UNNAMED") \
            .config("spark.executor.extraJavaOptions", "--add-opens=java.base/java.nio=ALL-UNNAMED --add-opens=java.base/sun.nio.ch=ALL-UNNAMED") \
            .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
            .config("spark.sql.parquet.datetimeRebaseModeInWrite", "LEGACY") \
            .getOrCreate()

        try:
            # Шлях до даних (працює як локально, так і в Docker з монтованим volume)
            data_path = "data"
            
            print("=" * 60)
            print("🌟 СТВОРЕННЯ ЗІРЧАСТОЇ СХЕМИ ДАНИХ")
            print("=" * 60)
            
            # Створюємо зірчасту схему
            dim_user, dim_anime, dim_date, fact_ratings = create_star_schema(spark, data_path)
            
            print("\n" + "=" * 60)
            print("📊 ПЕРЕВІРКА СТВОРЕНОЇ СХЕМИ")
            print("=" * 60)
            
            # Показуємо структуру вимірів
            print("\n📋 Dim_User (перші 5 рядків):")
            dim_user.show(5, truncate=False)
            
            print("\n📋 Dim_Anime (перші 5 рядків):")
            dim_anime.show(5, truncate=False)
            
            print("\n📋 Dim_Date (перші 10 рядків):")
            dim_date.show(10, truncate=False)
            
            print("\n📋 Fact_UserRatings (перші 10 рядків):")
            fact_ratings.show(10, truncate=False)
            
            # Показуємо схеми
            print("\n📐 Схема Dim_User:")
            dim_user.printSchema()
            
            print("\n📐 Схема Dim_Anime:")
            dim_anime.printSchema()
            
            print("\n📐 Схема Fact_UserRatings:")
            fact_ratings.printSchema()
            
            # Опціонально: зберігаємо у Parquet для швидшого доступу
            print("\n" + "=" * 60)
            print("💾 ЗБЕРЕЖЕННЯ СХЕМИ У PARQUET")
            print("=" * 60)
            try:
                save_star_schema_to_parquet(
                    dim_user, dim_anime, dim_date, fact_ratings,
                    output_path=f"{data_path}/star_schema"
                )
            except Exception as e:
                print(f"⚠️  Не вдалося зберегти у Parquet: {e}")
            
            print("\n" + "=" * 60)
            print("❓ БІЗНЕС-ПИТАННЯ")
            print("=" * 60)
            print("\n📝 Тут будуть додаватися бізнес-питання до даних...")
            print("   Використовуйте dim_user, dim_anime, dim_date, fact_ratings для аналізу.\n")
            
            # ============================================================
            # ТУТ БУДУТЬ ДОДАВАТИСЯ БІЗНЕС-ПИТАННЯ
            # ============================================================
            
            # Приклад: Показуємо статистику по оцінкам
            print("📊 Приклад: Статистика по оцінкам користувачів")
            fact_ratings.select("User_Rating").describe().show()
            
            print("\n✅ Всі кроки виконано успішно!")

        except Exception as e:
            print(f"❌ Помилка: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            spark.stop()
    else:
        print("❌ PySpark недоступний. Будь ласка, використовуйте Docker для запуску.")
        print("   Запустіть: docker run -v \"$(pwd)/data:/app/data\" my-spark-img")


if __name__ == "__main__":
    main()
