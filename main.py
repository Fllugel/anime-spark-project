from data_extraction import (
    create_star_schema,
    save_star_schema_to_parquet,
    load_star_schema_from_parquet
)
from business_questions import run_artem_questions, run_bohdan_questions, run_oskar_questions # ⬅️ ОБИДВІ ФУНКЦІЇ ІМПОРТОВАНО


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
            .config("spark.driver.memory", "2g") \
            .config("spark.executor.memory", "2g") \
            .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .config("spark.sql.execution.arrow.maxRecordsPerBatch", "1000") \
            .getOrCreate()
        
        # Налаштовуємо рівень логування для приховування попереджень про Window operations
        spark.sparkContext.setLogLevel("ERROR")

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
                print(f"⚠️  Не вдалося зберегти у Parquet: {e}")
            
            print("\n" + "=" * 60)
            print("❓ БІЗНЕС-ПИТАННЯ")
            print("=" * 60)
            
            # ============================================================
            # БІЗНЕС-ПИТАННЯ ВІД РІЗНИХ ЧЛЕНІВ КОМАНДИ
            # ============================================================
            
            # Бізнес-питання від Artem (Аналітик 4)
            results_artem = run_artem_questions(
                fact_ratings, dim_user, dim_anime, dim_date,
                results_path=f"{data_path}/results"
            )
            
            # Бізнес-питання від Bohdan (Аналітик 2) ⬅️ ДОДАНО ВИКЛИК БОГДАНА
            results_bohdan = run_bohdan_questions(
                fact_ratings, dim_user, dim_anime, dim_date,
                results_path=f"{data_path}/results"
            )

            # Бізнес-питання від Oskar ⬅️ ДОДАНО ВИКЛИК ОСКАРА
            results_oskar = run_oskar_questions(
                fact_ratings, dim_user, dim_anime, dim_date,
                results_path=f"{data_path}/results"
            )
            
            # ============================================================
            # ТУТ МОЖУТЬ ДОДАВАТИСЯ ПИТАННЯ ВІД ІНШИХ ЧЛЕНІВ КОМАНДИ
            # ============================================================
            # Приклад:
            # from business_questions import run_teammate_name_questions
            # results_teammate = run_teammate_name_questions(
            #     fact_ratings, dim_user, dim_anime, dim_date,
            #     results_path=f"{data_path}/results"
            # )
            
            print("\n✅ Всі кроки виконано успішно!")

        except Exception as e:
            print(f"❌ Помилка: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            spark.stop()
    else:
        print("❌ PySpark недоступний. Будь ласка, використовуйте Docker для запуску.")
        print("   Запустіть: docker run -v \"$(pwd)/data:/app/data\" my-spark-img")


if __name__ == "__main__":
    main()