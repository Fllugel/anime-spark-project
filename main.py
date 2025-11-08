try:
    from pyspark.sql import SparkSession
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False
    print("⚠️  PySpark недоступний локально, використовую pandas як альтернативу")

try:
    import pandas as pd
except ImportError:
    pd = None
from data_extraction import create_anime_dataframe, validate_dataframe

# Import transformation stage modules
try:
    from transformation import dataset_info, numeric_statistics
    TRANSFORMATION_AVAILABLE = True
except ImportError:
    TRANSFORMATION_AVAILABLE = False
    print("⚠️  Модулі трансформації недоступні")

def main():
    if SPARK_AVAILABLE:
        print("🚀 Використовую PySpark")
        # Створіть SparkSession з налаштуваннями пам'яті
        spark = SparkSession.builder \
            .appName("AnimeDataExtraction") \
            .config("spark.driver.memory", "4g") \
            .config("spark.driver.maxResultSize", "2g") \
            .config("spark.executor.memory", "4g") \
            .config("spark.sql.shuffle.partitions", "200") \
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

            print("\n✅ Всі кроки витягування даних виконано успішно!")
            
            # ЕТАП ТРАНСФОРМАЦІЇ
            if TRANSFORMATION_AVAILABLE:
                print("\n" + "="*80)
                print("ПОЧАТОК ЕТАПУ ТРАНСФОРМАЦІЇ")
                print("="*80)
                
                try:
                    # Етап 1: Загальна інформація про набір даних
                    dataset_info.run_dataset_info_analysis(anime_df)
                    
                    # Етап 2: Статистика числових стовпців
                    numeric_statistics.run_numeric_statistics_analysis(anime_df)
                    
                    print("\n" + "="*80)
                    print("✅ ЕТАП ТРАНСФОРМАЦІЇ ЗАВЕРШЕНО УСПІШНО!")
                    print("="*80)
                    
                except Exception as e:
                    print(f"\n❌ Помилка під час трансформації: {str(e)}")
                    import traceback
                    traceback.print_exc()
            else:
                print("\n⚠️  Етап трансформації пропущено (модулі недоступні)")

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

            print("\n✅ Всі кроки витягування даних виконано успішно!")
            
            # ЕТАП ТРАНСФОРМАЦІЇ (тільки для PySpark)
            print("\n⚠️  Етап трансформації доступний тільки з PySpark")

        except Exception as e:
            print(f"❌ Помилка: {str(e)}")

if __name__ == "__main__":
    main()
