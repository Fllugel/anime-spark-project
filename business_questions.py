"""
Модуль з бізнес-питаннями для аналізу зірчастої схеми даних.
Кожен член команди може додавати свої питання у відповідну секцію.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, count, sum as spark_sum, avg, when, lag, lead
)
from pyspark.sql.window import Window


# ============================================================================
# ПИТАННЯ ВІД ARTEM (Аналітик 4)
# ============================================================================

def question_1_artem(fact_ratings, dim_anime):
    """
    (Filters) Скільки існує "дуже низьких" оцінок (Is_Low_Rating = 1),
    які були поставлені аніме з менш ніж 1000 учасниками (Members < 1000 з Dim_Anime)?
    """
    print("\n" + "=" * 60)
    print("❓ Питання 1 від Artem (Filters)")
    print("=" * 60)
    print("Скільки існує 'дуже низьких' оцінок, поставлених аніме з < 1000 учасниками?")
    
    result = fact_ratings \
        .join(dim_anime, fact_ratings.Anime_SK == dim_anime.Anime_SK, "inner") \
        .filter((col("Is_Low_Rating") == 1) & (col("Members") < 1000)) \
        .agg(count("*").alias("total_low_ratings"))
    
    result.show()
    return result


def question_2_artem(fact_ratings, dim_user):
    """
    (JOIN) Знайти користувачів, які не поставили жодної "низької оцінки"
    (Fact.Is_Low_Rating = 0 для всіх їхніх оцінок).
    """
    print("\n" + "=" * 60)
    print("❓ Питання 2 від Artem (JOIN)")
    print("=" * 60)
    print("Знайти користувачів, які не поставили жодної 'низької оцінки'")
    
    # Знаходимо користувачів, які мають хоча б одну низьку оцінку
    users_with_low_ratings = fact_ratings \
        .filter(col("Is_Low_Rating") == 1) \
        .select("User_SK") \
        .distinct()
    
    # Знаходимо всіх користувачів, які мають оцінки
    all_users_with_ratings = fact_ratings \
        .select("User_SK") \
        .distinct()
    
    # Знаходимо користувачів БЕЗ низьких оцінок
    users_without_low_ratings = all_users_with_ratings \
        .join(users_with_low_ratings, on="User_SK", how="left_anti") \
        .join(dim_user, on="User_SK", how="inner") \
        .select("User_SK", "User_ID", "Username", "User_Mean_Score") \
        .orderBy("User_SK")
    
    print(f"\nЗнайдено {users_without_low_ratings.count()} користувачів без низьких оцінок")
    users_without_low_ratings.show(10)
    
    return users_without_low_ratings


def question_3_artem(fact_ratings, dim_user):
    """
    (GROUP BY) Визначити 5 найкращих країн (Dim_User.Location) за сумарною
    кількістю "фанатських" оцінок (SUM(Fact.Is_Above_Average)).
    """
    print("\n" + "=" * 60)
    print("❓ Питання 3 від Artem (GROUP BY)")
    print("=" * 60)
    print("Топ 5 країн за сумарною кількістю 'фанатських' оцінок")
    
    result = fact_ratings \
        .join(dim_user, on="User_SK", how="inner") \
        .filter(col("Location").isNotNull()) \
        .groupBy("Location") \
        .agg(spark_sum("Is_Above_Average").alias("total_fan_ratings")) \
        .orderBy(col("total_fan_ratings").desc()) \
        .limit(5)
    
    result.show(truncate=False)
    return result


def question_4_artem(fact_ratings, dim_user):
    """
    (Window Functions) Для кожної окремої оцінки (Fact.User_Rating) показати
    відхилення цієї оцінки від середньої оцінки *цього* користувача
    (Fact.User_Rating - Dim_User.User_Mean_Score), використовуючи
    AVG() OVER (PARTITION BY d_user.User_SK).
    """
    print("\n" + "=" * 60)
    print("❓ Питання 4 від Artem (Window Functions)")
    print("=" * 60)
    print("Відхилення оцінки від середньої оцінки користувача")
    
    # Створюємо window для обчислення середньої оцінки користувача
    window_spec = Window.partitionBy("User_SK")
    
    result = fact_ratings \
        .join(dim_user.select("User_SK", "User_Mean_Score"), on="User_SK", how="inner") \
        .withColumn(
            "avg_user_rating",
            avg("User_Rating").over(window_spec)
        ) \
        .withColumn(
            "deviation_from_user_mean",
            col("User_Rating") - col("User_Mean_Score")
        ) \
        .withColumn(
            "deviation_from_avg_rating",
            col("User_Rating") - col("avg_user_rating")
        ) \
        .select(
            "User_SK",
            "Anime_SK",
            "User_Rating",
            "User_Mean_Score",
            "avg_user_rating",
            "deviation_from_user_mean",
            "deviation_from_avg_rating"
        ) \
        .limit(20)
    
    result.show(truncate=False)
    return result


def question_5_artem(dim_anime):
    """
    (Window Functions) Показати список аніме, відсортований за рангом популярності
    (Popularity_Rank), і вивести різницю (розрив) у популярності між поточним
    аніме і наступним, використовуючи LAG() або LEAD().
    """
    print("\n" + "=" * 60)
    print("❓ Питання 5 від Artem (Window Functions)")
    print("=" * 60)
    print("Різниця популярності між поточним аніме і наступним")
    
    # Створюємо window для сортування за популярністю
    window_spec = Window.orderBy("Popularity_Rank")
    
    result = dim_anime \
        .filter(col("Popularity_Rank").isNotNull()) \
        .withColumn(
            "next_popularity_rank",
            lead("Popularity_Rank", 1).over(window_spec)
        ) \
        .withColumn(
            "popularity_gap",
            col("next_popularity_rank") - col("Popularity_Rank")
        ) \
        .select(
            "Anime_SK",
            "Anime_ID",
            "Name",
            "Popularity_Rank",
            "next_popularity_rank",
            "popularity_gap"
        ) \
        .limit(20)
    
    result.show(truncate=False)
    return result


def question_6_artem(fact_ratings, dim_user):
    """
    (Window Functions) Для кожного користувача показати його загальну кількість
    оцінок (COUNT(*) OVER (PARTITION BY d_user.User_SK)) поруч з кожною
    його індивідуальною оцінкою.
    """
    print("\n" + "=" * 60)
    print("❓ Питання 6 від Artem (Window Functions)")
    print("=" * 60)
    print("Загальна кількість оцінок користувача поруч з кожною оцінкою")
    
    # Створюємо window для підрахунку загальної кількості оцінок користувача
    window_spec = Window.partitionBy("User_SK")
    
    result = fact_ratings \
        .join(dim_user.select("User_SK", "Username"), on="User_SK", how="inner") \
        .withColumn(
            "total_user_ratings",
            count("*").over(window_spec)
        ) \
        .select(
            "User_SK",
            "Username",
            "Anime_SK",
            "User_Rating",
            "total_user_ratings"
        ) \
        .limit(20)
    
    result.show(truncate=False)
    return result


def run_artem_questions(fact_ratings, dim_user, dim_anime, dim_date, results_path="results"):
    """
    Запускає всі бізнес-питання від Artem та зберігає результати у CSV.
    
    Args:
        fact_ratings: DataFrame з таблицею фактів
        dim_user: DataFrame з виміром користувачів
        dim_anime: DataFrame з виміром аніме
        dim_date: DataFrame з виміром дати
        results_path: Шлях для збереження результатів
    """
    print("\n" + "=" * 60)
    print("📊 БІЗНЕС-ПИТАННЯ ВІД ARTEM (Аналітик 4)")
    print("=" * 60)
    
    results = {}
    
    try:
        # Питання 1: Filters
        results['artem_q1'] = question_1_artem(fact_ratings, dim_anime)
        
        # Питання 2: JOIN
        results['artem_q2'] = question_2_artem(fact_ratings, dim_user)
        
        # Питання 3: GROUP BY
        results['artem_q3'] = question_3_artem(fact_ratings, dim_user)
        
        # Питання 4: Window Functions
        results['artem_q4'] = question_4_artem(fact_ratings, dim_user)
        
        # Питання 5: Window Functions
        results['artem_q5'] = question_5_artem(dim_anime)
        
        # Питання 6: Window Functions
        results['artem_q6'] = question_6_artem(fact_ratings, dim_user)
        
        # Зберігаємо результати у CSV
        print("\n" + "=" * 60)
        print("💾 ЗБЕРЕЖЕННЯ РЕЗУЛЬТАТІВ У CSV")
        print("=" * 60)
        
        import os
        os.makedirs(results_path, exist_ok=True)
        
        for key, df in results.items():
            try:
                output_file = f"{results_path}/{key}.csv"
                # Використовуємо coalesce(1) для створення одного файлу
                df.coalesce(1).write.mode("overwrite").option("header", "true").csv(output_file)
                print(f"✅ Збережено: {output_file}")
            except Exception as e:
                print(f"⚠️  Помилка збереження {key}: {e}")
        
        print(f"\n✅ Всі результати збережено в папці: {results_path}/")
        
    except Exception as e:
        print(f"❌ Помилка при виконанні питань від Artem: {str(e)}")
        import traceback
        traceback.print_exc()
    
    return results


# ============================================================================
# ТУТ МОЖУТЬ ДОДАВАТИСЯ ПИТАННЯ ВІД ІНШИХ ЧЛЕНІВ КОМАНДИ
# ============================================================================

"""
ІНСТРУКЦІЯ ДЛЯ ДОДАВАННЯ СВОЇХ БІЗНЕС-ПИТАНЬ:

1. Створіть функції для ваших питань у форматі:
   def question_N_yourname(fact_ratings, dim_user, dim_anime, dim_date):
       '''Опис питання'''
       print("\n" + "=" * 60)
       print("❓ Питання N від [Ваше ім'я]")
       print("=" * 60)
       # Ваш код тут
       result = ...
       result.show()
       return result

2. Створіть функцію run_yourname_questions() для запуску всіх ваших питань:
   def run_yourname_questions(fact_ratings, dim_user, dim_anime, dim_date, results_path="results"):
       '''Запускає всі бізнес-питання від [Ваше ім'я]'''
       print("\n" + "=" * 60)
       print("📊 БІЗНЕС-ПИТАННЯ ВІД [ВАШЕ ІМ'Я]")
       print("=" * 60)
       results = {}
       results['yourname_q1'] = question_1_yourname(fact_ratings, dim_user, dim_anime, dim_date)
       # Додайте інші питання...
       # Збереження результатів (опціонально)
       return results

3. Імпортуйте та викличте вашу функцію в main.py:
   from business_questions import run_yourname_questions
   results_yourname = run_yourname_questions(
       fact_ratings, dim_user, dim_anime, dim_date,
       results_path=f"{data_path}/results"
   )

ПРИКЛАД:
"""

# def question_1_teammate_name(fact_ratings, dim_user, dim_anime, dim_date):
#     """
#     (Filters) Приклад питання з фільтрами
#     """
#     print("\n" + "=" * 60)
#     print("❓ Питання 1 від Teammate Name")
#     print("=" * 60)
#     
#     result = fact_ratings \
#         .filter(col("User_Rating") >= 8) \
#         .count()
#     
#     print(f"Результат: {result}")
#     return result
#
# def run_teammate_name_questions(fact_ratings, dim_user, dim_anime, dim_date, results_path="results"):
#     """Запускає всі питання від Teammate Name"""
#     print("\n" + "=" * 60)
#     print("📊 БІЗНЕС-ПИТАННЯ ВІД TEAMMATE NAME")
#     print("=" * 60)
#     
#     results = {}
#     results['teammate_q1'] = question_1_teammate_name(fact_ratings, dim_user, dim_anime, dim_date)
#     
#     # Збереження результатів
#     import os
#     os.makedirs(results_path, exist_ok=True)
#     for key, df in results.items():
#         output_file = f"{results_path}/{key}.csv"
#         df.coalesce(1).write.mode("overwrite").option("header", "true").csv(output_file)
#     
#     return results

