"""
Модуль з бізнес-питаннями для аналізу зірчастої схеми даних.
Кожен член команди може додавати свої питання у відповідну секцію.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, count, sum as spark_sum, avg, when, lag, lead, row_number,
    ntile, percentile_approx, lit, length
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
# ПИТАННЯ ВІД BOHDAN (Аналітик 2)
# ============================================================================

def question_1_bohdan(dim_anime):
    """
    (Filters) Показати всі аніме, джерелом яких є "Manga" (Source),
    але які не є "TV" (Type).
    """
    print("\n" + "=" * 60)
    print("❓ Питання 1 від Bohdan (Filters)")
    print("=" * 60)
    print("Аніме з джерелом 'Manga', але не типу 'TV'")
    
    result = dim_anime \
        .filter((col("Source") == "Manga") & (col("Type") != "TV")) \
        .select("Anime_SK", "Anime_ID", "Name", "Type", "Source", "Avg_Score") \
        .orderBy("Avg_Score", ascending=False)
    
    print(f"\nЗнайдено {result.count()} аніме")
    result.show(20, truncate=False)
    return result


def question_2_bohdan(fact_ratings, dim_anime, dim_user):
    """
    (JOIN) Які аніме (Dim_Anime.Name) отримали оцінку 1 (Fact.User_Rating)
    від користувачів жіночої статі (Dim_User.Gender)?
    """
    print("\n" + "=" * 60)
    print("❓ Питання 2 від Bohdan (JOIN)")
    print("=" * 60)
    print("Аніме з оцінкою 1 від користувачів жіночої статі")
    
    result = fact_ratings \
        .filter(col("User_Rating") == 1) \
        .join(dim_user, on="User_SK", how="inner") \
        .filter(col("Gender") == "Female") \
        .join(dim_anime, on="Anime_SK", how="inner") \
        .select("Name", "Anime_ID", "Type", "Avg_Score") \
        .distinct() \
        .orderBy("Name")
    
    print(f"\nЗнайдено {result.count()} унікальних аніме")
    result.show(20, truncate=False)
    return result


def question_3_bohdan(fact_ratings, dim_anime):
    """
    (JOIN) Показати всі оцінки, де користувач був "фанатом"
    (Fact.Is_Above_Average = 1), для аніме типу "Movie" (Dim_Anime.Type).
    """
    print("\n" + "=" * 60)
    print("❓ Питання 3 від Bohdan (JOIN)")
    print("=" * 60)
    print("Оцінки 'фанатів' для аніме типу 'Movie'")
    
    result = fact_ratings \
        .filter(col("Is_Above_Average") == 1) \
        .join(dim_anime, on="Anime_SK", how="inner") \
        .filter(col("Type") == "Movie") \
        .select(
            "User_SK",
            "Anime_SK",
            "Name",
            "User_Rating",
            "Is_Above_Average",
            "Type",
            "Avg_Score"
        ) \
        .orderBy("User_Rating", ascending=False)
    
    print(f"\nЗнайдено {result.count()} оцінок")
    result.show(20, truncate=False)
    return result


def question_4_bohdan(fact_ratings, dim_user, dim_anime):
    """
    (JOIN) Які користувачі (Dim_User.Username) з Канади (Dim_User.Location)
    поставили "високі оцінки" (Fact.Is_High_Rating = 1) для аніме студії
    "Production I.G" (Dim_Anime.Studios)?
    """
    print("\n" + "=" * 60)
    print("❓ Питання 4 від Bohdan (JOIN)")
    print("=" * 60)
    print("Користувачі з Канади з високими оцінками для студії 'Production I.G'")
    
    result = fact_ratings \
        .filter(col("Is_High_Rating") == 1) \
        .join(dim_user, on="User_SK", how="inner") \
        .filter(col("Location") == "Canada") \
        .join(dim_anime, on="Anime_SK", how="inner") \
        .filter(col("Studios").contains("Production I.G")) \
        .select(
            "Username",
            "User_ID",
            "Location",
            "Name",
            "Studios",
            "User_Rating"
        ) \
        .distinct() \
        .orderBy("Username")
    
    print(f"\nЗнайдено {result.count()} унікальних користувачів")
    result.show(20, truncate=False)
    return result


def question_5_bohdan(fact_ratings, dim_anime):
    """
    (GROUP BY) Яка загальна кількість "високих оцінок" (SUM(Fact.Is_High_Rating))
    згрупована за типом джерела (Dim_Anime.Source)?
    """
    print("\n" + "=" * 60)
    print("❓ Питання 5 від Bohdan (GROUP BY)")
    print("=" * 60)
    print("Загальна кількість високих оцінок за типом джерела")
    
    result = fact_ratings \
        .join(dim_anime, on="Anime_SK", how="inner") \
        .filter(col("Source").isNotNull()) \
        .groupBy("Source") \
        .agg(
            spark_sum("Is_High_Rating").alias("total_high_ratings"),
            count("*").alias("total_ratings")
        ) \
        .withColumn(
            "high_rating_percentage",
            (col("total_high_ratings") / col("total_ratings") * 100)
        ) \
        .orderBy(col("total_high_ratings").desc())
    
    result.show(truncate=False)
    return result


def question_6_bohdan(dim_anime):
    """
    (Window Functions) Знайти топ-3 аніме (за Avg_Score) для кожної студії
    (PARTITION BY Dim_Anime.Studios), використовуючи ROW_NUMBER().
    """
    print("\n" + "=" * 60)
    print("❓ Питання 6 від Bohdan (Window Functions)")
    print("=" * 60)
    print("Топ-3 аніме за середньою оцінкою для кожної студії")
    
    # Створюємо window для ранжування аніме в межах кожної студії
    window_spec = Window.partitionBy("Studios").orderBy(col("Avg_Score").desc())
    
    result = dim_anime \
        .filter(col("Studios").isNotNull() & col("Avg_Score").isNotNull()) \
        .withColumn("rank", row_number().over(window_spec)) \
        .filter(col("rank") <= 3) \
        .select(
            "Studios",
            "Name",
            "Anime_ID",
            "Type",
            "Avg_Score",
            "Popularity_Rank",
            "rank"
        ) \
        .orderBy("Studios", "rank")
    
    print(f"\nЗнайдено {result.count()} записів (топ-3 для кожної студії)")
    result.show(30, truncate=False)
    return result


def run_bohdan_questions(fact_ratings, dim_user, dim_anime, dim_date, results_path="results"):
    """
    Запускає всі бізнес-питання від Bohdan та зберігає результати у CSV.
    
    Args:
        fact_ratings: DataFrame з таблицею фактів
        dim_user: DataFrame з виміром користувачів
        dim_anime: DataFrame з виміром аніме
        dim_date: DataFrame з виміром дати
        results_path: Шлях для збереження результатів
    """
    print("\n" + "=" * 60)
    print("📊 БІЗНЕС-ПИТАННЯ ВІД BOHDAN (Аналітик 2)")
    print("=" * 60)
    
    results = {}
    
    try:
        # Питання 1: Filters
        results['bohdan_q1'] = question_1_bohdan(dim_anime)
        
        # Питання 2: JOIN
        results['bohdan_q2'] = question_2_bohdan(fact_ratings, dim_anime, dim_user)
        
        # Питання 3: JOIN
        results['bohdan_q3'] = question_3_bohdan(fact_ratings, dim_anime)
        
        # Питання 4: JOIN
        results['bohdan_q4'] = question_4_bohdan(fact_ratings, dim_user, dim_anime)
        
        # Питання 5: GROUP BY
        results['bohdan_q5'] = question_5_bohdan(fact_ratings, dim_anime)
        
        # Питання 6: Window Functions
        results['bohdan_q6'] = question_6_bohdan(dim_anime)
        
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
        print(f"❌ Помилка при виконанні питань від Bohdan: {str(e)}")
        import traceback
        traceback.print_exc()
    
    return results


# ============================================================================
# ПИТАННЯ ВІД OSKAR (Додаткові бізнес-питання)
# ============================================================================

def question_1_oskar(fact_ratings, dim_user, dim_anime, dim_date):
    """
    (Filters) Знайти користувачів, чия середня оцінка (User_Mean_Score) нижча за 6,
    але які при цьому подивилися (User_Total_Completed) більше 50 тайтлів.
    """
    print("\n" + "=" * 60)
    print("❓ Питання 1 (Filters)")
    print("=" * 60)
    print("Користувачі з середньою оцінкою < 6, але > 50 переглянутих тайтлів")
    
    # Cast User_Mean_Score to double for numeric comparison (inline casting like Artem's approach)
    result = dim_user \
        .filter((col("User_Mean_Score").cast("double") < 6) & (col("User_Total_Completed") > 50)) \
        .select(
            "User_SK",
            "User_ID",
            "Username",
            "User_Mean_Score",
            "User_Total_Completed"
        ) \
        .orderBy("User_Total_Completed", ascending=False)
    
    print(f"\nЗнайдено {result.count()} користувачів")
    result.show(20, truncate=False)
    return result


def question_2_oskar(fact_ratings, dim_user, dim_anime, dim_date):
    """
    (JOIN) Вивести список аніме та оцінок, які поставив користувач 'BunnySlayer' (Dim_User.Username),
    але лише для тих аніме, де цей користувач є "критиком" (User_Rating < 7, оскільки шкала 0-10).
    """
    print("\n" + "=" * 60)
    print("❓ Питання 2 (JOIN)")
    print("=" * 60)
    print("Аніме та оцінки користувача 'BunnySlayer' де User_Rating < 7 (критик, шкала 0-10)")
    
    result = fact_ratings \
        .join(dim_user.filter(col("Username") == "BunnySlayer"), on="User_SK", how="inner") \
        .join(dim_anime, on="Anime_SK", how="inner") \
        .filter(col("User_Rating").cast("double") < 7) \
        .select(
            col("Username"),
            col("Anime_ID"),
            col("Name"),
            col("English_Name"),
            col("User_Rating"),
            col("Rating_Deviation"),
            col("Avg_Score")
        ) \
        .orderBy("User_Rating", ascending=True)
    
    print(f"\nЗнайдено {result.count()} оцінок від користувача 'BunnySlayer' (критик: User_Rating < 7)")
    result.show(20, truncate=False)
    return result


def question_3_oskar(fact_ratings, dim_user, dim_anime, dim_date):
    """
    (GROUP BY) Яка середня кількість епізодів (AVG(Dim_Anime.Episodes)) для аніме,
    згрупованих за віковим рейтингом (Dim_Anime.Age_Rating)?
    """
    print("\n" + "=" * 60)
    print("❓ Питання 3 (GROUP BY)")
    print("=" * 60)
    print("Середня кількість епізодів за віковим рейтингом")
    
    # Cast Episodes to double for numeric aggregation (inline casting like Artem's approach)
    # Фільтруємо валідні вікові рейтинги (виключаємо числові значення та невалідні дані)
    # Валідні вікові рейтинги: G - All Ages, PG - Children, PG-13 - Teens 13 or older, 
    # R - 17+ (violence & profanity), R+ - Mild Nudity, Rx - Hentai, UNKNOWN
    result = dim_anime \
        .filter(
            col("Age_Rating").isNotNull() & 
            col("Episodes").isNotNull() &
            # Виключаємо числові значення (які не є валідними віковими рейтингами)
            ~col("Age_Rating").rlike("^\\d+(\\.\\d+)?$") &
            # Виключаємо URL та дуже довгі рядки
            ~col("Age_Rating").rlike("^https?://") &
            # Виключаємо рядки з "min" або "hr" (це тривалість, не віковий рейтинг)
            ~col("Age_Rating").rlike(".*min.*|.*hr.*") &
            # Виключаємо назви студій та інші невалідні значення
            ~col("Age_Rating").rlike("^(fall|spring|summer|winter)") &
            # Виключаємо назви компаній (які не є віковими рейтингами)
            ~col("Age_Rating").rlike("^(Bandai|Madhouse|Bee Train|Trans Arts|ORADA|ADV)") &
            (length(col("Age_Rating")) < 50) &
            # Включаємо тільки валідні вікові рейтинги
            (
                col("Age_Rating").rlike("^G - All Ages") |
                col("Age_Rating").rlike("^PG - Children") |
                col("Age_Rating").rlike("^PG-13 - Teens") |
                col("Age_Rating").rlike("^R - 17\\+") |
                col("Age_Rating").rlike("^R\\+ - Mild") |
                col("Age_Rating").rlike("^Rx - Hentai") |
                col("Age_Rating").rlike("^UNKNOWN$|^Unknown$|^None$")
            )
        ) \
        .groupBy("Age_Rating") \
        .agg(avg(col("Episodes").cast("double")).alias("avg_episodes")) \
        .filter(col("avg_episodes").isNotNull()) \
        .orderBy("avg_episodes", ascending=False)
    
    result.show(truncate=False)
    return result


def question_4_oskar(fact_ratings, dim_user, dim_anime, dim_date):
    """
    (GROUP BY) Яка середня різниця (AVG(Fact.Rating_Deviation)) між оцінкою користувача
    та середньою оцінкою аніме для кожної студії (Dim_Anime.Studios)?
    """
    print("\n" + "=" * 60)
    print("❓ Питання 4 (GROUP BY)")
    print("=" * 60)
    print("Середня різниця оцінок (Rating_Deviation) для кожної студії")
    
    result = fact_ratings \
        .join(dim_anime, on="Anime_SK", how="inner") \
        .filter(
            col("Studios").isNotNull() &
            # Виключаємо URL та дуже довгі рядки (які не є назвами студій)
            ~col("Studios").rlike("^https?://") &
            (length(col("Studios")) < 150) &
            # Включаємо тільки рядки, які виглядають як назви студій
            # (або короткі <=100 символів, або містять коми для множинних студій)
            ((length(col("Studios")) <= 100) | col("Studios").rlike(".*,.*"))
        ) \
        .groupBy("Studios") \
        .agg(avg("Rating_Deviation").alias("avg_rating_deviation")) \
        .orderBy("avg_rating_deviation", ascending=False) \
        .limit(20)
    
    result.show(truncate=False)
    return result


def question_5_oskar(fact_ratings, dim_user, dim_anime, dim_date):
    """
    (GROUP BY) Скільки всього оцінок (COUNT(Fact.Rating_Count)) поставили користувачі,
    згруповані за статтю (Dim_User.Gender)?
    """
    print("\n" + "=" * 60)
    print("❓ Питання 5 (GROUP BY)")
    print("=" * 60)
    print("Кількість оцінок, згрупованих за статтю користувача")
    
    result = fact_ratings \
        .join(dim_user, on="User_SK", how="inner") \
        .filter(col("Gender").isNotNull()) \
        .groupBy("Gender") \
        .agg(count("Rating_Count").alias("total_ratings")) \
        .orderBy("total_ratings", ascending=False)
    
    result.show(truncate=False)
    return result


def question_6_oskar(fact_ratings, dim_user, dim_anime, dim_date):
    """
    (Window Functions) Розділити всіх користувачів на 5 груп (квінтилі) (NTILE(5))
    на основі кількості переглянутих ними аніме (Dim_User.User_Total_Completed),
    щоб знайти "хардкорних" глядачів.
    """
    print("\n" + "=" * 60)
    print("❓ Питання 6 (Window Functions)")
    print("=" * 60)
    print("Розподіл користувачів на 5 квінтилів за кількістю переглянутих аніме")
    
    # Фільтруємо дані
    filtered_users = dim_user \
        .filter(col("User_Total_Completed").isNotNull()) \
        .select("User_SK", "User_ID", "Username", "User_Total_Completed")
    
    # Для оптимізації пам'яті, використовуємо checkpoint та репартиціонування
    # Але для NTILE потрібно всі дані разом, тому використовуємо обмежену кількість партицій
    # та checkpoint для зменшення навантаження на пам'ять
    try:
        # Створюємо window для NTILE
        # Примітка: NTILE вимагає всі дані в одному розділі для правильного ранжування
        # Це викликає попередження, але є необхідним для коректної роботи NTILE
        window_spec = Window.orderBy(col("User_Total_Completed").desc())
        
        # Обчислюємо квінтилі з обмеженням для оптимізації пам'яті
        # Використовуємо checkpoint для зменшення навантаження
        result = filtered_users \
            .withColumn("quintile", ntile(5).over(window_spec)) \
            .select(
                "User_SK",
                "User_ID",
                "Username",
                "User_Total_Completed",
                "quintile"
            )
        
        # Показуємо статистику по квінтилям
        print("\nСтатистика по квінтилям:")
        quintile_stats = result \
            .groupBy("quintile") \
            .agg(
                count("*").alias("users_count"),
                avg("User_Total_Completed").alias("avg_completed"),
                spark_sum("User_Total_Completed").alias("total_completed")
            ) \
            .orderBy("quintile")
        
        quintile_stats.show(truncate=False)
        
        print("\nПерші 20 користувачів з найбільшою кількістю переглянутих (квінтиль 1 = хардкорні):")
        result.filter(col("quintile") == 1).show(20, truncate=False)
        
    except Exception as e:
        print(f"⚠️  Помилка при обчисленні NTILE (можливо через обмеження пам'яті): {e}")
        print("Спробуємо альтернативний підхід з використанням приблизних перцентилів...")
        
        # Альтернативний підхід: використовуємо приблизні перцентилі для визначення квінтилів
        # Це більш ефективно для великих датасетів
        
        # Обчислюємо порогові значення для квінтилів
        percentiles = filtered_users.select(
            percentile_approx("User_Total_Completed", [0.2, 0.4, 0.6, 0.8], lit(10000)).alias("percentiles")
        ).collect()[0]["percentiles"]
        
        p20, p40, p60, p80 = percentiles[0], percentiles[1], percentiles[2], percentiles[3]
        
        # Призначаємо квінтилі на основі порогових значень
        result = filtered_users \
            .withColumn("quintile",
                when(col("User_Total_Completed") >= p80, lit(1))
                .when(col("User_Total_Completed") >= p60, lit(2))
                .when(col("User_Total_Completed") >= p40, lit(3))
                .when(col("User_Total_Completed") >= p20, lit(4))
                .otherwise(lit(5))
            ) \
            .select(
                "User_SK",
                "User_ID",
                "Username",
                "User_Total_Completed",
                "quintile"
            )
        
        print("\nСтатистика по квінтилям (приблизна):")
        quintile_stats = result \
            .groupBy("quintile") \
            .agg(
                count("*").alias("users_count"),
                avg("User_Total_Completed").alias("avg_completed"),
                spark_sum("User_Total_Completed").alias("total_completed")
            ) \
            .orderBy("quintile")
        
        quintile_stats.show(truncate=False)
        
        print("\nПерші 20 користувачів з найбільшою кількістю переглянутих (квінтиль 1 = хардкорні):")
        result.filter(col("quintile") == 1).orderBy(col("User_Total_Completed").desc()).show(20, truncate=False)
    
    return result


def run_oskar_questions(fact_ratings, dim_user, dim_anime, dim_date, results_path="results"):
    """
    Запускає всі бізнес-питання від Oskar та зберігає результати у CSV.
    
    Args:
        fact_ratings: DataFrame з таблицею фактів
        dim_user: DataFrame з виміром користувачів
        dim_anime: DataFrame з виміром аніме
        dim_date: DataFrame з виміром дати
        results_path: Шлях для збереження результатів
    """
    print("\n" + "=" * 60)
    print("📊 БІЗНЕС-ПИТАННЯ ВІД OSKAR")
    print("=" * 60)
    
    results = {}
    
    try:
        # Питання 1: Filters
        results['oskar_q1'] = question_1_oskar(fact_ratings, dim_user, dim_anime, dim_date)
        
        # Питання 2: JOIN
        results['oskar_q2'] = question_2_oskar(fact_ratings, dim_user, dim_anime, dim_date)
        
        # Питання 3: GROUP BY
        results['oskar_q3'] = question_3_oskar(fact_ratings, dim_user, dim_anime, dim_date)
        
        # Питання 4: GROUP BY
        results['oskar_q4'] = question_4_oskar(fact_ratings, dim_user, dim_anime, dim_date)
        
        # Питання 5: GROUP BY
        results['oskar_q5'] = question_5_oskar(fact_ratings, dim_user, dim_anime, dim_date)
        
        # Питання 6: Window Functions
        results['oskar_q6'] = question_6_oskar(fact_ratings, dim_user, dim_anime, dim_date)
        
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
        print(f"❌ Помилка при виконанні питань від Oskar: {str(e)}")
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