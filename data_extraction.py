"""
Модуль для створення зірчастої схеми даних (Star Schema) для аналізу аніме датасету.
Створює виміри (Dimensions) та таблицю фактів (Fact Table) для подальшого аналізу.
"""

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, IntegerType, StringType, DoubleType, 
    DateType, BooleanType, FloatType
)
from pyspark.sql.functions import (
    col, when, isnull, lit, row_number, to_date, year, quarter, month,
    dayofweek, date_format, expr, monotonically_increasing_id
)
from pyspark.sql.window import Window


def create_dim_user(spark: SparkSession, users_details_path: str):
    """
    Створює вимір користувачів (Dim_User) з сурогатним ключем.
    
    Args:
        spark: SparkSession
        users_details_path: Шлях до файлу users-details-2023.csv
        
    Returns:
        DataFrame з виміром користувачів
    """
    print("📊 Створення Dim_User...")
    
    # Зчитуємо дані користувачів
    df_users = spark.read.csv(users_details_path, header=True, inferSchema=True)
    
    # Перейменовуємо колонки для відповідності схемі
    df_users = df_users.select(
        col("Mal ID").alias("User_ID"),
        col("Username").alias("Username"),
        col("Gender").alias("Gender"),
        col("Birthday").alias("Birthday"),
        col("Location").alias("Location"),
        col("Joined").alias("Joined_Date"),
        col("Mean Score").alias("User_Mean_Score"),
        col("Completed").alias("User_Total_Completed"),
        col("Watching").alias("User_Watching"),
        col("On Hold").alias("User_On_Hold"),
        col("Dropped").alias("User_Dropped"),
        col("Plan to Watch").alias("User_Plan_to_Watch"),
        col("Total Entries").alias("User_Total_Entries"),
        col("Days Watched").alias("User_Days_Watched"),
        col("Episodes Watched").alias("User_Episodes_Watched")
    )
    
    # Конвертуємо дати (обробляємо формат ISO з часовим поясом)
    from pyspark.sql.functions import regexp_replace, split
    
    # Видаляємо часовий пояс та час з дат (формат: 2011-01-10T00:00:00+00:00 -> 2011-01-10)
    df_users = df_users.withColumn(
        "Birthday", 
        when(col("Birthday").isNotNull(), 
             to_date(regexp_replace(col("Birthday"), "T.*", ""), "yyyy-MM-dd"))
        .otherwise(None)
    )
    df_users = df_users.withColumn(
        "Joined_Date",
        when(col("Joined_Date").isNotNull(),
             to_date(regexp_replace(col("Joined_Date"), "T.*", ""), "yyyy-MM-dd"))
        .otherwise(None)
    )
    
    # Створюємо сурогатний ключ (User_SK)
    window = Window.orderBy("User_ID")
    df_users = df_users.withColumn("User_SK", row_number().over(window))
    
    # Вибераємо колонки в правильному порядку
    dim_user = df_users.select(
        "User_SK",
        "User_ID",
        "Username",
        "Gender",
        "Birthday",
        "Location",
        "Joined_Date",
        "User_Mean_Score",
        "User_Total_Completed",
        "User_Watching",
        "User_On_Hold",
        "User_Dropped",
        "User_Plan_to_Watch",
        "User_Total_Entries",
        "User_Days_Watched",
        "User_Episodes_Watched"
    )
    
    print(f"✅ Dim_User створено: {dim_user.count()} користувачів")
    return dim_user


def create_dim_anime(spark: SparkSession, anime_dataset_path: str):
    """
    Створює вимір аніме (Dim_Anime) з сурогатним ключем.
    
    Args:
        spark: SparkSession
        anime_dataset_path: Шлях до файлу anime-dataset-2023.csv
        
    Returns:
        DataFrame з виміром аніме
    """
    print("📊 Створення Dim_Anime...")
    
    # Зчитуємо дані аніме
    df_anime = spark.read.csv(anime_dataset_path, header=True, inferSchema=True)
    
    # Перейменовуємо та вибираємо потрібні колонки
    df_anime = df_anime.select(
        col("anime_id").alias("Anime_ID"),
        col("Name").alias("Name"),
        col("English name").alias("English_Name"),
        col("Type").alias("Type"),
        col("Source").alias("Source"),
        col("Genres").alias("Genres"),
        col("Studios").alias("Studios"),
        col("Producers").alias("Producers"),
        col("Score").alias("Avg_Score"),
        col("Popularity").alias("Popularity_Rank"),
        col("Episodes").alias("Episodes"),
        col("Rating").alias("Age_Rating"),
        col("Rank").alias("Rank"),
        col("Members").alias("Members"),
        col("Favorites").alias("Favorites"),
        col("Scored By").alias("Scored_By"),
        col("Aired").alias("Aired"),
        col("Premiered").alias("Premiered"),
        col("Status").alias("Status"),
        col("Duration").alias("Duration")
    )
    
    # Створюємо сурогатний ключ (Anime_SK)
    window = Window.orderBy("Anime_ID")
    df_anime = df_anime.withColumn("Anime_SK", row_number().over(window))
    
    # Вибераємо колонки в правильному порядку
    dim_anime = df_anime.select(
        "Anime_SK",
        "Anime_ID",
        "Name",
        "English_Name",
        "Type",
        "Source",
        "Genres",
        "Studios",
        "Producers",
        "Avg_Score",
        "Popularity_Rank",
        "Episodes",
        "Age_Rating",
        "Rank",
        "Members",
        "Favorites",
        "Scored_By",
        "Aired",
        "Premiered",
        "Status",
        "Duration"
    )
    
    print(f"✅ Dim_Anime створено: {dim_anime.count()} аніме")
    return dim_anime


def create_dim_date(spark: SparkSession, start_date: str = "2000-01-01", end_date: str = "2025-12-31"):
    """
    Створює вимір дати (Dim_Date) з атрибутами для аналізу трендів.
    
    Args:
        spark: SparkSession
        start_date: Початкова дата (формат: 'YYYY-MM-DD')
        end_date: Кінцева дата (формат: 'YYYY-MM-DD')
        
    Returns:
        DataFrame з виміром дати
    """
    print("📊 Створення Dim_Date...")
    
    # Генеруємо послідовність дат через SQL (для Spark 3.0+)
    try:
        df_dates = spark.sql(f"""
            SELECT explode(sequence(to_date('{start_date}'), to_date('{end_date}'), interval 1 day)) as Full_Date
        """)
    except:
        # Альтернативний спосіб для старіших версій Spark
        from datetime import datetime, timedelta
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        dates = []
        current = start
        while current <= end:
            dates.append((current.date(),))
            current += timedelta(days=1)
        
        from pyspark.sql.types import StructType, StructField, DateType
        schema = StructType([StructField("Full_Date", DateType(), True)])
        df_dates = spark.createDataFrame(dates, schema)
    
    # Додаємо атрибути дати
    dim_date = df_dates.select(
        col("Full_Date"),
        year(col("Full_Date")).alias("Year"),
        quarter(col("Full_Date")).alias("Quarter"),
        month(col("Full_Date")).alias("Month"),
        date_format(col("Full_Date"), "MMMM").alias("Month_Name"),
        date_format(col("Full_Date"), "EEEE").alias("Day_of_Week"),
        # В Spark dayofweek: 1=Sunday, 7=Saturday
        when(dayofweek(col("Full_Date")).isin([1, 7]), True).otherwise(False).alias("Is_Weekend")
    )
    
    # Створюємо сурогатний ключ (Date_SK) у форматі YYYYMMDD
    dim_date = dim_date.withColumn(
        "Date_SK",
        expr("cast(date_format(Full_Date, 'yyyyMMdd') as int)")
    )
    
    # Вибераємо колонки в правильному порядку
    dim_date = dim_date.select(
        "Date_SK",
        "Full_Date",
        "Year",
        "Quarter",
        "Month",
        "Month_Name",
        "Day_of_Week",
        "Is_Weekend"
    )
    
    print(f"✅ Dim_Date створено: {dim_date.count()} дат")
    return dim_date


def create_fact_user_ratings(
    spark: SparkSession,
    users_score_path: str,
    dim_user,
    dim_anime,
    dim_date
):
    """
    Створює таблицю фактів (Fact_UserRatings) з метриками та обчисленими полями.
    
    Args:
        spark: SparkSession
        users_score_path: Шлях до файлу users-score-2023.csv
        dim_user: DataFrame з виміром користувачів
        dim_anime: DataFrame з виміром аніме
        dim_date: DataFrame з виміром дати
        
    Returns:
        DataFrame з таблицею фактів
    """
    print("📊 Створення Fact_UserRatings...")
    
    # Зчитуємо дані оцінок
    df_ratings = spark.read.csv(users_score_path, header=True, inferSchema=True)
    
    # Перейменовуємо колонки
    df_ratings = df_ratings.select(
        col("user_id").alias("User_ID"),
        col("anime_id").alias("Anime_ID"),
        col("rating").alias("User_Rating")
    )
    
    # Об'єднуємо з Dim_User для отримання User_SK
    df_ratings = df_ratings.join(
        dim_user.select("User_SK", "User_ID"),
        on="User_ID",
        how="inner"
    )
    
    # Об'єднуємо з Dim_Anime для отримання Anime_SK та Avg_Score
    df_ratings = df_ratings.join(
        dim_anime.select("Anime_SK", "Anime_ID", "Avg_Score"),
        on="Anime_ID",
        how="inner"
    )
    
    # Обчислюємо метрики
    df_ratings = df_ratings.withColumn(
        "Rating_Deviation",
        col("User_Rating") - col("Avg_Score")
    )
    
    df_ratings = df_ratings.withColumn(
        "Is_Above_Average",
        when(col("User_Rating") > col("Avg_Score"), 1).otherwise(0)
    )
    
    df_ratings = df_ratings.withColumn(
        "Is_High_Rating",
        when(col("User_Rating") >= 8, 1).otherwise(0)
    )
    
    df_ratings = df_ratings.withColumn(
        "Is_Low_Rating",
        when(col("User_Rating") <= 4, 1).otherwise(0)
    )
    
    df_ratings = df_ratings.withColumn("Rating_Count", lit(1))
    
    # Для Date_SK використовуємо поточну дату (або можна додати дату з іншого джерела)
    # Поки що використовуємо дату за замовчуванням (сьогодні)
    from datetime import datetime
    today = datetime.now().strftime("%Y%m%d")
    df_ratings = df_ratings.withColumn("Date_SK", lit(int(today)))
    
    # Вибераємо фінальні колонки
    fact_ratings = df_ratings.select(
        "User_SK",
        "Anime_SK",
        "Date_SK",
        "User_Rating",
        "Rating_Deviation",
        "Is_Above_Average",
        "Is_High_Rating",
        "Is_Low_Rating",
        "Rating_Count"
    )
    
    print(f"✅ Fact_UserRatings створено: {fact_ratings.count()} оцінок")
    return fact_ratings


def create_star_schema(spark: SparkSession, data_path: str = "data"):
    """
    Створює повну зірчасту схему даних.
    
    Args:
        spark: SparkSession
        data_path: Шлях до папки з даними
        
    Returns:
        Tuple з (dim_user, dim_anime, dim_date, fact_ratings)
    """
    print("🌟 Створення зірчастої схеми даних...\n")
    
    # Створюємо виміри
    dim_user = create_dim_user(spark, f"{data_path}/users-details-2023.csv")
    dim_anime = create_dim_anime(spark, f"{data_path}/anime-dataset-2023.csv")
    dim_date = create_dim_date(spark)
    
    # Створюємо таблицю фактів
    fact_ratings = create_fact_user_ratings(
        spark,
        f"{data_path}/users-score-2023.csv",
        dim_user,
        dim_anime,
        dim_date
    )
    
    print("\n✅ Зірчаста схема успішно створена!")
    print("\n📋 Структура схеми:")
    print(f"  - Dim_User: {dim_user.count()} рядків")
    print(f"  - Dim_Anime: {dim_anime.count()} рядків")
    print(f"  - Dim_Date: {dim_date.count()} рядків")
    print(f"  - Fact_UserRatings: {fact_ratings.count()} рядків")
    
    return dim_user, dim_anime, dim_date, fact_ratings


def save_star_schema_to_parquet(
    dim_user,
    dim_anime,
    dim_date,
    fact_ratings,
    output_path: str = "data/star_schema"
):
    """
    Зберігає зірчасту схему у форматі Parquet для швидшого доступу.
    
    Args:
        dim_user: DataFrame з виміром користувачів
        dim_anime: DataFrame з виміром аніме
        dim_date: DataFrame з виміром дати
        fact_ratings: DataFrame з таблицею фактів
        output_path: Шлях для збереження
    """
    print(f"\n💾 Збереження зірчастої схеми у Parquet форматі в {output_path}...")
    
    dim_user.write.mode("overwrite").parquet(f"{output_path}/dim_user")
    dim_anime.write.mode("overwrite").parquet(f"{output_path}/dim_anime")
    dim_date.write.mode("overwrite").parquet(f"{output_path}/dim_date")
    fact_ratings.write.mode("overwrite").parquet(f"{output_path}/fact_user_ratings")
    
    print("✅ Зірчаста схема збережена у Parquet форматі!")


def load_star_schema_from_parquet(spark: SparkSession, parquet_path: str = "data/star_schema"):
    """
    Завантажує зірчасту схему з Parquet файлів.
    
    Args:
        spark: SparkSession
        parquet_path: Шлях до папки з Parquet файлами
        
    Returns:
        Tuple з (dim_user, dim_anime, dim_date, fact_ratings)
    """
    print(f"📂 Завантаження зірчастої схеми з Parquet файлів з {parquet_path}...")
    
    dim_user = spark.read.parquet(f"{parquet_path}/dim_user")
    dim_anime = spark.read.parquet(f"{parquet_path}/dim_anime")
    dim_date = spark.read.parquet(f"{parquet_path}/dim_date")
    fact_ratings = spark.read.parquet(f"{parquet_path}/fact_user_ratings")
    
    print("✅ Зірчаста схема завантажена з Parquet файлів!")
    
    return dim_user, dim_anime, dim_date, fact_ratings
