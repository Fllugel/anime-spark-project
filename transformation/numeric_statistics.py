"""
Numeric Statistics Module

This module provides functions to calculate and analyze statistics
for numeric columns in the anime dataset.
"""

import json
import os
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, mean, stddev, min as spark_min, max as spark_max, count, isnull
from pyspark.sql.types import NumericType


def get_numeric_columns(df: DataFrame) -> list:
    """
    Отримує список числових стовпців у DataFrame.

    Args:
        df: Spark DataFrame

    Returns:
        Список назв числових стовпців
    """
    numeric_cols = []
    for field in df.schema.fields:
        if isinstance(field.dataType, NumericType):
            numeric_cols.append(field.name)
    return numeric_cols


def get_numeric_statistics(df: DataFrame) -> dict:
    """
    Отримує статистику щодо числових стовпців.

    Args:
        df: Spark DataFrame з аніме даними

    Returns:
        Dict зі статистикою для кожного числового стовпця:
        - count: кількість ненульових значень
        - mean: середнє значення
        - stddev: стандартне відхилення
        - min: мінімальне значення
        - max: максимальне значення
    """
    print("\n" + "="*60)
    print("ЕТАП 2: Отримання статистики щодо числових стовпців")
    print("="*60)
    
    numeric_cols = get_numeric_columns(df)
    
    if not numeric_cols:
        print("⚠️  Числові стовпці не знайдені!")
        return {}
    
    print(f"\nЗнайдено {len(numeric_cols)} числових стовпців: {', '.join(numeric_cols)}")
    
    statistics = {}
    
    for col_name in numeric_cols:
        print(f"\n📊 Статистика для стовпця '{col_name}':")
        
        # Отримуємо базову статистику
        stats_df = df.select(
            count(col(col_name)).alias('count'),
            mean(col(col_name)).alias('mean'),
            stddev(col(col_name)).alias('stddev'),
            spark_min(col(col_name)).alias('min'),
            spark_max(col(col_name)).alias('max')
        )
        
        stats_row = stats_df.collect()[0]
        
        col_stats = {
            'count': stats_row['count'],
            'mean': float(stats_row['mean']) if stats_row['mean'] is not None else None,
            'stddev': float(stats_row['stddev']) if stats_row['stddev'] is not None else None,
            'min': float(stats_row['min']) if stats_row['min'] is not None else None,
            'max': float(stats_row['max']) if stats_row['max'] is not None else None
        }
        
        # Підраховуємо null значення
        null_count = df.filter(isnull(col(col_name)) | col(col_name).isNull()).count()
        total_count = df.count()
        col_stats['null_count'] = null_count
        col_stats['null_percentage'] = (null_count / total_count * 100) if total_count > 0 else 0
        
        statistics[col_name] = col_stats
        
        # Виводимо статистику
        print(f"  • Кількість значень: {col_stats['count']:,}")
        print(f"  • Null значень: {col_stats['null_count']:,} ({col_stats['null_percentage']:.2f}%)")
        if col_stats['mean'] is not None:
            print(f"  • Середнє значення: {col_stats['mean']:.2f}")
        if col_stats['stddev'] is not None:
            print(f"  • Стандартне відхилення: {col_stats['stddev']:.2f}")
        if col_stats['min'] is not None:
            print(f"  • Мінімальне значення: {col_stats['min']:.2f}")
        if col_stats['max'] is not None:
            print(f"  • Максимальне значення: {col_stats['max']:.2f}")
    
    return statistics


def analyze_numeric_columns(df: DataFrame, statistics: dict = None) -> dict:
    """
    Проводить аналіз отриманої інформації про числові стовпці.

    Args:
        df: Spark DataFrame з аніме даними
        statistics: Dict зі статистикою (якщо None, буде обчислено)

    Returns:
        Dict з аналізом числових стовпців
    """
    if statistics is None:
        statistics = get_numeric_statistics(df)
    
    print("\n" + "="*60)
    print("АНАЛІЗ ЧИСЛОВИХ СТОВПЦІВ")
    print("="*60)
    
    analysis = {}
    
    for col_name, stats in statistics.items():
        print(f"\n🔍 Аналіз стовпця '{col_name}':")
        
        col_analysis = {
            'data_quality': {},
            'distribution': {},
            'insights': []
        }
        
        # Аналіз якості даних
        null_pct = stats['null_percentage']
        if null_pct == 0:
            quality = "Відмінна - немає пропущених значень"
        elif null_pct < 5:
            quality = "Добра - мінімальна кількість пропущених значень"
        elif null_pct < 20:
            quality = "Прийнятна - помірна кількість пропущених значень"
        else:
            quality = "Погана - багато пропущених значень"
        
        col_analysis['data_quality']['null_percentage'] = null_pct
        col_analysis['data_quality']['assessment'] = quality
        print(f"  • Якість даних: {quality} ({null_pct:.2f}% null)")
        
        # Аналіз розподілу
        if stats['mean'] is not None and stats['stddev'] is not None and stats['stddev'] > 0:
            cv = (stats['stddev'] / stats['mean']) * 100  # коефіцієнт варіації
            col_analysis['distribution']['coefficient_of_variation'] = cv
            
            if cv < 15:
                dist_assessment = "Низька варіативність - дані досить однорідні"
            elif cv < 35:
                dist_assessment = "Помірна варіативність - дані мають середню різноманітність"
            else:
                dist_assessment = "Висока варіативність - дані дуже різноманітні"
            
            col_analysis['distribution']['assessment'] = dist_assessment
            print(f"  • Розподіл: {dist_assessment} (CV: {cv:.2f}%)")
        
        # Діапазон значень
        if stats['min'] is not None and stats['max'] is not None:
            range_val = stats['max'] - stats['min']
            col_analysis['distribution']['range'] = range_val
            col_analysis['distribution']['min'] = stats['min']
            col_analysis['distribution']['max'] = stats['max']
            print(f"  • Діапазон: від {stats['min']:.2f} до {stats['max']:.2f} (розмах: {range_val:.2f})")
        
        # Інсайти
        insights = []
        if stats['mean'] is not None:
            insights.append(f"Середнє значення: {stats['mean']:.2f}")
        if stats['stddev'] is not None and stats['stddev'] > 0:
            insights.append(f"Стандартне відхилення: {stats['stddev']:.2f}")
        
        col_analysis['insights'] = insights
        analysis[col_name] = col_analysis
    
    return analysis


def save_numeric_statistics(statistics: dict, analysis: dict, output_dir: str = "output/results"):
    """
    Зберігає статистику та аналіз числових стовпців у файли.

    Args:
        statistics: Dict зі статистикою
        analysis: Dict з аналізом
        output_dir: Директорія для збереження результатів
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Зберігаємо JSON з повною інформацією
    json_path = os.path.join(output_dir, "numeric_statistics.json")
    json_data = {
        'statistics': statistics,
        'analysis': analysis
    }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Статистика збережена у JSON: {json_path}")
    
    # Зберігаємо CSV з базовою статистикою
    csv_path = os.path.join(output_dir, "numeric_statistics.csv")
    import csv
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Column', 'Count', 'Mean', 'StdDev', 'Min', 'Max', 'Null Count', 'Null %'])
        
        for col_name, stats in statistics.items():
            writer.writerow([
                col_name,
                stats['count'],
                stats['mean'] if stats['mean'] is not None else '',
                stats['stddev'] if stats['stddev'] is not None else '',
                stats['min'] if stats['min'] is not None else '',
                stats['max'] if stats['max'] is not None else '',
                stats['null_count'],
                f"{stats['null_percentage']:.2f}"
            ])
    
    print(f"✅ Статистика збережена у CSV: {csv_path}")


def run_numeric_statistics_analysis(df: DataFrame, output_dir: str = "output/results") -> tuple:
    """
    Запускає повний аналіз статистики числових стовпців.

    Args:
        df: Spark DataFrame з аніме даними
        output_dir: Директорія для збереження результатів

    Returns:
        Tuple (statistics, analysis)
    """
    statistics = get_numeric_statistics(df)
    analysis = analyze_numeric_columns(df, statistics)
    save_numeric_statistics(statistics, analysis, output_dir)
    return statistics, analysis

