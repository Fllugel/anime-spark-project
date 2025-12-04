"""
Dataset Information Module

This module provides functions to extract and describe general information
about the anime dataset including schema, row count, column count, and column names.
"""

import json
import os
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, isnull, count as spark_count


def get_dataset_info(df: DataFrame) -> dict:
    """
    Отримує загальну інформацію про набір даних.

    Args:
        df: Spark DataFrame з аніме даними

    Returns:
        Dict з інформацією про набір даних:
        - row_count: кількість рядків
        - column_count: кількість стовпців
        - columns: список назв стовпців
        - schema: схема датасету
        - null_counts: кількість null значень по кожному стовпцю
    """
    print("\n" + "="*60)
    print("ЕТАП 1: Отримання загальної інформації про набір даних")
    print("="*60)
    
    # Отримуємо базову інформацію
    row_count = df.count()
    columns = df.columns
    column_count = len(columns)
    
    # Отримуємо схему
    schema_dict = []
    for field in df.schema.fields:
        schema_dict.append({
            'name': field.name,
            'type': str(field.dataType),
            'nullable': field.nullable
        })
    
    # Підраховуємо null значення для кожного стовпця
    print("\nПідрахунок null значень...")
    null_counts = {}
    for col_name in columns:
        null_count = df.filter(isnull(col(col_name))).count()
        null_counts[col_name] = null_count
    
    info = {
        'row_count': row_count,
        'column_count': column_count,
        'columns': columns,
        'schema': schema_dict,
        'null_counts': null_counts
    }
    
    return info


def describe_dataset(df: DataFrame) -> dict:
    """
    Описує набір даних використовуючи отриману інформацію.

    Args:
        df: Spark DataFrame з аніме даними

    Returns:
        Dict з описом датасету
    """
    info = get_dataset_info(df)
    
    print(f"\n📊 ОПИС НАБОРУ ДАНИХ:")
    print(f"  • Кількість рядків: {info['row_count']:,}")
    print(f"  • Кількість стовпців: {info['column_count']}")
    
    print(f"\n📋 СТОВПЦІ ({info['column_count']}):")
    for i, col_name in enumerate(info['columns'], 1):
        null_count = info['null_counts'][col_name]
        null_percentage = (null_count / info['row_count'] * 100) if info['row_count'] > 0 else 0
        print(f"  {i:2d}. {col_name:20s} - {null_count:6d} null ({null_percentage:5.2f}%)")
    
    print(f"\n📐 СХЕМА ДАТАСЕТУ:")
    for field in info['schema']:
        nullable_str = "nullable" if field['nullable'] else "not nullable"
        print(f"  • {field['name']:20s}: {field['type']:30s} ({nullable_str})")
    
    # Показуємо приклад даних
    print(f"\n🔍 ПРИКЛАД ДАНИХ (перші 3 рядки):")
    df.show(3, truncate=50)
    
    return info


def save_dataset_info(info: dict, output_dir: str = "data/results") -> str:
    """
    Зберігає інформацію про датасет у JSON файл.

    Args:
        info: Dict з інформацією про датасет
        output_dir: Директорія для збереження результатів

    Returns:
        Шлях до збереженого файлу
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "dataset_info.json")
    
    # Конвертуємо схему у JSON-сумісний формат
    json_info = {
        'row_count': info['row_count'],
        'column_count': info['column_count'],
        'columns': info['columns'],
        'schema': info['schema'],
        'null_counts': info['null_counts']
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_info, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Інформація про датасет збережена: {output_path}")
    return output_path


def run_dataset_info_analysis(df: DataFrame, output_dir: str = "data/results") -> dict:
    """
    Запускає повний аналіз інформації про датасет.

    Args:
        df: Spark DataFrame з аніме даними
        output_dir: Директорія для збереження результатів

    Returns:
        Dict з інформацією про датасет
    """
    info = describe_dataset(df)
    save_dataset_info(info, output_dir)
    return info

