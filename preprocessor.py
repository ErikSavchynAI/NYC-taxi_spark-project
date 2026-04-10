from pyspark.sql import DataFrame
import pyspark.sql.functions as F


def show_basic_statistics(df: DataFrame, dataset_name: str):
    print(f"\n=== АНАЛІЗ: {dataset_name} ===")

    # total_fares = fare_df.count()
    # print(f"Загальна кількість записів у Fare Data: {total_fares:,}")

    print(f"Загальна кількість записів у кожному наборі: 173,179,759 (прочитано з логів)")

    print("Створюємо вибірку для аналізу (0.1% даних)...")
    sample_df = df.sample(withReplacement=False, fraction=0.001, seed=42)

    # Аналіз числових колонки
    if "trip_distance" in df.columns:  # Trip Data
        num_cols = ["passenger_count", "trip_time_in_secs", "trip_distance"]
    else:  # Fare Data
        num_cols = ["fare_amount", "tip_amount", "total_amount"]

    print(f"Статистика числових ознак {dataset_name}:")
    sample_df.select(num_cols).summary().show()

    # ПЕРЕВІРКА НА ПРОПУСКИ (на вибірці)
    # print("Перевірка на наявність пропущених значень (Null)...")
    # for col in sample_df.columns:
    #     null_count = sample_df.filter(F.col(col).isNull()).count()
    #     if null_count > 0:
    #         print(f" У колонці '{col}' знайдено {null_count} пропусків (у вибірці).")
    #     else:
    #         print(f" Колонка '{col}' не має пропусків.")

    sample_df.unpersist()


def preprocess_data(df: DataFrame, dataset_name: str) -> DataFrame:
    print(f"\n--- ЕТАП ПОПЕРЕДНЬОЇ ОБРОБКИ: {dataset_name} ---")

    for col_name in df.columns:
        df = df.withColumnRenamed(col_name, col_name.strip())

    # Видалення неінформативних колонок (технічний прапор та координати)
    cols_to_drop = ["store_and_fwd_flag", "pickup_longitude", "pickup_latitude", "dropoff_longitude",
                    "dropoff_latitude"]
    df = df.drop(*[c for c in cols_to_drop if c in df.columns])
    print(f"  - Вилучено неінформативні колонки ")


    print("  - Видалення дублікатів та логічне очищення ")


    df = df.dropDuplicates()

    # Фільтрація аномалій (на основі нашого аналізу summary)
    if "trip_distance" in df.columns:
        df = df.filter(
            (F.col("trip_distance") > 0) &
            (F.col("trip_time_in_secs") > 0) &
            (F.col("trip_time_in_secs") < 18000) &
            (F.col("passenger_count") > 0)
        )

    if "total_amount" in df.columns:
        df = df.filter(F.col("total_amount") > 0)

    print(f"  - Очищення {dataset_name} завершено успішно.")
    return df