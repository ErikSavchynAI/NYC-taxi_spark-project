from pyspark.sql import SparkSession
from extractor import get_trip_data_schema, get_fare_data_schema, load_data
from preprocessor import show_basic_statistics, preprocess_data

def main():
    print("Ініціалізація Spark...")
    spark = SparkSession.builder \
        .appName("NYC_Taxi_2013_ETL") \
        .master("local[*]") \
        .getOrCreate()

    trip_data_path = "data/raw/tripData2013/"
    fare_data_path = "data/raw/faredata2013/"

    print("Зчитування даних про поїздки (Trip Data)...")
    trip_df = load_data(spark, trip_data_path, get_trip_data_schema())

    print("Зчитування фінансових даних (Fare Data)...")
    fare_df = load_data(spark, fare_data_path, get_fare_data_schema())

    print("\n--- Схема Trip Data ---")
    trip_df.printSchema()

    print("\n--- Перші 5 записів Trip Data ---")
    trip_df.show(5)

    print("\n--- Перші 5 записів Fare Data ---")
    fare_df.show(5)


    show_basic_statistics(trip_df, "Trip Data")
    show_basic_statistics(fare_df, "Fare Data")


    print("Попередня обробка для зрізу 50тис.")
    trip_df_test = trip_df.limit(50000)
    fare_df_test = fare_df.limit(50000)

    # Запускаємо очищення для тестових даних
    trip_df_cleaned = preprocess_data(trip_df_test, "Trip Data")
    fare_df_cleaned = preprocess_data(fare_df_test, "Fare Data")

    # ПЕРЕВІРКА: показуємо статистику ПІСЛЯ очищення
    # Тепер min у trip_distance має бути > 0, а max у часі — адекватним
    print("\n--- СТАТИСТИКА ПІСЛЯ ОЧИЩЕННЯ ")
    trip_df_cleaned.select("passenger_count", "trip_time_in_secs", "trip_distance").summary().show()

    print("\n--- СТАТИСТИКА Fare Data ПІСЛЯ ОЧИЩЕННЯ ")
    fare_df_cleaned.select("fare_amount", "tip_amount", "total_amount").summary().show()

    spark.stop()
if __name__ == "__main__":
    main()
