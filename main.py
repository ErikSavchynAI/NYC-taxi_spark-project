from pyspark.sql import SparkSession
from extractor import get_trip_data_schema, get_fare_data_schema, load_data


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

    spark.stop()


if __name__ == "__main__":
    main()
