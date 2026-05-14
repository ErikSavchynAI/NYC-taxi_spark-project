import os
import time
import logging
from pyspark.sql import SparkSession
from extractor import get_trip_data_schema, get_fare_data_schema, load_data
from preprocessor import show_basic_statistics, preprocess_data
import transformations as T
import visualizer as V

os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def save_result(df, name):
    output_path = f"data/results/{name}"
    df.coalesce(1).write.csv(output_path, header=True, mode="overwrite")


def execute_and_log(name, desc, df, visualize=False):
    logger.info(f"Початок: {desc}")
    start_time = time.time()
    df.explain()
    save_result(df, name)
    if visualize:
        V.plot_query(df, name)
    elapsed = time.time() - start_time
    logger.info(f"Завершено: {desc} | Час: {elapsed:.2f} с")


def main():
    spark = SparkSession.builder \
        .appName("NYC_Taxi_2013_ETL") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "8") \
        .config("spark.memory.fraction", "0.8") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    trip_data_path = "data/raw/tripData2013/"
    fare_data_path = "data/raw/faredata2013/"

    logger.info("Зчитування даних...")
    trip_df = load_data(spark, trip_data_path, get_trip_data_schema())
    fare_df = load_data(spark, fare_data_path, get_fare_data_schema())

    logger.info("Попередня обробка...")
    trip_df_cleaned = preprocess_data(trip_df, "Trip Data")
    fare_df_cleaned = preprocess_data(fare_df, "Fare Data")

    logger.info("Кешування даних у пам'ять (це найдовший етап)...")
    trip_df_cleaned.cache()
    fare_df_cleaned.cache()

    trip_count = trip_df_cleaned.count()
    fare_count = fare_df_cleaned.count()
    logger.info(f"Дані успішно закешовано. Рядків Trip: {trip_count}, Fare: {fare_count}")

    logger.info("Виконання бізнес-трансформацій...")

    q1 = T.get_avg_tip_by_passenger_count(trip_df_cleaned, fare_df_cleaned)
    execute_and_log("q1_avg_tip", "Q1: Чайові від кількості пасажирів", q1, True)

    q2 = T.get_longest_trips_no_tip_crd(trip_df_cleaned, fare_df_cleaned)
    execute_and_log("q2_no_tip", "Q2: Найдовші поїздки без чайових", q2, False)

    q3 = T.get_total_revenue_by_vendor(fare_df_cleaned)
    execute_and_log("q3_revenue_vendor", "Q3: Дохід за постачальником", q3, True)

    q4 = T.get_daily_revenue_moving_avg(fare_df_cleaned)
    execute_and_log("q4_daily_ma", "Q4: Щоденний дохід та ковзне середнє", q4, True)

    q5 = T.get_top_3_expensive_trips_per_vendor(fare_df_cleaned)
    execute_and_log("q5_top3_vendor", "Q5: Топ 3 найдорожчі поїздки", q5, False)

    q6 = T.get_fast_expensive_trips(trip_df_cleaned, fare_df_cleaned)
    execute_and_log("q6_fast_exp", "Q6: Швидкі та дорогі поїздки", q6, False)

    logger.info("Усі операції завершено.")
    spark.stop()


if __name__ == "__main__":
    main()
