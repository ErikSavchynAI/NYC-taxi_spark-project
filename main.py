import os
import time
import logging
import contextlib
from tqdm import tqdm
from pyspark.sql import SparkSession
from extractor import get_trip_data_schema, get_fare_data_schema, load_data
from preprocessor import preprocess_data
import transformations as T
import visualizer as V

os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def save_result(df, name):
    output_path = f"data/results/{name}"
    df.coalesce(1).write.csv(output_path, header=True, mode="overwrite")


def execute_and_log(name, desc, df, visualize=False):
    start_time = time.time()

    os.makedirs("data/results", exist_ok=True)
    with open(f"data/results/{name}_plan.txt", "w") as f:
        with contextlib.redirect_stdout(f):
            df.explain()

    save_result(df, name)
    if visualize:
        V.plot_query(df, name)

    elapsed = time.time() - start_time
    return elapsed


def main():
    spark = SparkSession.builder \
        .appName("NYC_Taxi_2013_ETL") \
        .master("local[*]") \
        .config("spark.driver.memory", "3g") \
        .config("spark.sql.shuffle.partitions", "16") \
        .config("spark.memory.fraction", "0.8") \
        .config("spark.sql.files.maxPartitionBytes", "67108864") \
        .config("spark.ui.showConsoleProgress", "true") \
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

    logger.info("Кешування даних у пам'ять...")
    trip_df_cleaned.cache()
    fare_df_cleaned.cache()

    trip_count = trip_df_cleaned.count()
    fare_count = fare_df_cleaned.count()
    logger.info(f"Дані закешовано. Рядків Trip: {trip_count}, Fare: {fare_count}")

    queries = [
        ("q1_avg_tip", "Q1: Чайові від пасажирів", T.get_avg_tip_by_passenger_count(trip_df_cleaned, fare_df_cleaned),
         True),
        ("q2_no_tip", "Q2: Найдовші без чайових", T.get_longest_trips_no_tip_crd(trip_df_cleaned, fare_df_cleaned),
         False),
        ("q3_revenue_vendor", "Q3: Дохід за постачальником", T.get_total_revenue_by_vendor(fare_df_cleaned), True),
        ("q4_daily_ma", "Q4: Щоденний дохід та ковзне", T.get_daily_revenue_moving_avg(fare_df_cleaned), True),
        ("q5_top3_vendor", "Q5: Топ 3 дорогі поїздки", T.get_top_3_expensive_trips_per_vendor(fare_df_cleaned), False),
        ("q6_fast_exp", "Q6: Швидкі та дорогі", T.get_fast_expensive_trips(trip_df_cleaned, fare_df_cleaned), False)
    ]

    logger.info("Виконання трансформацій...")

    for name, desc, df, vis in tqdm(queries, desc="Прогрес запитів", unit="запит"):
        elapsed = execute_and_log(name, desc, df, vis)
        logger.info(f"Завершено: {desc} | {elapsed:.2f} с")

    spark.stop()


if __name__ == "__main__":
    main()
