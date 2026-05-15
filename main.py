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

    return time.time() - start_time


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

    trip_df = load_data(spark, "data/raw/tripData2013/", get_trip_data_schema())
    fare_df = load_data(spark, "data/raw/faredata2013/", get_fare_data_schema())

    trip_df_cleaned = preprocess_data(trip_df, "Trip Data")
    fare_df_cleaned = preprocess_data(fare_df, "Fare Data")

    logger.info("Кешування даних у пам'ять...")
    trip_df_cleaned.cache()
    fare_df_cleaned.cache()

    logger.info(f"Дані закешовано. Рядків Trip: {trip_df_cleaned.count()}, Fare: {fare_df_cleaned.count()}")

    joined_df = T.get_joined(trip_df_cleaned, fare_df_cleaned)
    joined_df.cache()
    joined_df.count()

    queries = [
        ("erik_q1", "Erik: Завантажені години вихідних", T.erik_q1(joined_df), True),
        ("erik_q2", "Erik: Вартість від пасажирів", T.erik_q2(joined_df), True),
        ("erik_q3", "Erik: Топ 3 довгі поїздки за днями", T.erik_q3(joined_df), False),
        ("erik_q4", "Erik: Відхилення часу від середнього", T.erik_q4(joined_df), False),
        ("erik_q5", "Erik: Чайові > Тариф", T.erik_q5(joined_df), False),
        ("erik_q6", "Erik: Макс дохід провайдера по годинах", T.erik_q6(joined_df), True),

        ("denys_q1", "Denys: Дохід та чайові за типом оплати", T.denys_q1(joined_df), True),
        ("denys_q2", "Denys: Топ 10 водіїв без чайових", T.denys_q2(joined_df), True),
        ("denys_q3", "Denys: Вартість милі (>5 миль)", T.denys_q3(joined_df), True),
        ("denys_q4", "Denys: Накопичувальний заробіток", T.denys_q4(joined_df), True),
        ("denys_q5", "Denys: Аномалії (<1 милі, >$50)", T.denys_q5(joined_df), False),
        ("denys_q6", "Denys: Динаміка середнього чеку", T.denys_q6(joined_df), True),

        ("mark_q1", "Markiyan: Чайові (Платні vs Безкоштовні)", T.mark_q1(joined_df), True),
        ("mark_q2", "Markiyan: Топ 10 найшвидших (>15 хв)", T.mark_q2(joined_df), False),
        ("mark_q3", "Markiyan: Категорії дистанції", T.mark_q3(joined_df), True),
        ("mark_q4", "Markiyan: Макс вартість за категорією", T.mark_q4(joined_df), False),
        ("mark_q5", "Markiyan: Затори (>1.5 год, <2 милі)", T.mark_q5(joined_df), False),
        ("mark_q6", "Markiyan: Ранжування днів за оплатою карткою", T.mark_q6(joined_df), True),

        ("sanya_q1", "Sanya: Авто >30 поїздок", T.sanya_q1(joined_df), True),
        ("sanya_q2", "Sanya: Топ 3 поїздки для топ 5 водіїв", T.sanya_q2(joined_df), False),
        ("sanya_q3", "Sanya: Збори > 30% тарифу", T.sanya_q3(joined_df), False),
        ("sanya_q4", "Sanya: Ковзне середнє поїздок (3 год)", T.sanya_q4(joined_df), True),
        ("sanya_q5", "Sanya: Водії на різних авто в один день", T.sanya_q5(joined_df), False),
        ("sanya_q6", "Sanya: Безперервне зростання заробітку 3 дні", T.sanya_q6(joined_df), False)
    ]

    logger.info("Виконання 24 трансформацій...")

    for name, desc, df, vis in tqdm(queries, desc="Прогрес", unit="запит"):
        elapsed = execute_and_log(name, desc, df, vis)
        logger.info(f"Завершено: {desc} | {elapsed:.2f} с")

    spark.stop()


if __name__ == "__main__":
    main()
