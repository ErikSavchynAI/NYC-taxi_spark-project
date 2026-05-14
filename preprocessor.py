from pyspark.sql import DataFrame
import pyspark.sql.functions as F


def preprocess_data(df: DataFrame, dataset_name: str) -> DataFrame:
    for col_name in df.columns:
        df = df.withColumnRenamed(col_name, col_name.strip())

    cols_to_drop = [
        "store_and_fwd_flag",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude"
    ]
    df = df.drop(*[c for c in cols_to_drop if c in df.columns])

    df = df.dropDuplicates()

    if "trip_distance" in df.columns:
        df = df.filter(
            (F.col("trip_distance") > 0) &
            (F.col("trip_time_in_secs") > 0) &
            (F.col("trip_time_in_secs") < 18000) &
            (F.col("passenger_count") > 0)
        )

    if "total_amount" in df.columns:
        df = df.filter(F.col("total_amount") > 0)

    return df
