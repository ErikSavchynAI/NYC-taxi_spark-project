from pyspark.sql import DataFrame
import pyspark.sql.functions as F
from pyspark.sql.window import Window


def get_avg_tip_by_passenger_count(trip_df: DataFrame, fare_df: DataFrame) -> DataFrame:
    join_cond = ["medallion", "hack_license", "vendor_id", "pickup_datetime"]
    joined_df = trip_df.join(fare_df, on=join_cond, how="inner")
    return joined_df.groupBy("passenger_count").agg(F.avg("tip_amount").alias("avg_tip")).orderBy("passenger_count")


def get_longest_trips_no_tip_crd(trip_df: DataFrame, fare_df: DataFrame) -> DataFrame:
    join_cond = ["medallion", "hack_license", "vendor_id", "pickup_datetime"]
    joined_df = trip_df.join(fare_df, on=join_cond, how="inner")
    return joined_df.filter((F.col("payment_type") == "CRD") & (F.col("tip_amount") == 0)).orderBy(F.col("trip_distance").desc())


def get_total_revenue_by_vendor(fare_df: DataFrame) -> DataFrame:
    return fare_df.groupBy("vendor_id").agg(F.sum("total_amount").alias("total_revenue"))


def get_daily_revenue_moving_avg(fare_df: DataFrame) -> DataFrame:
    daily_df = fare_df.withColumn("date", F.to_date("pickup_datetime")).groupBy("date").agg(F.sum("total_amount").alias("daily_revenue"))
    window_spec = Window.orderBy("date").rowsBetween(-2, Window.currentRow)
    return daily_df.withColumn("3_day_moving_avg", F.avg("daily_revenue").over(window_spec))


def get_top_3_expensive_trips_per_vendor(fare_df: DataFrame) -> DataFrame:
    window_spec = Window.partitionBy("vendor_id").orderBy(F.col("total_amount").desc())
    ranked_df = fare_df.withColumn("rank", F.dense_rank().over(window_spec))
    return ranked_df.filter(F.col("rank") <= 3).drop("rank")


def get_fast_expensive_trips(trip_df: DataFrame, fare_df: DataFrame) -> DataFrame:
    join_cond = ["medallion", "hack_license", "vendor_id", "pickup_datetime"]
    joined_df = trip_df.join(fare_df, on=join_cond, how="inner")
    return joined_df.filter((F.col("trip_time_in_secs") < 600) & (F.col("total_amount") > 50))
