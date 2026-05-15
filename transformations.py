from pyspark.sql import DataFrame
import pyspark.sql.functions as F
from pyspark.sql.window import Window


def get_joined(trip_df: DataFrame, fare_df: DataFrame) -> DataFrame:
    return trip_df.join(fare_df, on=["medallion", "hack_license", "vendor_id", "pickup_datetime"], how="inner")


def erik_q1(joined_df: DataFrame) -> DataFrame:
    return joined_df.withColumn("day", F.dayofweek("pickup_datetime")) \
        .filter(F.col("day").isin([1, 7])) \
        .withColumn("hour", F.hour("pickup_datetime")) \
        .groupBy("hour").count().orderBy(F.col("count").desc())


def erik_q2(joined_df: DataFrame) -> DataFrame:
    return joined_df.groupBy("passenger_count").agg(F.avg("total_amount").alias("avg_total")).orderBy("passenger_count")


def erik_q3(joined_df: DataFrame) -> DataFrame:
    df = joined_df.filter(F.col("total_amount") > 0).withColumn("day", F.dayofweek("pickup_datetime"))
    win = Window.partitionBy("day").orderBy(F.col("trip_distance").desc())
    return df.withColumn("rank", F.dense_rank().over(win)).filter(F.col("rank") <= 3).drop("rank")


def erik_q4(joined_df: DataFrame) -> DataFrame:
    df = joined_df.withColumn("date", F.to_date("pickup_datetime"))
    win = Window.partitionBy("date")
    return df.withColumn("daily_avg_time", F.avg("trip_time_in_secs").over(win)) \
        .withColumn("deviation", F.col("trip_time_in_secs") - F.col("daily_avg_time")) \
        .select("medallion", "date", "trip_time_in_secs", "daily_avg_time", "deviation")


def erik_q5(joined_df: DataFrame) -> DataFrame:
    return joined_df.filter(F.col("tip_amount") > F.col("fare_amount"))


def erik_q6(joined_df: DataFrame) -> DataFrame:
    hourly = joined_df.withColumn("hour", F.hour("pickup_datetime")) \
        .groupBy("vendor_id", "hour").agg(F.sum("total_amount").alias("total_revenue"))
    win = Window.partitionBy("vendor_id").orderBy(F.col("total_revenue").desc())
    return hourly.withColumn("rank", F.dense_rank().over(win)).filter(F.col("rank") == 1).drop("rank")


def denys_q1(joined_df: DataFrame) -> DataFrame:
    return joined_df.groupBy("payment_type").agg(
        F.sum("total_amount").alias("total_revenue"),
        F.avg("tip_amount").alias("avg_tip")
    )


def denys_q2(joined_df: DataFrame) -> DataFrame:
    df = joined_df.filter(F.col("tip_amount") == 0).groupBy("hack_license").count()
    win = Window.orderBy(F.col("count").desc())
    return df.withColumn("rank", F.dense_rank().over(win)).filter(F.col("rank") <= 10).drop("rank")


def denys_q3(joined_df: DataFrame) -> DataFrame:
    return joined_df.filter(F.col("trip_distance") > 5) \
        .withColumn("cost_per_mile", F.col("total_amount") / F.col("trip_distance")) \
        .groupBy("vendor_id").agg(F.avg("cost_per_mile").alias("avg_cost_per_mile"))


def denys_q4(joined_df: DataFrame) -> DataFrame:
    target_driver = joined_df.limit(1).select("hack_license").collect()[0][0]
    df = joined_df.filter(F.col("hack_license") == target_driver).withColumn("date", F.to_date("pickup_datetime"))
    win = Window.partitionBy("date").orderBy("pickup_datetime").rowsBetween(Window.unboundedPreceding, Window.currentRow)
    return df.withColumn("running_total", F.sum("total_amount").over(win)).select("pickup_datetime", "total_amount", "running_total")


def denys_q5(joined_df: DataFrame) -> DataFrame:
    return joined_df.filter((F.col("trip_distance") < 1) & (F.col("total_amount") > 50))


def denys_q6(joined_df: DataFrame) -> DataFrame:
    daily = joined_df.withColumn("date", F.to_date("pickup_datetime")).groupBy("date").agg(F.avg("total_amount").alias("avg_daily"))
    win = Window.orderBy("date")
    return daily.withColumn("prev_day_avg", F.lag("avg_daily").over(win)) \
        .withColumn("diff", F.col("avg_daily") - F.col("prev_day_avg"))


def mark_q1(joined_df: DataFrame) -> DataFrame:
    return joined_df.withColumn("has_tolls", F.when(F.col("tolls_amount") > 0, "Yes").otherwise("No")) \
        .groupBy("has_tolls").agg(F.avg("tip_amount").alias("avg_tip"))


def mark_q2(joined_df: DataFrame) -> DataFrame:
    df = joined_df.filter(F.col("trip_time_in_secs") > 900) \
        .withColumn("speed_mph", F.col("trip_distance") / (F.col("trip_time_in_secs") / 3600))
    win = Window.orderBy(F.col("speed_mph").desc())
    return df.withColumn("rank", F.dense_rank().over(win)).filter(F.col("rank") <= 10).drop("rank")


def mark_q3(joined_df: DataFrame) -> DataFrame:
    return joined_df.withColumn("dist_cat",
        F.when(F.col("trip_distance") < 2, "Short")
        .when((F.col("trip_distance") >= 2) & (F.col("trip_distance") <= 10), "Medium")
        .otherwise("Long")
    ).groupBy("dist_cat").count()


def mark_q4(joined_df: DataFrame) -> DataFrame:
    df = joined_df.withColumn("dist_cat",
        F.when(F.col("trip_distance") < 2, "Short")
        .when((F.col("trip_distance") >= 2) & (F.col("trip_distance") <= 10), "Medium")
        .otherwise("Long")
    )
    win = Window.partitionBy("dist_cat").orderBy(F.col("total_amount").desc())
    return df.withColumn("rank", F.dense_rank().over(win)).filter(F.col("rank") == 1).drop("rank")


def mark_q5(joined_df: DataFrame) -> DataFrame:
    return joined_df.filter((F.col("trip_time_in_secs") > 5400) & (F.col("trip_distance") < 2))


def mark_q6(joined_df: DataFrame) -> DataFrame:
    daily = joined_df.withColumn("date", F.to_date("pickup_datetime"))
    crd_counts = daily.filter(F.col("payment_type") == "CRD").groupBy("date").count().withColumnRenamed("count", "crd_count")
    total_counts = daily.groupBy("date").count().withColumnRenamed("count", "total_count")
    joined = total_counts.join(crd_counts, on="date")
    df = joined.withColumn("crd_percent", (F.col("crd_count") / F.col("total_count")) * 100)
    win = Window.orderBy(F.col("crd_percent").desc())
    return df.withColumn("rank", F.dense_rank().over(win))


def sanya_q1(joined_df: DataFrame) -> DataFrame:
    daily = joined_df.withColumn("date", F.to_date("pickup_datetime")).groupBy("medallion", "date").agg(
        F.count("*").alias("trips"),
        F.sum("trip_distance").alias("sum_dist"),
        F.sum("total_amount").alias("sum_rev")
    ).filter(F.col("trips") > 30)
    return daily.groupBy("medallion").agg(
        F.avg("sum_dist").alias("avg_daily_dist"),
        F.avg("sum_rev").alias("avg_daily_rev")
    )


def sanya_q2(joined_df: DataFrame) -> DataFrame:
    top_drivers = joined_df.groupBy("hack_license").agg(F.sum("total_amount").alias("tot")).orderBy(F.col("tot").desc()).limit(5)
    driver_list = [row["hack_license"] for row in top_drivers.collect()]
    df = joined_df.filter(F.col("hack_license").isin(driver_list))
    win = Window.partitionBy("hack_license").orderBy(F.col("total_amount").desc())
    return df.withColumn("rank", F.dense_rank().over(win)).filter(F.col("rank") <= 3).drop("rank")


def sanya_q3(joined_df: DataFrame) -> DataFrame:
    return joined_df.filter((F.col("surcharge") + F.col("mta_tax")) > (0.3 * F.col("fare_amount")))


def sanya_q4(joined_df: DataFrame) -> DataFrame:
    hourly = joined_df.withColumn("time_hr", F.date_trunc("hour", "pickup_datetime")).groupBy("vendor_id", "time_hr").count()
    win = Window.partitionBy("vendor_id").orderBy("time_hr").rowsBetween(-2, 0)
    return hourly.withColumn("moving_avg_3h", F.avg("count").over(win))


def sanya_q5(joined_df: DataFrame) -> DataFrame:
    return joined_df.withColumn("date", F.to_date("pickup_datetime")).groupBy("hack_license", "date") \
        .agg(F.countDistinct("medallion").alias("unique_cars")).filter(F.col("unique_cars") > 1)


def sanya_q6(joined_df: DataFrame) -> DataFrame:
    daily = joined_df.withColumn("date", F.to_date("pickup_datetime")).groupBy("hack_license", "date").agg(F.avg("total_amount").alias("avg_earn"))
    win = Window.partitionBy("hack_license").orderBy("date")
    df = daily.withColumn("lag_1", F.lag("avg_earn", 1).over(win)).withColumn("lag_2", F.lag("avg_earn", 2).over(win))
    return df.filter((F.col("avg_earn") > F.col("lag_1")) & (F.col("lag_1") > F.col("lag_2"))).select("hack_license").distinct()
