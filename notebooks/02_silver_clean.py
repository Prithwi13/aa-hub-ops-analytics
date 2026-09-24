# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Bronze → Silver: clean, type, and enrich flights
# MAGIC * Keep every flight (all carriers) that departs from or arrives at the 5 AA hubs, so AA can be benchmarked against peers at the same airport.
# MAGIC * Cast types, build local timestamps (handles `2400` times, overnight and cross-time-zone arrivals), derive airport-ops KPI flags.
# MAGIC * Deduplicate, then enforce data-quality rules; rows that fail go to a quarantine table instead of being silently dropped.
# MAGIC * Reconstruct **aircraft rotations** by tail number to measure hub turn times and delay propagation.

# COMMAND ----------

from pyspark.sql import functions as F, Window as W

CATALOG, SCHEMA = "workspace", "aa_ops"
T = lambda n: f"{CATALOG}.{SCHEMA}.{n}"
HUBS = ["DFW", "CLT", "ORD", "PHL", "MIA"]
AA_FAMILY = {"AA": "American", "MQ": "Envoy", "OH": "PSA", "PT": "Piedmont", "OO": "SkyWest", "YX": "Republic"}

b = spark.table(T("bronze_flights"))
print("bronze rows:", b.count())

# COMMAND ----------

num = lambda c: F.col(c).cast("double")

hub_flights = (b
    .filter(F.col("Origin").isin(HUBS) | F.col("Dest").isin(HUBS))
    .select(
        F.to_date("FlightDate").alias("flight_date"),
        F.col("Year").cast("int").alias("year"),
        F.col("Month").cast("int").alias("month"),
        F.col("DayOfWeek").cast("int").alias("day_of_week"),       # 1=Mon … 7=Sun
        F.col("Marketing_Airline_Network").alias("mkt_carrier"),
        F.col("Operating_Airline").alias("op_carrier"),
        F.col("Flight_Number_Operating_Airline").cast("int").alias("op_flight_num"),
        F.upper(F.trim("Tail_Number")).alias("tail_number"),
        "Origin", "Dest",
        F.col("CRSDepTime").cast("int").alias("crs_dep_hhmm"),
        F.col("CRSArrTime").cast("int").alias("crs_arr_hhmm"),
        num("DepDelay").alias("dep_delay"),
        num("ArrDelay").alias("arr_delay"),
        num("TaxiOut").alias("taxi_out"),
        num("TaxiIn").alias("taxi_in"),
        (num("Cancelled") == 1).alias("cancelled"),
        F.col("CancellationCode").alias("cancel_code"),
        (num("Diverted") == 1).alias("diverted"),
        num("CRSElapsedTime").alias("crs_elapsed"),
        num("Distance").alias("distance"),
        num("CarrierDelay").alias("carrier_delay"),
        num("WeatherDelay").alias("weather_delay"),
        num("NASDelay").alias("nas_delay"),
        num("SecurityDelay").alias("security_delay"),
        num("LateAircraftDelay").alias("late_aircraft_delay"),
        "_source_file")
    .withColumnRenamed("Origin", "origin").withColumnRenamed("Dest", "dest"))

# COMMAND ----------

# Local timestamps. Scheduled arrival is in the destination's local time, so an arrival that looks
# "earlier" than departure by more than the widest U.S. time-zone gap (6h) must be the next day.
def to_ts(hhmm):
    return F.expr(f"timestampadd(MINUTE, (div({hhmm}, 100) * 60 + {hhmm} % 100), cast(flight_date as timestamp))")

s = (hub_flights
    .withColumn("sched_dep_local", to_ts("crs_dep_hhmm"))
    .withColumn("_arr0", to_ts("crs_arr_hhmm"))
    .withColumn("sched_arr_local", F.when(
        (F.col("_arr0").cast("long") - F.col("sched_dep_local").cast("long")) / 60 < F.col("crs_elapsed") - 360,
        F.col("_arr0") + F.expr("INTERVAL 1 DAY")).otherwise(F.col("_arr0")))
    .drop("_arr0")
    .withColumn("actual_dep_local", F.when(~F.col("cancelled"),
        F.expr("timestampadd(MINUTE, cast(dep_delay as int), sched_dep_local)")))
    .withColumn("actual_arr_local", F.when(~F.col("cancelled") & ~F.col("diverted"),
        F.expr("timestampadd(MINUTE, cast(arr_delay as int), sched_arr_local)")))
    # ---- airport-ops KPI flags (NULL when not applicable, so AVG() gives the rate) ----
    .withColumn("is_aa", F.col("mkt_carrier") == "AA")
    .withColumn("carrier_group", F.when(F.col("is_aa"), "American (AA + regionals)").otherwise(F.col("mkt_carrier")))
    .withColumn("d0",  F.when(~F.col("cancelled"), (F.col("dep_delay") <= 0).cast("int")))
    .withColumn("d15", F.when(~F.col("cancelled"), (F.col("dep_delay") >= 15).cast("int")))
    .withColumn("a14", F.when(~F.col("cancelled") & ~F.col("diverted"), (F.col("arr_delay") <= 14).cast("int")))
    .withColumn("dep_hour", F.floor(F.col("crs_dep_hhmm") / 100) % 24)
    .withColumn("arr_hour", F.floor(F.col("crs_arr_hhmm") / 100) % 24))

# COMMAND ----------

# Deduplicate on the natural key of an operated flight leg
key = ["flight_date", "op_carrier", "op_flight_num", "origin", "dest", "crs_dep_hhmm"]
before = s.count()
s = s.dropDuplicates(key)
dupes = before - s.count()

# Data-quality rules -> quarantine instead of silent drop
dq = (F.when(F.col("flight_date").isNull(), "missing_date")
       .when(F.col("crs_dep_hhmm").isNull() | (F.col("crs_dep_hhmm") > 2400), "bad_sched_dep")
       .when(~F.col("cancelled") & F.col("dep_delay").isNull(), "operated_but_no_dep_delay")
       .when(~F.col("cancelled") & ~F.col("diverted") & F.col("arr_delay").isNull(), "completed_but_no_arr_delay")
       .when(F.col("dep_delay") > 24 * 60, "dep_delay_over_24h")
       .when(F.col("taxi_out") > 300, "taxi_out_over_5h"))
s = s.withColumn("dq_issue", dq)

s.filter(F.col("dq_issue").isNotNull()).write.mode("overwrite").option("overwriteSchema", True).saveAsTable(T("silver_quarantine"))
clean = s.filter(F.col("dq_issue").isNull()).drop("dq_issue")
(clean.write.mode("overwrite").option("overwriteSchema", True)
      .partitionBy("year", "month").saveAsTable(T("silver_hub_flights")))

# COMMAND ----------

# MAGIC %md ## Aircraft rotations → hub turns
# MAGIC For each AA-family tail number, pair every arrival into a hub with the **same aircraft's next departure** from that hub.

# COMMAND ----------

f = spark.table(T("silver_hub_flights")).filter("is_aa AND tail_number IS NOT NULL AND tail_number NOT IN ('', 'NA')")
w = W.partitionBy("tail_number").orderBy("sched_dep_local")
nxt = lambda c: F.lead(c).over(w)

pairs = (f
    .withColumn("out_origin", nxt("origin")).withColumn("out_dest", nxt("dest"))
    .withColumn("out_carrier", nxt("op_carrier")).withColumn("out_flight_num", nxt("op_flight_num"))
    .withColumn("out_sched_dep", nxt("sched_dep_local")).withColumn("out_actual_dep", nxt("actual_dep_local"))
    .withColumn("out_dep_delay", nxt("dep_delay")).withColumn("out_d0", nxt("d0")).withColumn("out_d15", nxt("d15"))
    .withColumn("out_late_aircraft_delay", nxt("late_aircraft_delay")).withColumn("out_cancelled", nxt("cancelled"))
    .withColumn("out_dep_hour", nxt("dep_hour")))

turns = (pairs
    .filter(F.col("dest").isin(HUBS) & (F.col("out_origin") == F.col("dest")))
    .filter(~F.col("cancelled") & ~F.col("diverted") & ~F.col("out_cancelled"))
    .select(
        F.col("dest").alias("hub"), "flight_date", "day_of_week", "tail_number",
        F.col("origin").alias("in_origin"), F.col("op_carrier").alias("in_carrier"), F.col("op_flight_num").alias("in_flight_num"),
        F.col("sched_arr_local").alias("in_sched_arr"), F.col("actual_arr_local").alias("in_actual_arr"),
        F.col("arr_delay").alias("in_arr_delay"),
        "out_dest", "out_carrier", "out_flight_num", "out_sched_dep", "out_actual_dep", "out_dep_hour",
        "out_dep_delay", "out_d0", "out_d15", "out_late_aircraft_delay")
    .withColumn("sched_turn_min", (F.col("out_sched_dep").cast("long") - F.col("in_sched_arr").cast("long")) / 60)
    .withColumn("actual_turn_min", (F.col("out_actual_dep").cast("long") - F.col("in_actual_arr").cast("long")) / 60)
    # keep genuine same-visit turns: scheduled ground time between 20 min and 6 h
    .filter(F.col("sched_turn_min").between(20, 360))
    .withColumn("turn_bucket", F.when(F.col("sched_turn_min") < 45, "1: <45 min")
                                .when(F.col("sched_turn_min") < 60, "2: 45-59 min")
                                .when(F.col("sched_turn_min") < 90, "3: 60-89 min")
                                .when(F.col("sched_turn_min") < 120, "4: 90-119 min")
                                .otherwise("5: 120+ min"))
    .withColumn("in_late15", (F.col("in_arr_delay") >= 15).cast("int"))
    # delay the turn could NOT absorb: inbound lateness minus the scheduled buffer (floored at 0)
    .withColumn("slack_min", F.col("sched_turn_min") - F.greatest(F.lit(0), F.col("in_arr_delay")))
    .withColumn("propagated", ((F.col("in_arr_delay") >= 15) & (F.col("out_dep_delay") >= 15)).cast("int")))

turns.write.mode("overwrite").option("overwriteSchema", True).saveAsTable(T("silver_aircraft_turns"))

# COMMAND ----------

res = {
  "hub_flights": spark.table(T("silver_hub_flights")).count(),
  "quarantined": spark.table(T("silver_quarantine")).count(),
  "duplicates_removed": dupes,
  "aa_turns": spark.table(T("silver_aircraft_turns")).count(),
}
print(res)
import json; dbutils.notebook.exit(json.dumps(res))
