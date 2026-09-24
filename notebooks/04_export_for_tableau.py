# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Export gold marts for Tableau Public
# MAGIC Tableau Public can't connect live to Databricks, so each gold table is written as one CSV and zipped.

# COMMAND ----------

import os, zipfile, shutil
CATALOG, SCHEMA = "workspace", "aa_ops"
OUT = f"/Volumes/{CATALOG}/{SCHEMA}/raw_bts/exports"
os.makedirs(OUT, exist_ok=True)
tables = ["gold_hub_daily", "gold_hub_hourly", "gold_delay_causes", "gold_turn_propagation",
          "gold_route_turn_risk", "gold_rotation_wave", "gold_carrier_benchmark"]

for t in tables:
    spark.table(f"{CATALOG}.{SCHEMA}.{t}").toPandas().to_csv(f"/tmp/{t}.csv", index=False)

with zipfile.ZipFile("/tmp/aa_hub_ops_gold.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for t in tables:
        z.write(f"/tmp/{t}.csv", f"{t}.csv")
shutil.copy("/tmp/aa_hub_ops_gold.zip", f"{OUT}/aa_hub_ops_gold.zip")
dbutils.notebook.exit(f"{OUT}/aa_hub_ops_gold.zip {os.path.getsize(OUT + '/aa_hub_ops_gold.zip')} bytes")
