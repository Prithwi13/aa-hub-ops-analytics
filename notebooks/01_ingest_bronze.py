# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Ingest → Bronze
# MAGIC Downloads 24 months of the BTS **Marketing Carrier On-Time Performance** files straight from
# MAGIC transtats.bts.gov into a Unity Catalog volume, unzips them, and lands every row (all U.S. carriers,
# MAGIC all airports) as-is in a Delta table. Bronze = raw, append-only, fully traceable to source file.

# COMMAND ----------

dbutils.widgets.text("start_ym", "2024-08")
dbutils.widgets.text("end_ym", "2026-07")
START, END = dbutils.widgets.get("start_ym"), dbutils.widgets.get("end_ym")

CATALOG, SCHEMA = "workspace", "aa_ops"
VOL = f"/Volumes/{CATALOG}/{SCHEMA}/raw_bts"
BASE = "https://transtats.bts.gov/PREZIP/"
NAME = "On_Time_Marketing_Carrier_On_Time_Performance_Beginning_January_2018_{y}_{m}.zip"

# COMMAND ----------

import os, zipfile, urllib.request, shutil
from concurrent.futures import ThreadPoolExecutor

def months(start, end):
    y, m = map(int, start.split("-")); ey, em = map(int, end.split("-"))
    while (y, m) <= (ey, em):
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)

def fetch(ym):
    y, m = ym
    csv_path = f"{VOL}/csv/bts_{y}_{m:02d}.csv"
    if os.path.exists(csv_path):
        return f"{y}-{m:02d} cached"
    os.makedirs(f"{VOL}/csv", exist_ok=True)
    tmp_zip = f"/tmp/bts_{y}_{m}.zip"
    req = urllib.request.Request(BASE + NAME.format(y=y, m=m), headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=300) as r, open(tmp_zip, "wb") as f:
                shutil.copyfileobj(r, f)
            break
        except Exception as e:
            if attempt == 2:
                return f"{y}-{m:02d} FAILED {e}"
    with zipfile.ZipFile(tmp_zip) as z:
        inner = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
        with z.open(inner) as src, open(csv_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
    os.remove(tmp_zip)
    return f"{y}-{m:02d} ok"

with ThreadPoolExecutor(max_workers=4) as ex:
    log = list(ex.map(fetch, list(months(START, END))))
print("\n".join(log))
assert not any("FAILED" in l for l in log), "some months failed to download"

# COMMAND ----------

from pyspark.sql import functions as F

raw = (spark.read.option("header", True).option("inferSchema", False)
       .csv(f"{VOL}/csv/")
       .withColumn("_source_file", F.col("_metadata.file_path"))
       .withColumn("_ingested_at", F.current_timestamp()))

# BTS files carry an empty trailing column ("_c1xx"); drop anything unnamed
raw = raw.select([c for c in raw.columns if not c.startswith("_c")])
# Some BTS headers have stray spaces (e.g. "Operating_Airline "); Delta forbids them, so normalise names
import re
raw = raw.toDF(*[re.sub(r"[ ,;{}()\n\t=]+", "_", c.strip()) for c in raw.columns])

(raw.write.mode("overwrite").option("overwriteSchema", True)
    .saveAsTable(f"{CATALOG}.{SCHEMA}.bronze_flights"))

n = spark.table(f"{CATALOG}.{SCHEMA}.bronze_flights").count()
dbutils.notebook.exit(f"bronze_flights rows={n}; files={len(log)}")
