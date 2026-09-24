-- Databricks notebook source
-- MAGIC %md
-- MAGIC # 03 · Silver → Gold: airport-operations KPI marts (SQL)
-- MAGIC Business-ready tables for Tableau and Genie. Definitions:
-- MAGIC * **D0** – share of operated departures leaving at or before scheduled time (dep_delay ≤ 0)
-- MAGIC * **D15** – share of departures 15+ min late
-- MAGIC * **A14** – share of completed arrivals within 14 min of schedule
-- MAGIC * **Controllable delay** – BTS *carrier* delay minutes; **propagated** – *late-aircraft* minutes

-- COMMAND ----------

USE CATALOG workspace;
USE SCHEMA aa_ops;

-- COMMAND ----------

-- 1. Daily hub scorecard: AA-family vs. all other carriers at the same hub
CREATE OR REPLACE TABLE gold_hub_daily AS
WITH dep AS (
  SELECT origin AS hub, flight_date, is_aa,
         COUNT(*)                               AS sched_departures,
         SUM(CAST(cancelled AS INT))            AS cancellations,
         AVG(d0)                                AS d0_rate,
         AVG(d15)                               AS d15_rate,
         AVG(CASE WHEN NOT cancelled THEN dep_delay END) AS avg_dep_delay_min,
         AVG(taxi_out)                          AS avg_taxi_out_min
  FROM silver_hub_flights WHERE origin IN ('DFW','CLT','ORD','PHL','MIA')
  GROUP BY ALL),
arr AS (
  SELECT dest AS hub, flight_date, is_aa,
         COUNT(*) AS sched_arrivals, AVG(a14) AS a14_rate, AVG(taxi_in) AS avg_taxi_in_min
  FROM silver_hub_flights WHERE dest IN ('DFW','CLT','ORD','PHL','MIA')
  GROUP BY ALL)
SELECT d.hub, d.flight_date, dayofweek(d.flight_date) AS dow_num, date_format(d.flight_date, 'E') AS dow,
       CASE WHEN d.is_aa THEN 'American' ELSE 'Other carriers' END AS carrier_group,
       d.sched_departures, d.cancellations, d.cancellations / d.sched_departures AS cancel_rate,
       d.d0_rate, d.d15_rate, d.avg_dep_delay_min, d.avg_taxi_out_min,
       a.sched_arrivals, a.a14_rate, a.avg_taxi_in_min
FROM dep d LEFT JOIN arr a USING (hub, flight_date, is_aa);

-- COMMAND ----------

-- 2. When: AA departures by hub × weekday × scheduled departure hour (heatmap)
CREATE OR REPLACE TABLE gold_hub_hourly AS
SELECT origin AS hub, day_of_week, date_format(flight_date, 'E') AS dow, dep_hour,
       COUNT(*) AS sched_departures,
       AVG(d0) AS d0_rate, AVG(d15) AS d15_rate,
       AVG(CASE WHEN NOT cancelled THEN dep_delay END) AS avg_dep_delay_min,
       AVG(taxi_out) AS avg_taxi_out_min
FROM silver_hub_flights
WHERE is_aa AND origin IN ('DFW','CLT','ORD','PHL','MIA')
GROUP BY ALL;

-- COMMAND ----------

-- 3. Why: delay-cause minutes for AA departures (BTS causes, reported for flights 15+ min late)
CREATE OR REPLACE TABLE gold_delay_causes AS
SELECT origin AS hub, make_date(year, month, 1) AS month_start, cause, SUM(minutes) AS delay_minutes,
       CASE WHEN cause IN ('Carrier') THEN 'Controllable'
            WHEN cause = 'Late aircraft' THEN 'Propagated'
            ELSE 'Uncontrollable' END AS cause_class
FROM silver_hub_flights
LATERAL VIEW STACK(5,
   'Carrier', carrier_delay, 'Weather', weather_delay, 'NAS (ATC/airspace)', nas_delay,
   'Security', security_delay, 'Late aircraft', late_aircraft_delay) c AS cause, minutes
WHERE is_aa AND origin IN ('DFW','CLT','ORD','PHL','MIA') AND minutes > 0
GROUP BY ALL;

-- COMMAND ----------

-- 4. Turn risk: does a short scheduled turn pass inbound delay to the next departure?
CREATE OR REPLACE TABLE gold_turn_propagation AS
SELECT hub, turn_bucket,
       COUNT(*)                                          AS turns,
       AVG(sched_turn_min)                               AS avg_sched_turn_min,
       AVG(in_late15)                                    AS inbound_late15_rate,
       AVG(out_d0)                                       AS outbound_d0_rate,
       AVG(out_d15)                                      AS outbound_d15_rate,
       -- of turns whose inbound was 15+ late, how many departed 15+ late too?
       SUM(propagated) / NULLIF(SUM(in_late15), 0)       AS propagation_rate,
       SUM(COALESCE(out_late_aircraft_delay, 0))         AS late_aircraft_minutes
FROM silver_aircraft_turns
GROUP BY ALL;

-- COMMAND ----------

-- 5. Riskiest scheduled connections: inbound station -> hub -> outbound station (min 100 turns)
CREATE OR REPLACE TABLE gold_route_turn_risk AS
SELECT hub, in_origin, out_dest,
       COUNT(*) AS turns, AVG(sched_turn_min) AS avg_sched_turn_min,
       AVG(in_late15) AS inbound_late15_rate, AVG(out_d15) AS outbound_d15_rate,
       SUM(propagated) / NULLIF(SUM(in_late15), 0) AS propagation_rate,
       SUM(COALESCE(out_late_aircraft_delay, 0)) AS late_aircraft_minutes
FROM silver_aircraft_turns
GROUP BY ALL
HAVING COUNT(*) >= 100;

-- COMMAND ----------

-- 6. How delay builds through the day: each aircraft's 1st hub departure of the day vs. later hub departures
CREATE OR REPLACE TABLE gold_rotation_wave AS
WITH seq AS (
  SELECT origin AS hub, flight_date, tail_number, d0, d15, dep_delay,
         ROW_NUMBER() OVER (PARTITION BY tail_number, flight_date ORDER BY sched_dep_local) AS hub_dep_of_day
  FROM silver_hub_flights
  WHERE is_aa AND NOT cancelled AND tail_number IS NOT NULL AND origin IN ('DFW','CLT','ORD','PHL','MIA'))
SELECT hub, CASE WHEN hub_dep_of_day >= 4 THEN '4+' ELSE CAST(hub_dep_of_day AS STRING) END AS hub_dep_of_day,
       COUNT(*) AS departures, AVG(d0) AS d0_rate, AVG(d15) AS d15_rate, AVG(dep_delay) AS avg_dep_delay_min
FROM seq WHERE hub IN ('DFW','CLT','ORD','PHL','MIA')
GROUP BY ALL;

-- COMMAND ----------

-- 7. Peer benchmark by marketing carrier at each hub (carriers with 1,000+ departures)
CREATE OR REPLACE TABLE gold_carrier_benchmark AS
SELECT origin AS hub, carrier_group AS carrier,
       COUNT(*) AS sched_departures, AVG(CAST(cancelled AS INT)) AS cancel_rate,
       AVG(d0) AS d0_rate, AVG(d15) AS d15_rate, AVG(taxi_out) AS avg_taxi_out_min
FROM silver_hub_flights WHERE origin IN ('DFW','CLT','ORD','PHL','MIA')
GROUP BY ALL HAVING COUNT(*) >= 1000;

-- COMMAND ----------

-- Table & column comments help Genie understand the data
COMMENT ON TABLE gold_hub_daily IS 'Daily departure/arrival KPIs (D0, D15, A14, cancellations, taxi) for 5 American Airlines hubs, American vs other carriers. Source: BTS on-time data.';
COMMENT ON TABLE gold_hub_hourly IS 'American departures at each hub by weekday and scheduled departure hour: D0, D15, avg delay, taxi-out.';
COMMENT ON TABLE gold_delay_causes IS 'Monthly BTS delay-cause minutes for American departures by hub; cause_class = Controllable / Propagated / Uncontrollable.';
COMMENT ON TABLE gold_turn_propagation IS 'American aircraft turns at hubs grouped by scheduled turn-time bucket: how often inbound delay (15+) propagates to the next departure.';
COMMENT ON TABLE gold_route_turn_risk IS 'Inbound station -> hub -> outbound station aircraft connections with delay-propagation rates (100+ turns).';
COMMENT ON TABLE gold_rotation_wave IS 'On-time performance by the aircraft''s hub departure number of the day (1st hub departure vs later), showing how delay accumulates.';
COMMENT ON TABLE gold_carrier_benchmark IS 'Departure KPIs by marketing carrier at each hub for peer benchmarking.';

-- COMMAND ----------

SELECT 'gold_hub_daily' t, COUNT(*) n FROM gold_hub_daily UNION ALL
SELECT 'gold_hub_hourly', COUNT(*) FROM gold_hub_hourly UNION ALL
SELECT 'gold_delay_causes', COUNT(*) FROM gold_delay_causes UNION ALL
SELECT 'gold_turn_propagation', COUNT(*) FROM gold_turn_propagation UNION ALL
SELECT 'gold_route_turn_risk', COUNT(*) FROM gold_route_turn_risk UNION ALL
SELECT 'gold_rotation_wave', COUNT(*) FROM gold_rotation_wave UNION ALL
SELECT 'gold_carrier_benchmark', COUNT(*) FROM gold_carrier_benchmark;
