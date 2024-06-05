CREATE OR REPLACE VIEW `global-fishing-watch.paper_industrial_activity.vessels` AS

WITH vessels as (
  SELECT *
  FROM `global-fishing-watch.paper_industrial_activity.vessels_v20231013`
),
vessels_meta as (
  SELECT * except(mmsi), mmsi as mmsi_str
  FROM `global-fishing-watch.paper_industrial_activity.vessels_meta_v20231106`
)

SELECT * except(mmsi_str)
FROM vessels
LEFT JOIN vessels_meta
ON(
  CAST(mmsi AS STRING) = mmsi_str
  AND EXTRACT(YEAR FROM vessels.timestamp) = vessels_meta.year
  )
