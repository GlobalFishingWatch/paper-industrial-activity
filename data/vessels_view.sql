CREATE OR REPLACE VIEW `global-fishing-watch.paper_industrial_activity.vessels` AS

WITH vessels as (
  SELECT *
  FROM `global-fishing-watch.paper_industrial_activity.vessels_v20231013`
),
vessels_meta as (
  SELECT *
  FROM `global-fishing-watch.paper_industrial_activity.vessels_meta_v20231106`
)

SELECT *
FROM vessels
LEFT JOIN vessels_meta
USING (detect_id)
