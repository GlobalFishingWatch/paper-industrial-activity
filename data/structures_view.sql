CREATE OR REPLACE VIEW `global-fishing-watch.paper_industrial_activity.combined_offshore_infrastructure_v20231106` AS

WITH structures as (
  SELECT * FROM `global-fishing-watch.paper_industrial_activity.offshore_infrastructure_v20231106`
),
structures_meta as (
  SELECT * FROM `global-fishing-watch.paper_industrial_activity.offshore_infrastructure_meta_v20231106`
  WHERE POL_TYPE NOT IN ('Joint regime (EEZ)')  -- Exclude duplicate entries in the Ecuador/Peru joint management regime
)

SELECT *
FROM structures
LEFT JOIN structures_meta
USING (structure_id)

