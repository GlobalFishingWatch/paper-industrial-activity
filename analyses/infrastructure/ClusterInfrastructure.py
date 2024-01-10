# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: rad
#     language: python
#     name: python3
# ---

# %%
import pandas as pd

# %% [markdown]
# ## This script is used to cluster infrastructure detections accross time, in order to associate detections with slightly different detection coordinates but are likely the same object. It also removes detections that were only seen in a single month, and fills in detection gaps where there was a detection on the previous and following month. 
#
# ### The steps include:
# 1. Create cluster locations and unique cluster number, and assign cluster number to detections
# 2. Create cluster features and label detections
# 3. Reclassify detection label
# 4. Reclassify cluster detection label
# 5. Reclassified subcluster detection label, and assemble final table

# %% [markdown]
# #### 1. Create cluster locations and unique cluster number, and assign cluster number to detections. 

# %%
q = '''with
detections as ((select detect_lon, detect_lat, detect_id, start_time, end_time, 
st_geogpoint(detect_lon, detect_lat) loc
 from  `proj_sentinel1_v20210924.detect_comp_raw_*`)),

clust_2016_2018 as (
  select
  st_centroid(st_union_agg(loc)) as clust_centr
   from (
select
*,
ST_CLUSTERDBSCAN(loc,50, 1) over () as cluster_num_50_m,
from
detections
where extract(year from start_time) in(2016, 2017, 2018))
group by cluster_num_50_m
),

clust_2019_2021 as (
  select
  st_centroid(st_union_agg(loc)) as clust_centr
   from (
select
*,
ST_CLUSTERDBSCAN(loc,50, 1) over () as cluster_num_50_m,
from
detections
where extract(year from start_time) in(2019, 2020, 2021))
group by cluster_num_50_m),

clust_all_unique as (
  select
  st_centroid(st_union_agg(clust_centr)) as clust_centr_final
   from (
  select
  *,
  ST_CLUSTERDBSCAN(clust_centr,50, 1) over () as clusters_final 
  from (
select * from clust_2016_2018
union all 
select * from clust_2019_2021)
)
group by clusters_final
qualify row_number() over (partition by clusters_final) = 1),


-- Assign cluster to detection. If the detection is greater than 100-m from
-- a cluster, make it it's own cluster. This edge effect is probably a result
-- of taking cluster centroids
detections_clusters_less100 as (
select 
distinct
detect_id, ST_ASTEXT(clust_centr_final) clust_centr_final
from detections cross join clust_all_unique
where
st_distance(loc, clust_centr_final) <=100),


-- select detections that are > 100 from a cluster 
-- and make their own cluster by adding them
-- to the other clusters. Assing final cluster numbers to clusterse here and save
cluster_locations as (
  select distinct *,
  row_number() over() as cluster_number
  from(
(select distinct clust_centr_final from detections_clusters_less100 )
union all 
(select distinct ST_ASTEXT(loc) as clust_centr_final from detections where detect_id not in (select detect_id from detections_clusters_less100 where detect_id is not null)))),


final_cluster_locations as (
select ST_GEOGFROMTEXT(clust_centr_final) clust_centr_final, cluster_number from cluster_locations

),
#assign cluster numbers to the detections and save
assign_clusters as (
select 
*, 
st_distance(loc, clust_centr_final) as dist_to_cluster
from detections cross join final_cluster_locations
where
st_distance(loc, clust_centr_final) <=100
qualify row_number() over (partition by detect_id order by dist_to_cluster) = 1)

select * except(dist_to_cluster) from assign_clusters'''

# %%
# run query
clusters = pd.read_gbq(q)

# %%
# push cluster locations to gbq
date = 20231201
clustered_out = f'proj_sentinel1_v20210924.detect_comp_clustered_raw_{date}'
# clusters.to_gbq(clustered_out, if_exists = 'fail')

# %%
clustered_out

# %%
# input of new detections table that includes new and old detections
#  to clutster and reclassfiy
raw_detections = 'proj_sentinel1_v20210924.detect_comp_pred_v2_*'

# %% [markdown]
# #### 2. Create cluster features and label detections

# %%
# create cluster features
q = f'''
with
#################
-- Creating clusters and sub clusters
#################

-- create midpoint of composite. If it is on the 31, round up to next month

labels as (
select * except (comp_mid),
if(extract(day from comp_mid) > 15, date_add(date_trunc(comp_mid, month), interval 1 MONTH), date_trunc(comp_mid, month)) midpoint
from(
 select *,
 date(date_add(start_time, interval cast((date_diff(end_time, start_time, day)/2) as int64) day)) as comp_mid
  from `{raw_detections}`
 left join `{clustered_out}`
 using (detect_id))),

 detections_count as (
SELECT cluster_number, count(*) as num_detections
 FROM labels
 group by cluster_number),

 predictions as (select * from labels
 left join detections_count
 using (cluster_number)),

-- get cluster start end times, and time differences of cluster detections
start_end as (
select *,
first_value(midpoint) over (partition by cluster_number order by midpoint ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as cluster_start,
last_value(midpoint) over (partition by cluster_number order by midpoint ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as cluster_end,
lead(midpoint, 1) OVER (PARTITION BY cluster_number ORDER BY midpoint) next_cluster_midpoint
 from predictions),
 
time_between_clusters as (
select *,
if(midpoint != cluster_end, date_diff(next_cluster_midpoint, midpoint, month), null) months_to_next,
from start_end),

-- Start of subclusters
-- include months in determining subcluster start/end times if there is no detection, 
-- but there is a detection for the month before and after. (just 1 month missing)
cluster_time as (
select 
cluster_number,
start_time,
end_time,
midpoint,
cluster_start,
cluster_end,
next_cluster_midpoint,
months_to_next,
if(lag(months_to_next) over (partition by cluster_number order by midpoint) > 2 or (midpoint = cluster_start), 'start', 'null') start_subcluster,
from
time_between_clusters),

-- Assign sub clusters number
subclusters as (
select *,
countif(start_subcluster = 'start') over (partition by cluster_number order by midpoint rows between UNBOUNDED preceding and current row) as subcluster_number
from cluster_time),

count_subclusters as (
select *,
count(*) over (partition by cluster_number, subcluster_number) as subcluster_count
 from subclusters
 ),

-- detection lasts at least 2 months (remove single month detections)
min_cluster_duration as (
select * from count_subclusters
where subcluster_count >=2),

-- Create months that the infrastructure wasn't seen to fill in missing dates
fill_in_date_gaps as (
select
distinct 
cluster_number, 
subcluster_number,
midpoint
from(
select
cluster_number, 
subcluster_number, 
generate_date_array(first_value(midpoint) over (partition by cluster_number, subcluster_number order by midpoint ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING),(last_value(midpoint) over (partition by cluster_number, subcluster_number order by midpoint ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)), INTERVAL 1 MONTH) as dates
from min_cluster_duration),
unnest(dates) midpoint),

-- backfill missing dates
all_dates as (
select 
*
from fill_in_date_gaps
left join min_cluster_duration
using(cluster_number, subcluster_number, midpoint)),

-- cluster start end
cluster_range as (select cluster_number, 
min(midpoint) as cluster_start_date,
max(midpoint) as cluster_end_date
from all_dates 
group by cluster_number),

-- subcluster start end
subcluster_range as (select cluster_number, 
subcluster_number,
min(midpoint) as subcluster_start_date,
max(midpoint) as subcluster_end_date
from all_dates 
group by cluster_number, subcluster_number),

-- add back in features
features as (select 
detect_id,
wind,
oil,
other,
noise,
detect_lon,
detect_lat,
start_time,
end_time,
midpoint,
cluster_number,
if(num_detections is null, lag(num_detections) over (partition by cluster_number order by midpoint), num_detections) as cluster_detections,
if(subcluster_number is null, lag(subcluster_number) over (partition by cluster_number order by midpoint), subcluster_number) as subcluster_number,
if(subcluster_count is null, lag(subcluster_count) over (partition by cluster_number order by midpoint), subcluster_count) as subcluster_detections,
cluster_start_date,
cluster_end_date,
subcluster_start_date,
subcluster_end_date,
if(clust_centr_final is null, lag(clust_centr_final) over (partition by cluster_number order by midpoint), clust_centr_final) as clust_centr_final,
case greatest(wind,oil,other, noise) 
  when wind then 'wind'
  when oil then 'oil'
  when other then 'other'
  when noise then 'noise'
end as detection_label
 from all_dates
left join cluster_range
using(cluster_number)
left join subcluster_range
using(cluster_number,subcluster_number)
left join predictions
using (cluster_number, start_time, end_time, midpoint)
)

select distinct * except(clust_centr_final), 
ST_Y(clust_centr_final) clust_centr_lat,
ST_X(clust_centr_final) clust_centr_lon from features'''

# %%
features = pd.read_gbq(q)

# %%
# push features and detections table to gbq
date = 20231201
features_out = f'proj_sentinel1_v20210924.detect_comp_clustered_features_{date}'
# features.to_gbq(features_out, if_exists = 'fail')

# %% [markdown]
# #### Reclassify using manual review data and classification scheme
# ##### The complexity of the query required it to be broken down into several queries to run successfully

# %% [markdown]
# ##### 3. Reclassify detection label

# %%
q = f'''

-- #################
-- -- add manual review data here, and reclassification detection_label.
-- #################
with 

all_oil_polygons AS (
  SELECT
    ST_GEOGFROMTEXT(geometry, make_valid => TRUE) AS geometry
  FROM
    proj_global_sar.oil_polygons_buffer_10th_degree),

all_wind_polygons AS (
SELECT
  ST_GEOGFROMTEXT(geometry, make_valid => TRUE) AS geometry
FROM
  proj_global_sar.infra_wind_regions),

regions AS (
SELECT
region as oil_region,
  ST_GEOGFROMTEXT(geometry, make_valid => TRUE) AS region_geometry
FROM
  `proj_global_sar.infra_oil_regions`),

sk_wind AS (
SELECT
  ST_GEOGFROMTEXT(geometry, make_valid => TRUE) AS geometry
FROM
  proj_global_sar.skytruth_review_wind),

sk_other AS (
  SELECT
    ST_GEOGFROMTEXT(geometry, make_valid => TRUE) AS geometry
  FROM
    `proj_global_sar.skytruth_review_other_v20231103`),

sk_not_infra AS (
SELECT
  ST_GEOGFROMTEXT(geometry, make_valid => TRUE) AS geometry
FROM
  `proj_global_sar.skytruth_review_noise_v20230810`),

sk_oil AS (
SELECT
  ST_GEOGFROMTEXT(geometry, make_valid => TRUE) AS geometry
FROM
  `proj_global_sar.skytruth_review_oil_v20230810`),

oil_to_unknown AS (
  SELECT
    ST_GEOGFROMTEXT(geometry, make_valid => TRUE) AS geometry
  FROM
  `proj_global_sar.jc_oil_to_unknown_v20231103`),

reviewed_wind as (
SELECT
  detect_id,
FROM
  `{features_out}`
CROSS JOIN
  sk_wind
WHERE
  ST_CONTAINS(geometry, ST_GEOGPOINT(detect_lon, detect_lat)) ),

reviewed_oil as (
SELECT
  detect_id,
FROM
  `{features_out}`
CROSS JOIN
  sk_oil
WHERE
  ST_CONTAINS(geometry, ST_GEOGPOINT(detect_lon, detect_lat)) ),

reviewed_other as (
SELECT
  detect_id,
FROM
  `{features_out}`
CROSS JOIN
  sk_other
WHERE
  ST_CONTAINS(geometry, ST_GEOGPOINT(detect_lon, detect_lat)) ),

not_infra as (
SELECT
  detect_id,
FROM
  `{features_out}`
CROSS JOIN
  sk_not_infra
WHERE
  ST_CONTAINS(geometry, ST_GEOGPOINT(detect_lon, detect_lat)) ),

maracaibo AS (
SELECT
  detect_id,
FROM
  `{features_out}`
CROSS JOIN
  regions
WHERE
  ST_CONTAINS(region_geometry, ST_GEOGPOINT(detect_lon, detect_lat))
  AND oil_region IN ('Lake Maracaibo')),

-------------------
-- these are based on the rule set: if a label is within an area, change to this or that
-------------------  

oil_in_oil_polygon AS (
SELECT
  detect_id,
FROM
  `{features_out}`
CROSS JOIN
  all_oil_polygons
WHERE
  detection_label = 'oil'
  AND ST_CONTAINS(geometry, ST_GEOGPOINT(detect_lon, detect_lat)) ),

  reviewed_oil_to_unknown as (
SELECT
  detect_id,
FROM
  `{features_out}`
CROSS JOIN
  oil_to_unknown
WHERE
  detection_label = 'oil'
  and ST_CONTAINS(geometry, ST_GEOGPOINT(detect_lon, detect_lat)) ),

oil_detections_outside_polygon AS (
SELECT
  detect_id,
FROM
  `{features_out}`
WHERE
  detection_label = 'oil'
  AND detect_id NOT IN (
  SELECT
    detect_id
  FROM
    oil_in_oil_polygon
    where detect_id is not null) ),

other_in_oil_polygon AS (
SELECT
  detect_id,
FROM
  `{features_out}`
CROSS JOIN
  all_oil_polygons
WHERE
  detection_label = 'other'
  AND ST_CONTAINS(geometry, ST_GEOGPOINT(detect_lon, detect_lat)) ),

other_detections_outside_oil_polygon AS (
SELECT
  *
FROM
  `{features_out}`
WHERE
  detection_label = 'other'
  AND detect_id NOT IN (
  SELECT
    detect_id
  FROM
    other_in_oil_polygon
    where detect_id is not null) ),

wind_in_wind_regions AS (
SELECT
  detect_id,
FROM
  `{features_out}`
CROSS JOIN
  all_wind_polygons
WHERE
  detection_label = 'wind'
  AND ST_CONTAINS(geometry, ST_GEOGPOINT(detect_lon, detect_lat)) ),

wind_detections_outside_wind_regions AS (
SELECT
  detect_id,
FROM
  `{features_out}`
WHERE
  detection_label = 'wind'
  AND detect_id NOT IN (
  SELECT
    detect_id
  FROM
    wind_in_wind_regions
    where detect_id is not null) ),

all_oil AS (
SELECT
  *
FROM
  `{features_out}`
WHERE
  detection_label = 'oil'),

all_wind AS (
SELECT
  *
FROM
  `{features_out}`
WHERE
  detection_label = 'wind'),

oil_near_wind AS (
SELECT
  a.detect_id,
FROM
  all_oil a
CROSS JOIN
  all_wind b
CROSS JOIN
  all_wind_polygons c
WHERE
  ST_DISTANCE(ST_GEOGPOINT(a.detect_lon, a.detect_lat), ST_GEOGPOINT(b.detect_lon, b.detect_lat)) <= 2 * 1000
  AND ST_WITHIN(ST_GEOGPOINT(a.detect_lon, a.detect_lat), c.geometry)
  AND a.detection_label = 'oil'),

-- fixing oil and other near/ inside a windfarm. Need to apply the 2km condition iteratively 
-- until there are no detections left to reclassify. In this case, you will pickup detections close to previously reclassified points.
new_probable_wind AS (
SELECT
  distinct
  a.detect_id,
  a.detect_lon,
  a.detect_lat
FROM
  (select * from all_oil 
  union all 
  select * from other_detections_outside_oil_polygon) a
CROSS JOIN
  all_wind b
CROSS JOIN
  all_wind_polygons c
WHERE
  ST_DISTANCE(ST_GEOGPOINT(a.detect_lon, a.detect_lat), ST_GEOGPOINT(b.detect_lon, b.detect_lat)) <= 2 * 1000
  AND ST_WITHIN(ST_GEOGPOINT(a.detect_lon, a.detect_lat), c.geometry)),

new_probable_wind_2 as (
SELECT
    distinct
    a.detect_id,
    a.detect_lon,
    a.detect_lat
  FROM
    (select * from all_oil 
    union all 
    select * from other_detections_outside_oil_polygon) a
    CROSS JOIN
    (select detect_id, detect_lon, detect_lat
    from
    all_wind
    union all
    select * from new_probable_wind) b
  CROSS JOIN
    all_wind_polygons c
  WHERE
    ST_DISTANCE(ST_GEOGPOINT(a.detect_lon, a.detect_lat), ST_GEOGPOINT(b.detect_lon, b.detect_lat)) <= 2 * 1000
    AND ST_WITHIN(ST_GEOGPOINT(a.detect_lon, a.detect_lat), c.geometry)),

final_new_probable_wind as (
SELECT
  distinct
    a.detect_id,
  FROM
    (select * from all_oil 
    union all 
    select * from other_detections_outside_oil_polygon) a
    CROSS JOIN
    (select detect_id, detect_lon, detect_lat
    from
    all_wind
    union all
    select * from new_probable_wind_2) b
  CROSS JOIN
    all_wind_polygons c
  WHERE
    ST_DISTANCE(ST_GEOGPOINT(a.detect_lon, a.detect_lat), ST_GEOGPOINT(b.detect_lon, b.detect_lat)) <= 2 * 1000
    AND ST_WITHIN(ST_GEOGPOINT(a.detect_lon, a.detect_lat), c.geometry)),

noise_detections as (
  select detect_id from `{features_out}`
  where detection_label = 'noise'
),

label_reclass AS (
  SELECT
    *,
    CASE
      WHEN detect_id in (select detect_id from not_infra) THEN 'noise'
      WHEN detect_id in (select detect_id from reviewed_oil_to_unknown) THEN 'unknown'
      WHEN detect_id in (select detect_id from reviewed_oil) THEN 'oil'
      WHEN detect_id in (select detect_id from reviewed_wind) THEN 'wind'
      WHEN detect_id in (select detect_id from reviewed_other) THEN 'unknown'
      WHEN detect_id in (select detect_id from maracaibo) THEN 'lake_maracaibo'
      WHEN detect_id in (select detect_id from final_new_probable_wind) THEN 'probable_wind'
      WHEN detect_id in (select detect_id from oil_in_oil_polygon) THEN 'oil'
      WHEN detect_id in (select detect_id from oil_detections_outside_polygon) THEN 'possible_oil'
      WHEN detect_id in (select detect_id from other_in_oil_polygon) THEN 'probable_oil'
      WHEN detect_id in (select detect_id from other_detections_outside_oil_polygon) THEN 'unknown'
      WHEN detect_id in (select detect_id from wind_in_wind_regions) THEN 'wind'
      WHEN detect_id in (select detect_id from wind_detections_outside_wind_regions) THEN 'possible_wind'
      WHEN detect_id in (select detect_id from noise_detections) THEN 'noise'
      WHEN detect_id is null THEN null -- these are filled in date gaps. no detection label so no reclass label
    ELSE
    'unknown'
  END
    AS detection_label_reclassified, 
  FROM
    `{features_out}`)

    select * from label_reclass'''

# %%
detection_label = pd.read_gbq(q)

# %%
# push detection reclassification table to gbq
date = 20231203
label_reclass = f'scratch_pete.infra_cluster_label_reclass_v{date}'
detection_label.to_gbq(label_reclass, if_exists = 'replace')

# %% [markdown]
# ##### 4. Reclassify cluster detection label

# %%
q = f'''


with
-----------------------
-- reclass polygons
-----------------------
all_oil_polygons AS (
  SELECT
    ST_GEOGFROMTEXT(geometry, make_valid => TRUE) AS geometry
  FROM
    proj_global_sar.oil_polygons_buffer_10th_degree),

all_wind_polygons AS (
SELECT
  ST_GEOGFROMTEXT(geometry, make_valid => TRUE) AS geometry
FROM
  proj_global_sar.infra_wind_regions),

regions AS (
SELECT
region as oil_region,
  ST_GEOGFROMTEXT(geometry, make_valid => TRUE) AS region_geometry
FROM
  `proj_global_sar.infra_oil_regions`),

sk_wind AS (
SELECT
  ST_GEOGFROMTEXT(geometry, make_valid => TRUE) AS geometry
FROM
  `proj_global_sar.skytruth_review_wind`),

sk_other AS (
  SELECT
    ST_GEOGFROMTEXT(geometry, make_valid => TRUE) AS geometry
  FROM
    `proj_global_sar.skytruth_review_other_v20231103`),

sk_not_infra AS (
SELECT
  ST_GEOGFROMTEXT(geometry, make_valid => TRUE) AS geometry
FROM
  `proj_global_sar.skytruth_review_noise_v20230810`),

sk_oil AS (
SELECT
  ST_GEOGFROMTEXT(geometry, make_valid => TRUE) AS geometry
FROM
  `proj_global_sar.skytruth_review_oil_v20230810`),

oil_to_unknown AS (
  SELECT
    ST_GEOGFROMTEXT(geometry, make_valid => TRUE) AS geometry
  FROM
  `proj_global_sar.jc_oil_to_unknown_v20231103`),

---------------------------
-- avg cluster labels
---------------------------
cluster_labels as (
select
distinct
  detect_id,
  midpoint,
  detect_lon,
  detect_lat,
  cluster_number, 
  clust_centr_lat,
  clust_centr_lon,
  avg_cluster_noise,
  avg_cluster_oil,
  avg_cluster_other,
  avg_cluster_wind,
  case greatest(avg_cluster_wind,avg_cluster_oil,avg_cluster_other,avg_cluster_noise) 
  when avg_cluster_wind then 'wind'
  when avg_cluster_oil then 'oil'
  when avg_cluster_other then 'other'
  when avg_cluster_noise then 'noise'
end as cluster_label,
from `{features_out}`
left join (
  select
  cluster_number,
  avg(wind) as avg_cluster_wind, 
  avg(oil) as avg_cluster_oil,
  avg(other) as avg_cluster_other,
  avg(noise) as avg_cluster_noise
  from `{features_out}`
group by 1)
using(cluster_number)),

-- ------------------------
-- -- Create new cluster_number selections using manually reviewed areas for reclassifying clusters labels
-- ------------------------ 
reviewed_wind as (
SELECT
  detect_id,
FROM
  `{features_out}`
CROSS JOIN
  sk_wind
WHERE
  ST_CONTAINS(geometry, ST_GEOGPOINT(detect_lon, detect_lat)) ),

reviewed_oil as (
SELECT
  detect_id,
FROM
  `{features_out}`
CROSS JOIN
  sk_oil
WHERE
  ST_CONTAINS(geometry, ST_GEOGPOINT(detect_lon, detect_lat)) ),

reviewed_other as (
SELECT
  detect_id,
FROM
  `{features_out}`
CROSS JOIN
  sk_other
WHERE
  ST_CONTAINS(geometry, ST_GEOGPOINT(detect_lon, detect_lat)) ),

not_infra as (
SELECT
  detect_id,
FROM
  {features_out}
CROSS JOIN
  sk_not_infra
WHERE
  ST_CONTAINS(geometry, ST_GEOGPOINT(detect_lon, detect_lat)) ),

maracaibo AS (
SELECT
  detect_id,
FROM
  {features_out}
CROSS JOIN
  regions
WHERE
  ST_CONTAINS(region_geometry, ST_GEOGPOINT(detect_lon, detect_lat))
  AND oil_region IN ('Lake Maracaibo')),

-------------------
-- these are based on the rule set: if a label is within an area, change to this or that
-------------------
oil_in_oil_polygon_cluster AS (
SELECT
  detect_id,
FROM
  cluster_labels
CROSS JOIN
  all_oil_polygons
WHERE
  cluster_label = 'oil'
  AND ST_CONTAINS(geometry, ST_GEOGPOINT(clust_centr_lon, clust_centr_lat))),

reviewed_oil_to_unknown_cluster as (
SELECT
  detect_id,
FROM
  cluster_labels
CROSS JOIN
  oil_to_unknown
WHERE
  cluster_label = 'oil'
  and ST_CONTAINS(geometry, ST_GEOGPOINT(clust_centr_lon, clust_centr_lat))),


oil_detections_outside_polygon_cluster AS (
SELECT
  detect_id,
FROM
  cluster_labels
WHERE
  cluster_label = 'oil'
  AND detect_id NOT IN (
  SELECT
    detect_id
  FROM
    oil_in_oil_polygon_cluster
    where detect_id is not null)),

other_in_oil_polygon_cluster AS (
SELECT
  detect_id,
FROM
  cluster_labels
CROSS JOIN
  all_oil_polygons
WHERE
  cluster_label = 'other'
  AND ST_CONTAINS(geometry, ST_GEOGPOINT(clust_centr_lon, clust_centr_lat))),

other_detections_outside_oil_polygon_cluster AS (
SELECT
  *
FROM
  cluster_labels
WHERE
  cluster_label = 'other'
  AND detect_id NOT IN (
  SELECT
    detect_id
  FROM
    other_in_oil_polygon_cluster
    where detect_id is not null)) ,

wind_in_wind_regions_cluster AS (
SELECT
  detect_id,
FROM
  cluster_labels
CROSS JOIN
  all_wind_polygons
WHERE
  cluster_label = 'wind'
  AND ST_CONTAINS(geometry, ST_GEOGPOINT(clust_centr_lon, clust_centr_lat)) ),

wind_detections_outside_wind_regions_cluster AS (
SELECT
  detect_id,
FROM
  cluster_labels
WHERE
  cluster_label = 'wind'
  AND detect_id NOT IN (
  SELECT
    detect_id
  FROM
    wind_in_wind_regions_cluster
    where detect_id is not null) ),

all_oil_cluster AS (
SELECT
  *
FROM
  cluster_labels
WHERE
  cluster_label = 'oil'),

all_wind_cluster AS (
SELECT
  *
FROM
  cluster_labels
WHERE
  cluster_label = 'wind'),

oil_near_wind_cluster AS (
SELECT
  a.detect_id,
FROM
  all_oil_cluster a
CROSS JOIN
  all_wind_cluster b
CROSS JOIN
  all_wind_polygons c
WHERE
  ST_DISTANCE(ST_GEOGPOINT(a.clust_centr_lon, a.clust_centr_lat), ST_GEOGPOINT(b.clust_centr_lon, b.clust_centr_lat)) <= 2 * 1000
  AND ST_WITHIN(ST_GEOGPOINT(a.clust_centr_lon, a.clust_centr_lat), c.geometry)
  AND a.cluster_label = 'oil'),


-- fixing oil and other near/ inside a windfarm. Need to apply the 2km condition iteratively until there are no detections left to reclassify. In this case, you will pickup detections close to previously reclassified points.
new_probable_wind_cluster AS (
SELECT
  distinct
  a.detect_id,
  a.clust_centr_lon,
  a.clust_centr_lat
FROM
  (select * from all_oil_cluster 
  union all 
  select * from other_detections_outside_oil_polygon_cluster) a
CROSS JOIN
  all_wind_cluster b
CROSS JOIN
  all_wind_polygons c
WHERE
  ST_DISTANCE(ST_GEOGPOINT(a.clust_centr_lon, a.clust_centr_lat), ST_GEOGPOINT(b.clust_centr_lon, b.clust_centr_lat)) <= 2 * 1000
  AND ST_WITHIN(ST_GEOGPOINT(a.clust_centr_lon, a.clust_centr_lat), c.geometry)),

new_probable_wind_2_cluster as (
SELECT
    distinct
    a.detect_id,
    a.clust_centr_lon,
    a.clust_centr_lat
  FROM
    (select * from all_oil_cluster
    union all 
    select * from other_detections_outside_oil_polygon_cluster) a
    CROSS JOIN
    (select detect_id, clust_centr_lon, clust_centr_lat
    from
    all_wind_cluster
    union all
    select * from new_probable_wind_cluster) b
  CROSS JOIN
    all_wind_polygons c
  WHERE
    ST_DISTANCE(ST_GEOGPOINT(a.clust_centr_lon, a.clust_centr_lat), ST_GEOGPOINT(b.clust_centr_lon, b.clust_centr_lat)) <= 2 * 1000
    AND ST_WITHIN(ST_GEOGPOINT(a.clust_centr_lon, a.clust_centr_lat), c.geometry)),

final_new_probable_wind_cluster as (
SELECT
  distinct
    a.detect_id,
  FROM
    (select * from all_oil_cluster 
    union all 
    select * from other_detections_outside_oil_polygon_cluster) a
    CROSS JOIN
    (select detect_id, clust_centr_lon, clust_centr_lat
    from
    all_wind_cluster
    union all
    select * from new_probable_wind_2_cluster) b
  CROSS JOIN
    all_wind_polygons c
  WHERE
    ST_DISTANCE(ST_GEOGPOINT(a.clust_centr_lon, a.clust_centr_lat), ST_GEOGPOINT(b.clust_centr_lon, b.clust_centr_lat)) <= 2 * 1000
    AND ST_WITHIN(ST_GEOGPOINT(a.clust_centr_lon, a.clust_centr_lat), c.geometry)),

    noise_detections_cluster as (
  select detect_id from cluster_labels
  where cluster_label = 'noise'),

-- cluster labels reclassified
label_reclass_cluster AS (
  select * except(cluster_label_reclassified),
if(detect_id is null, first_value(cluster_label_reclassified IGNORE NULLS) over (partition by cluster_number order by cluster_number), cluster_label_reclassified) as cluster_label_reclassified
from
(
  SELECT
    *,
    CASE
      WHEN detect_id in (select detect_id from not_infra) THEN 'noise'
      WHEN detect_id in (select detect_id from reviewed_oil_to_unknown_cluster) THEN 'unknown'
      WHEN detect_id in (select detect_id from reviewed_oil) THEN 'oil'
      WHEN detect_id in (select detect_id from reviewed_wind) THEN 'wind'
      WHEN detect_id in (select detect_id from reviewed_other) THEN 'unknown'
      WHEN detect_id in (select detect_id from maracaibo) THEN 'lake_maracaibo'
      WHEN detect_id in (select detect_id from final_new_probable_wind_cluster) THEN 'probable_wind'
      WHEN detect_id in (select detect_id from oil_in_oil_polygon_cluster) THEN 'oil'
      WHEN detect_id in (select detect_id from oil_detections_outside_polygon_cluster) THEN 'possible_oil'
      WHEN detect_id in (select detect_id from other_in_oil_polygon_cluster) THEN 'probable_oil'
      WHEN detect_id in (select detect_id from other_detections_outside_oil_polygon_cluster) THEN 'unknown'
      WHEN detect_id in (select detect_id from wind_in_wind_regions_cluster) THEN 'wind'
      WHEN detect_id in (select detect_id from wind_detections_outside_wind_regions_cluster) THEN 'possible_wind'
      WHEN detect_id in (select detect_id from noise_detections_cluster) THEN 'noise'
      WHEN detect_id is null THEN null -- these are filled in date gaps. Assign this most common value of cluster above
    ELSE
    'unknown'
  END
    AS cluster_label_reclassified, 
  FROM
    cluster_labels))

    select distinct * from label_reclass_cluster
'''

# %%
clust_label = pd.read_gbq(q)

# %%
# push cluster detection reclassification table to gbq
date = 20231203
clust_label_reclass = f'scratch_pete.infra_reclass_clusters_v{date}'
clust_label.to_gbq(clust_label_reclass, if_exists = 'replace')

# %%
clust_label_reclass

# %% [markdown]
# ##### 5. Reclassified subcluster detection label, and assemble final table

# %%
q = f'''with

-----------------------
-- reclass polygons
-----------------------
all_oil_polygons AS (
  SELECT
    ST_GEOGFROMTEXT(geometry, make_valid => TRUE) AS geometry
  FROM
    proj_global_sar.oil_polygons_buffer_10th_degree),

all_wind_polygons AS (
SELECT
  ST_GEOGFROMTEXT(geometry, make_valid => TRUE) AS geometry
FROM
  proj_global_sar.infra_wind_regions),

regions AS (
SELECT
region as oil_region,
  ST_GEOGFROMTEXT(geometry, make_valid => TRUE) AS region_geometry
FROM
  `proj_global_sar.infra_oil_regions`),

sk_wind AS (
SELECT
  ST_GEOGFROMTEXT(geometry, make_valid => TRUE) AS geometry
FROM
  proj_global_sar.skytruth_review_wind),

sk_other AS (
  SELECT
    ST_GEOGFROMTEXT(geometry, make_valid => TRUE) AS geometry
  FROM
    `proj_global_sar.skytruth_review_other_v20231103`),

sk_not_infra AS (
SELECT
  ST_GEOGFROMTEXT(geometry, make_valid => TRUE) AS geometry
FROM
  `proj_global_sar.skytruth_review_noise_v20230810`),

sk_oil AS (
SELECT
  ST_GEOGFROMTEXT(geometry, make_valid => TRUE) AS geometry
FROM
  `proj_global_sar.skytruth_review_oil_v20230810`),

oil_to_unknown AS (
  SELECT
    ST_GEOGFROMTEXT(geometry, make_valid => TRUE) AS geometry
  FROM
  `proj_global_sar.jc_oil_to_unknown_v20231103`),

----------------------
-- avg subcluster labels
-- do it all over again for subclusters
---------------------
subcluster_labels as (
select
distinct
  detect_id,
  midpoint,
  cluster_number, 
  subcluster_number,
  clust_centr_lat,
  clust_centr_lon,
  avg_subcluster_noise,
  avg_subcluster_oil,
  avg_subcluster_other,
  avg_subcluster_wind,
  case greatest(avg_subcluster_wind,avg_subcluster_oil,avg_subcluster_other,avg_subcluster_noise) 
  when avg_subcluster_wind then 'wind'
  when avg_subcluster_oil then 'oil'
  when avg_subcluster_other then 'other'
  when avg_subcluster_noise then 'noise'
end as subcluster_label,
from {features_out}
left join (
  select
  cluster_number,
  subcluster_number,
  avg(wind) as avg_subcluster_wind, 
  avg(oil) as avg_subcluster_oil,
  avg(other) as avg_subcluster_other,
  avg(noise) as avg_subcluster_noise
  from {features_out}
group by 1,2)
using(cluster_number, subcluster_number)),

-- ------------------------
-- -- Create new cluster_number selections using manually reviewed areas for reclassifying clusters labels
-- ------------------------ 
reviewed_wind as (
SELECT
  detect_id,
FROM
  {features_out}
CROSS JOIN
  sk_wind
WHERE
  ST_CONTAINS(geometry, ST_GEOGPOINT(clust_centr_lon, clust_centr_lat)) ),

reviewed_oil as (
SELECT
  detect_id,
FROM
  {features_out}
CROSS JOIN
  sk_oil
WHERE
  ST_CONTAINS(geometry, ST_GEOGPOINT(clust_centr_lon, clust_centr_lat)) ),

reviewed_other as (
SELECT
  detect_id,
FROM
  {features_out}
CROSS JOIN
  sk_other
WHERE
  ST_CONTAINS(geometry, ST_GEOGPOINT(clust_centr_lon, clust_centr_lat)) ),

not_infra as (
SELECT
  detect_id,
FROM
  {features_out}
CROSS JOIN
  sk_not_infra
WHERE
  ST_CONTAINS(geometry, ST_GEOGPOINT(clust_centr_lon, clust_centr_lat)) ),

maracaibo AS (
SELECT
  detect_id,
FROM
  {features_out}
CROSS JOIN
  regions
WHERE
  ST_CONTAINS(region_geometry, ST_GEOGPOINT(clust_centr_lon, clust_centr_lat))
  AND oil_region IN ('Lake Maracaibo')),

------------------------
-- these are based on the rule set: if a subsubcluster label is within an area, change to this or that
-- so in this case we grab the detect_id rather than subcluster_number. For the manually reviewed ones, we can just say whatever
-- is associated with a subcluster is this or that. For these, we rely on the different subsubcluster labels in a subcluster, so 
-- we have to select the detect id.
------------------------ 
oil_in_oil_polygon_subcluster AS (
SELECT
  detect_id,
FROM
  subcluster_labels
CROSS JOIN
  all_oil_polygons
WHERE
  subcluster_label = 'oil'
  AND ST_CONTAINS(geometry, ST_GEOGPOINT(clust_centr_lon, clust_centr_lat)) ),

reviewed_oil_to_unknown_subcluster as (
SELECT
  detect_id,
FROM
  subcluster_labels
CROSS JOIN
  oil_to_unknown
WHERE
  subcluster_label = 'oil'
  and ST_CONTAINS(geometry, ST_GEOGPOINT(clust_centr_lon, clust_centr_lat)) ),

oil_detections_outside_polygon_subcluster AS (
SELECT
  detect_id,
FROM
  subcluster_labels
WHERE
  subcluster_label = 'oil'
  AND detect_id NOT IN (
  SELECT
    detect_id
  FROM
    oil_in_oil_polygon_subcluster
where detect_id is not null) ),

other_in_oil_polygon_subcluster AS (
SELECT
  detect_id,
FROM
  subcluster_labels
CROSS JOIN
  all_oil_polygons
WHERE
  subcluster_label = 'other'
  AND ST_CONTAINS(geometry, ST_GEOGPOINT(clust_centr_lon, clust_centr_lat)) ),

other_detections_outside_oil_polygon_subcluster AS (
SELECT
  *
FROM
  subcluster_labels
WHERE
  subcluster_label = 'other'
  AND detect_id NOT IN (
  SELECT
    detect_id
  FROM
    other_in_oil_polygon_subcluster
where detect_id is not null) ),

wind_in_wind_regions_subcluster AS (
SELECT
  detect_id,
FROM
  subcluster_labels
CROSS JOIN
  all_wind_polygons
WHERE
  subcluster_label = 'wind'
  AND ST_CONTAINS(geometry, ST_GEOGPOINT(clust_centr_lon, clust_centr_lat)) ),

wind_detections_outside_wind_regions_subcluster AS (
SELECT
  detect_id,
FROM
  subcluster_labels
WHERE
  subcluster_label = 'wind'
  AND detect_id NOT IN (
  SELECT
    detect_id
  FROM
    wind_in_wind_regions_subcluster
where detect_id is not null) ),

all_oil_subcluster AS (
SELECT
  *
FROM
  subcluster_labels
WHERE
  subcluster_label = 'oil'),

all_wind_subcluster AS (
SELECT
  *
FROM
  subcluster_labels
WHERE
  subcluster_label = 'wind'),

oil_near_wind_subcluster AS (
SELECT
  a.detect_id,
FROM
  all_oil_subcluster a
CROSS JOIN
  all_wind_subcluster b
CROSS JOIN
  all_wind_polygons c
WHERE
  ST_DISTANCE(ST_GEOGPOINT(a.clust_centr_lon, a.clust_centr_lat), ST_GEOGPOINT(b.clust_centr_lon, b.clust_centr_lat)) <= 2 * 1000
  AND ST_WITHIN(ST_GEOGPOINT(a.clust_centr_lon, a.clust_centr_lat), c.geometry)
  AND a.subcluster_label = 'oil'),

new_probable_wind_subcluster AS (
SELECT
  distinct
  a.detect_id,
  a.clust_centr_lon,
  a.clust_centr_lat
FROM
  (select * from all_oil_subcluster 
  union all 
  select * from other_detections_outside_oil_polygon_subcluster) a
CROSS JOIN
  all_wind_subcluster b
CROSS JOIN
  all_wind_polygons c
WHERE
  ST_DISTANCE(ST_GEOGPOINT(a.clust_centr_lon, a.clust_centr_lat), ST_GEOGPOINT(b.clust_centr_lon, b.clust_centr_lat)) <= 2 * 1000
  AND ST_WITHIN(ST_GEOGPOINT(a.clust_centr_lon, a.clust_centr_lat), c.geometry)),

new_probable_wind_2_subcluster as (
SELECT
    distinct
    a.detect_id,
    a.clust_centr_lon,
    a.clust_centr_lat
  FROM
    (select * from all_oil_subcluster
    union all 
    select * from other_detections_outside_oil_polygon_subcluster) a
    CROSS JOIN
    (select detect_id, clust_centr_lon, clust_centr_lat
    from
    all_wind_subcluster
    union all
    select * from new_probable_wind_subcluster) b
  CROSS JOIN
    all_wind_polygons c
  WHERE
    ST_DISTANCE(ST_GEOGPOINT(a.clust_centr_lon, a.clust_centr_lat), ST_GEOGPOINT(b.clust_centr_lon, b.clust_centr_lat)) <= 2 * 1000
    AND ST_WITHIN(ST_GEOGPOINT(a.clust_centr_lon, a.clust_centr_lat), c.geometry)),

final_new_probable_wind_subcluster as (
SELECT
  distinct
    a.detect_id,
  FROM
    (select * from all_oil_subcluster 
    union all 
    select * from other_detections_outside_oil_polygon_subcluster) a
    CROSS JOIN
    (select detect_id, clust_centr_lon, clust_centr_lat
    from
    all_wind_subcluster
    union all
    select * from new_probable_wind_2_subcluster) b
  CROSS JOIN
    all_wind_polygons c
  WHERE
    ST_DISTANCE(ST_GEOGPOINT(a.clust_centr_lon, a.clust_centr_lat), ST_GEOGPOINT(b.clust_centr_lon, b.clust_centr_lat)) <= 2 * 1000
    AND ST_WITHIN(ST_GEOGPOINT(a.clust_centr_lon, a.clust_centr_lat), c.geometry)),

    noise_detections_subcluster as (
  select detect_id from subcluster_labels
  where subcluster_label = 'noise'
),

#subcluster labels reclassified
label_reclass_subcluster AS (
select * except(subcluster_label_reclassified),
--fix that the null detect_id aren't captured in the subqueries because there is no detect_id, so they need to get the reclass label of the other detections in subcluster
if(detect_id is null, first_value(subcluster_label_reclassified IGNORE NULLS) over (partition by cluster_number, subcluster_number order by subcluster_number), subcluster_label_reclassified) as subcluster_label_reclassified
from
  (
  SELECT
    *,
    CASE
      WHEN detect_id in (select detect_id from not_infra) THEN 'noise'
      WHEN detect_id in (select detect_id from reviewed_oil_to_unknown_subcluster) THEN 'unknown'
      WHEN detect_id in (select detect_id from reviewed_oil) THEN 'oil'
      WHEN detect_id in (select detect_id from reviewed_wind) THEN 'wind'
      WHEN detect_id in (select detect_id from reviewed_other) THEN 'unknown'
      WHEN detect_id in (select detect_id from maracaibo) THEN 'lake_maracaibo'
      WHEN detect_id in (select detect_id from final_new_probable_wind_subcluster) THEN 'probable_wind'
      WHEN detect_id in (select detect_id from oil_in_oil_polygon_subcluster) THEN 'oil'
      WHEN detect_id in (select detect_id from oil_detections_outside_polygon_subcluster) THEN 'possible_oil'
      WHEN detect_id in (select detect_id from other_in_oil_polygon_subcluster) THEN 'probable_oil'
      WHEN detect_id in (select detect_id from other_detections_outside_oil_polygon_subcluster) THEN 'unknown'
      WHEN detect_id in (select detect_id from wind_in_wind_regions_subcluster) THEN 'wind'
      WHEN detect_id in (select detect_id from wind_detections_outside_wind_regions_subcluster) THEN 'possible_wind'
      WHEN detect_id in (select detect_id from noise_detections_subcluster) THEN 'noise'
      WHEN detect_id is null THEN null -- these are filled in date gaps. Assign this most common value of cluster above
    ELSE
    'start'
  END
    AS subcluster_label_reclassified, 
  FROM
    subcluster_labels)),

-- -------------------

final as (
select distinct
a.detect_id,
wind,
oil,
other,
noise,
a.detect_lon,
a.detect_lat,
start_time,
end_time,
a.midpoint,
a.cluster_number,
cluster_detections,
a.subcluster_number,
subcluster_detections,
cluster_start_date,
cluster_end_date,
subcluster_start_date,
subcluster_end_date,
a.clust_centr_lat,
a.clust_centr_lon,
detection_label,
-- incorporate final aton osm edits
if (new_reclass_label is null, detection_label_reclassified, new_reclass_label) detection_label_reclassified,
cluster_label,
-- incorporate final aton osm edits
if (new_reclass_label is null, cluster_label_reclassified, new_reclass_label) cluster_label_reclassified,
subcluster_label,
-- incorporate final aton osm edits
if (new_reclass_label is null, subcluster_label_reclassified, new_reclass_label) subcluster_label_reclassified
from `{label_reclass}` a
left join `{clust_label_reclass}` b
on (a.cluster_number = b.cluster_number)
left join label_reclass_subcluster c
on (a.cluster_number = c.cluster_number and a.subcluster_number = c.subcluster_number)
left join (select cluster_number, new_reclass_label from `proj_global_sar.infra_reclass_points_aton_osm_jc`) d
on(a.cluster_number = d.cluster_number))

select distinct *
from final
order by cluster_number, subcluster_number, midpoint'''

# %%
print(q)

# %%
final = pd.read_gbq(q)

# %%
# push final reclassification table to gbq
date = 20231203
final_reclassification = f'scratch_pete.detect_comp_pred_clustered_reclassified_v{date}'
final.to_gbq(final_reclassification, if_exists = 'replace')

# %%
final_reclassification

# %%
