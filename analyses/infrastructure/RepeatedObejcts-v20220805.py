# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Develop table of repeated objects and join it with composite objects
#
# Composite objects are what we used in our paper, 

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mpcolors
import pandas as pd
import pyseas.maps as psm
import pyseas.contrib as psc
import pyseas.cm
# %matplotlib inline
import cartopy.crs as ccrs
import subprocess
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, date
from google.cloud import bigquery
import pandas as pd
# Construct a BigQuery client object.
client = bigquery.Client()

q = '''
select distinct format_timestamp("%Y-%m-%d", start_time) start_time,
format_timestamp("%Y-%m-%d", timestamp_add(start_time, interval 90  day)) midpoint,
format_timestamp("%Y-%m-%d", end_time) end_time,
from `proj_sentinel1_v20210924.detect_comp_raw_*` 
order by start_time'''
df = pd.read_gbq(q)

df.head()


# +
# halfw = timedelta(days=90)
# dates = []
# for year in range(2017,2023):
#     dates+= [date(year, k, 1) for k in range(1, 13)]
# windows = [
#     ((d - halfw).strftime("%Y-%m-%d"), (d + halfw).strftime("%Y-%m-%d")) for d in dates
# ]
# windows = windows[:-11]


# +
# windows[:-11]
# -

q_template = '''
CREATE TEMP FUNCTION startday() AS (DATE('{startdate}')); 
CREATE TEMP FUNCTION endday() AS (
  -- time window of 180 days  
  date(timestamp_add(timestamp(startday()), interval 180 day))
);
CREATE TEMP FUNCTION YYYYMMDD(d DATE) AS (
  # Format a date as YYYYMMDD
  # e.g. DATE('2018-01-01') => '20180101'
  FORMAT_DATE('%Y%m%d',
    d) );

with
----------------
-- detections table
----------------
detections_table as (
   select
    *, _partitiontime date 
   from 
     `project-id.proj_sentinel1_v20210924.detect_scene_match` 
   where 
    date(_partitiontime) between startday() and endday()
),
--
-- predictions of length and presence 
prediction_table as (
  select * 
  from 
    `project-id.proj_sentinel1_v20210924.detect_scene_pred_*`
  where 
    _table_suffix between YYYYMMDD(startday()) and YYYYMMDD(endday())
),


----------------
-- Combosite detections -- these are from the composite table
----------------
composite_table as (
   SELECT * FROM `project-id.proj_sentinel1_v20210924.detect_comp_raw_*`
   where _table_suffix = YYYYMMDD(startday())
),

----------------
-- sar detections with lengths
----------------
sar_detections_w_predictions as (
  select 
    detect_id, detect_lon, detect_lat,
    st_geogpoint(detect_lon, detect_lat) as pos,
    presence,
    length_m,
    date
  FROM 
    detections_table
  left join
    prediction_table
  using(detect_id)
),

-- count detections by h3 cell
-- h3 level 9 is 0.1 km2 on average in area
-- https://h3geo.org/docs/core-library/restable/
with_h3 as (
  select 
    jslibs.h3.ST_H3(pos,9) h3_9,
    detect_id,
    pos,
    detect_id
  from 
    sar_detections_w_predictions
),

-- get the avg location of detections in this h3 cell
h3_clusters as (
  select 
   h3_9, 
   st_centroid(st_union_agg(pos)) loc,
    count(*) repeats
  from 
    with_h3
  group by 
    h3_9 
  having 
    repeats >=5 
),


-- cluster these with dbscan requireing min distance of 100 and 
-- at least two detections. I reveiwed this and there should only
-- be, at most, three of these that are within 100m of one
-- another. That is, each hexigon is on average ~300m across, so 
-- you'll only get these close to one another if they are at the edge.
-- The idea is when a cluster is close to the edge of an h3 cell
clustered as (
  select 
    -- note I cast this as a string so that I can use it later as a unique id
    cast(ST_CLUSTERDBSCAN(loc,100, 2) over () as string) as cluster_num_100_2,
    h3_9,
  from 
    h3_clusters),

 
-- Now get the average location of each of these clusters.
-- If the cluster_num_100_2 is null, that is because there was
-- no other clusters within 100m, and thus the repeat_object_id
-- will be the h3 cell. If it is part of a cluster, the id will
-- be the cluster number
repeat_objects as (
  select 
    ifnull(cluster_num_100_2, h3_9) repeat_object_id, 
    st_centroid(st_union_agg(pos)) loc
  from 
    with_h3
  join
    clustered
  using
    (h3_9)
  group by 
    repeat_object_id
),

-- Now make sure that these are actually repeated objects.
-- Make sure that there are at least five sar detections
-- within 100 meters
repeat_table as (
select 
  repeat_object_id,
  st_x(loc) lon,
  st_y(loc) lat,
  count(*) detections
from 
  repeat_objects
cross join
  sar_detections_w_predictions
where 
  st_distance(loc, pos) < 100
group by
  repeat_object_id,
  lat, lon
having detections >=5
),


-----------------
-- Get each composite oboject to repeat object pair that is within 100m of one
----------------
close_objects as (
select 
  detect_id, 
  repeat_object_id,
  detections,
  st_distance(st_geogpoint(detect_lon, detect_lat), st_geogpoint(lon, lat)) dist
from 
  composite_table
cross join
  repeat_table
where 
  st_distance(st_geogpoint(detect_lon, detect_lat), st_geogpoint(lon, lat)) < 100
),

--------------------
-- a composite might be within 100m of multiple repeat_objects (although very unlikely).
-- Choose the pair with the more objects close to the repeated object
-----------------
top_close as 
(
  select detect_id,repeat_object_id from 
    (select 
      *, row_number() over (partition by detect_id order by detections desc) row_number
     from 
       close_objects
    )
  where row_number = 1 
),

--------
-- Union the the tables
------
all_repeats_and_composites as (
select   
  detect_id, 
  repeat_object_id,
  detect_lon,
  detect_lat
from 
  composite_table
left join
  top_close
using(detect_id)
union all
select 
  null as detect_id,
  repeat_object_id,
  lon,
  lat
from 
  repeat_table
where 
  repeat_object_id not in (select repeat_object_id from close_objects)
),

-- How many of the objects are nosie?
-- And what is the average length of them?
-- Join this back on the predictions table. Note that 
-- a small number of predictions don't have values
detections_crossjoin as (
  select
    a.detect_lon,
    a.detect_lat,
    count(*) detections, 
    sum(if(presence<.6,1,0)) bad_detections,
    avg(length_m) avg_length_m
  from 
    all_repeats_and_composites a
  cross join
    sar_detections_w_predictions b
  where 
    st_distance(st_geogpoint(a.detect_lon, a.detect_lat), 
                    st_geogpoint(b.detect_lon, b.detect_lat))<100
  group by 
    detect_lon, detect_lat
)

--
-- put it all together
--
select
  timestamp_add(timestamp(startday()), interval 90 day) midpoint,
  timestamp(startday()) start_time, 
  timestamp(endday()) end_time,
  detect_id, 
  repeat_object_id,
  detect_lon,
  detect_lat,
  ifnull(detections, 0) detections,
  ifnull(bad_detections,0) bad_detections,
  avg_length_m 
from 
  all_repeats_and_composites
left join
  detections_crossjoin
using(detect_lon, detect_lat)'''


def query_to_table(query, table_id, max_retries=100, retry_delay=60):
    for _ in range(max_retries):

        config = bigquery.QueryJobConfig(
            destination=table_id, write_disposition="WRITE_TRUNCATE"
        )

        job = client.query(query, job_config=config)

        if job.error_result:
            err = job.error_result["reason"]
            msg = job.error_result["message"]
            if err == "rateLimitExceeded":
                print(f"retrying... {msg}")
                time.sleep(retry_delay)
                continue
            elif err == "notFound":
                print(f"skipping... {msg}")
                return
            else:
                raise RuntimeError(msg)

        job.result()  # wait to complete
        print(f"completed {table_id}")
        return

    raise RuntimeError("max_retries exceeded")


df.head()

with ThreadPoolExecutor(max_workers=50) as e:
    for index, row in df.iterrows():
        t = row.start_time.replace("-","")
        q = q_template.format(startdate=row.start_time)
        table_id = f"project-id.proj_global_sar.detect_comp_repeat_{t}"
        e.submit(query_to_table, q, table_id)

# +
# import pyperclip
# pyperclip.copy(q)

# +
# for w in windows:

#     t = w[0].replace("-","")
#     table_id = f"project-id.proj_global_sar.detect_comp_repeat_{t}"

#     config = bigquery.QueryJobConfig(
#         destination=table_id, write_disposition="WRITE_TRUNCATE"
#     )

#     t = w[0]
#     job = client.query(q_template.format(startdate=t), job_config=config)

#     job.result()
# # # Now get possible ambiguities
# #
# -

#
# # $45 query!

q = '''WITH
-----------------
-- overpass raster
---------------
  overpass_rasters AS (
  SELECT
    *,
    _partitiontime AS date
  FROM
    proj_sentinel1_v20210924.detect_foot_raster_200),
    --
    --
  repeats_composits AS (
  SELECT
    ROW_NUMBER() OVER(PARTITION BY midpoint ORDER BY detect_lon,detect_lat, rand()) ROW,
    midpoint,
    start_time,
    end_time,
    detect_id,
    repeat_object_id,
    detect_lon,
    detect_lat,
    bad_detections,
    avg_length_m
  FROM
    `project-id.proj_global_sar.detect_comp_repeat_*` ),


  detections AS (
  SELECT
    ROW,
    midpoint,
    start_time,
    end_time,
    scene_id,
    score > 1e-3 AS matched,
    a.detect_lat,
    a.detect_lon
  FROM
    repeats_composits a
  CROSS JOIN
    `project-id.proj_sentinel1_v20210924.detect_scene_match` b
  WHERE
    _partitiontime BETWEEN start_time
    AND end_time
    AND st_distance(st_geogpoint(a.detect_lon,
        a.detect_lat),
      st_geogpoint(b.detect_lon,
        b.detect_lat))<100 ),

  --
  --
  --
  overpasses_with_angle AS (
  SELECT
    ROW,
    midpoint,
    CAST(look_angle AS int64) AS look_angle,
    COUNT(*) overpasses
  FROM
    repeats_composits
  JOIN
    overpass_rasters
  ON
    FLOOR(detect_lon*200) = lon_index
    AND FLOOR(detect_lat*200) = lat_index
  WHERE
    date BETWEEN start_time
    AND end_time
  GROUP BY
    ROW,
    midpoint,
    look_angle ),
    --
    --
  detections_with_angle AS (
  SELECT
    ROW,
    midpoint,
    CAST(look_angle AS int64) AS look_angle,
    COUNT(*) detections,
    SUM(
    IF
      (matched,
        1,
        0)) matched_detections,
        --
        --
  FROM
    detections a
  JOIN
    overpass_rasters b
  ON
    FLOOR(detect_lon*200) = lon_index
    AND FLOOR(detect_lat*200) = lat_index
    AND a.scene_id = b.scene_id
  WHERE
    date BETWEEN start_time
    AND end_time
  GROUP BY
    ROW,
    midpoint,
    look_angle ),
    --
    --
  overpasses_and_detections AS (
  SELECT
    ROW,
    midpoint,
    look_angle,
    overpasses,
    ifnull(detections,
      0) detections,
    ifnull(matched_detections,
      0) matched_detections
  FROM
    overpasses_with_angle
  LEFT JOIN
    detections_with_angle
  USING
    (ROW,
      midpoint,
      look_angle) ),
      --
      --
  overpasses_and_detections_grouped AS (
  SELECT
    ROW,
    midpoint,
    SUM(overpasses) overpasses,
    SUM(detections) detections,
    SUM(matched_detections) matched_detections,
    ARRAY_AGG(STRUCT(look_angle,
        overpasses,
        detections)) AS by_angle
  FROM
    overpasses_and_detections
  GROUP BY
    ROW,
    midpoint ),
    --
    --
  detected_rarely_or_often AS (
  SELECT
    ROW,
    midpoint,
    detections/overpasses > .9 mostly_detected,
    detections/overpasses < .1 rarely_detected
  FROM (
    SELECT
      ROW,
      midpoint,
      b.look_angle,
      b.overpasses,
      b.detections,
      ARRAY_LENGTH(by_angle) num_angles
    FROM
      overpasses_and_detections_grouped
    CROSS JOIN
      UNNEST(by_angle) AS b
    WHERE
      ARRAY_LENGTH(by_angle) > 1 )
  WHERE
    overpasses > 5),
    --
    --
  detected_rarely_table AS (
  SELECT
    DISTINCT ROW,
    midpoint
  FROM
    detected_rarely_or_often
  WHERE
    rarely_detected),
    --
    --
  detected_mostly_table AS (
  SELECT
    DISTINCT ROW,
    midpoint
  FROM
    detected_rarely_or_often
  WHERE
    mostly_detected),
    --
    --
  with_ambiguity AS (
  SELECT
    *,
    b.ROW IS NOT NULL
    AND c.ROW IS NOT NULL AS possible_ambiguity
  FROM
    overpasses_and_detections_grouped
  LEFT JOIN
    detected_rarely_table b
  USING
    (ROW,
      midpoint)
  LEFT JOIN
    detected_mostly_table c
  USING
    (ROW,
      midpoint) ),
  --    
  --    
  eez_table as (
  select 
     midpoint, row, 
     array_agg(ifnull(MRGID,0)) as MRGID, 
     array_agg(ifnull(ISO_TER1,"")) as ISO_TER1, 
     array_agg(ifnull(TERRITORY1,"")) AS TERRITORY1
   from 
     (select distinct row, midpoint, detect_lon, detect_lat from repeats_composits)
   CROSS JOIN
     (select wkt, MRGID, ISO_TER1, TERRITORY1 from `project-id.minderoo.marine_regions_v11`)
   WHERE
     ST_CONTAINS(SAFE.ST_GEOGFROMTEXT(wkt),ST_GEOGPOINT(detect_lon, detect_lat) )
   group by midpoint, row
 )
      
      --
      --
SELECT
  row as id,
  midpoint,
  start_time,
  end_time,
  detect_id,
  repeat_object_id,
  detect_lon,
  detect_lat,
  overpasses,
  detections,
  bad_detections,
  avg_length_m,
  matched_detections,
  by_angle,
  possible_ambiguity,
  MRGID,
  ISO_TER1,
  TERRITORY1
FROM
  repeats_composits
LEFT JOIN
  with_ambiguity
USING
  (midpoint,
    ROW)
left join
  eez_table
using(midpoint,row)
'''

# +
table_id = f"project-id.proj_global_sar.infrastructure_repeat_cat_6m_v20220805"

config = bigquery.QueryJobConfig(
    destination=table_id, write_disposition="WRITE_TRUNCATE"
)

### uncomment if you want to spend $45

# job = client.query(q, job_config=config)

# job.result()

# -


def update_table_description(table_id, table_description):

    client = bigquery.Client()
    
    #get table
    table = client.get_table(table_id)

    # Set table description
    table.description = table_description

    #update table with description
    client.update_table(table, ["description"]) 


# +
description = '''
table combining repeated objects with composite detections

query to generate it:
''' +q 

update_table_description(table_id, description)
# -







