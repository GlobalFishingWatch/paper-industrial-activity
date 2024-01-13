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

# # Create Detections With Overpasses
#
#

# +
import sys
sys.path.append('../utils')

from bigquery_helper_functions import (
    update_table_description,
    query_to_table,
)

from proj_id import project_id
import pyperclip
# -

# ### Version 20220509

q = f'''with 
-- get lengths and presence for each detection
raw_detections as (
  select 
    _partitiontime date,
    *,
    floor(detect_lat * 200) lat_index,
    floor(detect_lon * 200) lon_index,
    st_geogpoint(detect_lon, detect_lat) pos,
    extract(year from _partitiontime) year,
    extract(month from _partitiontime) month
  FROM 
    `{project_id}.proj_sentinel1_v20210924.detect_scene_match` 
  join
    (select 
      detect_id, avg(presence) presence, 
      avg(length_m) length_m 
    from 
      `{project_id}.proj_sentinel1_v20210924.detect_scene_pred_*`
    group by detect_id
    )
  using(detect_id)
    -- where _partitiontime between "2017-01-01" and "2021-12-31"
 ),

-- overpasses table, but filtered for only scenes with scene_quality > .5
raw_overpasses as (
  select lat_index, lon_index, overpasses, year from 
  `proj_global_sar.overpasses_200_by_year_filtered_v20220508`
),

-- get overpasses across five years
overpasses_2017_2021_table as 
(
  select 
    lat_index, lon_index, 
    sum(overpasses) overpasses_2017_2021
  from 
    raw_overpasses
  where 
    year between 2017 and 2021
  group by 
    lat_index, lon_index
),

-- scene quality table
scene_quality as (
  select 
    scene_id,
    count(*) scene_detections,
    sum(if(presence < .7 or presence is null,1,0)) bad_detections
  from 
    raw_detections
  group by 
    scene_id
 ),

repeated_objects_center180_100m as (
  select 
  a.detect_id,
  count(*) repeats_100m_180days_center
  from 
    raw_detections a
  cross join 
    raw_detections b
  where st_distance(a.pos, b.pos) < 100
    and a.detect_id != b.detect_id
    and abs(timestamp_diff(a.date, b.date, day)) <= 90 
  group by a.detect_id
 -- having repeats >=3
),

repeated_objects_forward180_100m as (
  select 
  a.detect_id,
  count(*) repeats_100m_180days_forward
  from 
    raw_detections a
  cross join 
    raw_detections b
  where st_distance(a.pos, b.pos) < 100
    and a.detect_id != b.detect_id
    and timestamp_diff(b.date, a.date, day) between 0 and 180 
  group by a.detect_id
),

-- Get number of repeats... with 6 month window looking backwards,
-- centered, and looking forward. 

repeated_objects_back180_100m as (
  select 
    a.detect_id,
    count(*) repeats_100m_180days_back
  from 
    raw_detections a
  cross join 
    raw_detections b
  where 
    st_distance(a.pos, b.pos) < 100
    and a.detect_id != b.detect_id
    and timestamp_diff(a.date, b.date, day) between 0 and 180 
  group by a.detect_id
),

repated_detections as 
(
  select distinct detect_id 
    from (   
      select detect_id from 
      repeated_objects_center180_100m
    union all
    select detect_id from 
      repeated_objects_back180_100m
    union all
    select detect_id from 
      repeated_objects_forward180_100m
    )
)

select
  * except(lat_index, lon_index, 
  repeats_100m_180days_back,
  repeats_100m_180days_center,
  repeats_100m_180days_forward, overpasses, overpasses_2017_2021),
  1 - bad_detections/scene_detections scene_quality,
  ifnull(repeats_100m_180days_back, 0) repeats_100m_180days_back,
  ifnull(repeats_100m_180days_center, 0) repeats_100m_180days_center,
  ifnull(repeats_100m_180days_forward, 0) repeats_100m_180days_forward,
  overpasses as overpasses_year,
  overpasses_2017_2021
from
  raw_detections a
  left join
scene_quality
  using(scene_id)
left join
  repeated_objects_center180_100m 
using(detect_id)
left join
  repeated_objects_forward180_100m
using(detect_id)
left join
  repeated_objects_back180_100m
using(detect_id)
left join
  raw_overpasses
using(lat_index, lon_index, year)
left join
  overpasses_2017_2021_table
using(lat_index, lon_index)'''

# +
table_id = f'{project_id}.proj_global_sar.detections_w_overpasses_v20220509'
# query_to_table(q, table_id)

description = '''detections with overpasses. 

The query to make it: 
''' + q

update_table_description(table_id, 
                        description)
# -

# ### Version 20220805

q = f'''with 
yearly_overpasses as (
  select lat_index, lon_index, overpasses, year from 
  `proj_global_sar.overpasses_200_by_year_filtered_v20220508`
),

quarterly_overpasses as (
select 
  quarter,
  date as date_quarter,
  overpasses,
  lat_index, 
  lon_index 
from
  proj_global_sar.overpasses_200_byquarter_v20220805
),

quarters_with_overpass_table as (
  select 
    lat_index,
    lon_index,
    count(*) quarters_with_overpass,
  from 
    quarterly_overpasses
  group by
    lat_index,
    lon_index
),

every24day_overpasses as (
  select date as date_24, lat_index, lon_index, overpasses from 
  proj_global_sar.overpasses_200_every24days_v20220805
),

periods24_with_overpass_table as 
(
  select 
    count(*) periods24_with_overpass, 
    lat_index, lon_index 
  from 
    every24day_overpasses
  group by lat_index, lon_index
),

years_with_overpasses_table as (
  select 
    count(*) years_with_overpasses, 
    lat_index, lon_index 
  from 
    yearly_overpasses
  group by lat_index, lon_index
),

-- get overpasses across five years
overpasses_2017_2021_table as 
(
  select 
    lat_index, lon_index, 
    sum(overpasses) overpasses_2017_2021
  from 
    yearly_overpasses
  where 
    year between 2017 and 2021
  group by 
    lat_index, lon_index
),

-- get lengths and presence for each detection
raw_detections as (
  select 
    _partitiontime date,
    extract(year from _partitiontime)*10 + floor(extract(dayofyear from _partitiontime)/366.5*4) quarter,
    timestamp_add( timestamp("2017-01-01"),
       interval
        cast( floor(timestamp_diff(_partitiontime,  
                                    timestamp("2017-01-01"),
                                     day)/24)*24 as int64) day) as date_24,
    *,
    floor(detect_lat * 200) lat_index,
    floor(detect_lon * 200) lon_index,
    st_geogpoint(detect_lon, detect_lat) pos,
    extract(year from _partitiontime) year,
    extract(month from _partitiontime) month
  FROM 
    `{project_id}.proj_sentinel1_v20210924.detect_scene_match` 
  join
    (select 
      detect_id, avg(presence) presence, 
      avg(length_m) length_m 
    from 
      `{project_id}.proj_sentinel1_v20210924.detect_scene_pred_*`
    group by detect_id
    )
  using(detect_id)
    -- where _partitiontime between "2017-01-01" and "2021-12-31"
 ),

eez_table as (
  select 
    detect_id,
    if(array_length(ISO_TER1)>0, ISO_TER1[offset(0)],"none") eez_iso3
  from
    (
    select 
    detect_id, 
     array_agg(ifnull(ISO_TER1,"")) as ISO_TER1 
  from raw_detections
    CROSS JOIN
  (select wkt, MRGID, ISO_TER1, TERRITORY1 from `{project_id}.minderoo.marine_regions_v11`)
  WHERE
    ST_CONTAINS(SAFE.ST_GEOGFROMTEXT(wkt),ST_GEOGPOINT(detect_lon,detect_lat) )
  group by 
    detect_id)
),


-- scene quality table
scene_quality as (
  select 
    scene_id,
    count(*) scene_detections,
    sum(if(presence < .7 or presence is null,1,0)) bad_detections
  from 
    raw_detections
  group by 
    scene_id
 ),

repeated_objects_center180_100m as (
  select 
  a.detect_id,
  count(*) repeats_100m_180days_center
  from 
    raw_detections a
  cross join 
    raw_detections b
  where st_distance(a.pos, b.pos) < 100
    and a.detect_id != b.detect_id
    and abs(timestamp_diff(a.date, b.date, day)) <= 90 
  group by a.detect_id
 -- having repeats >=3
),

repeated_objects_forward180_100m as (
  select 
  a.detect_id,
  count(*) repeats_100m_180days_forward
  from 
    raw_detections a
  cross join 
    raw_detections b
  where st_distance(a.pos, b.pos) < 100
    and a.detect_id != b.detect_id
    and timestamp_diff(b.date, a.date, day) between 0 and 180 
  group by a.detect_id
),

-- Get number of repeats... with 6 month window looking backwards,
-- centered, and looking forward. 

repeated_objects_back180_100m as (
  select 
    a.detect_id,
    count(*) repeats_100m_180days_back
  from 
    raw_detections a
  cross join 
    raw_detections b
  where 
    st_distance(a.pos, b.pos) < 100
    and a.detect_id != b.detect_id
    and timestamp_diff(a.date, b.date, day) between 0 and 180 
  group by a.detect_id
),

repated_detections as 
(
  select distinct detect_id 
    from (   
      select detect_id from 
      repeated_objects_center180_100m
    union all
    select detect_id from 
      repeated_objects_back180_100m
    union all
    select detect_id from 
      repeated_objects_forward180_100m
    )
)

select
  date,
  year,
  date_quarter,
  date_24,
  detect_id,
  scene_id,
  ssvid,
  detect_timestamp,
  detect_lat,
  detect_lon,
  score,
  confidence,
  presence,
  length_m,
  scene_detections,
  bad_detections,
  1 - bad_detections/scene_detections scene_quality,
  ifnull(repeats_100m_180days_back, 0) repeats_100m_180days_back,
  ifnull(repeats_100m_180days_center, 0) repeats_100m_180days_center,
  ifnull(repeats_100m_180days_forward, 0) repeats_100m_180days_forward,
  overpasses_2017_2021,
  yearly_overpasses.overpasses as overpasses_year,
  quarterly_overpasses.overpasses as overpasses_quarter,
  every24day_overpasses.overpasses as overpasses_24day,
  years_with_overpasses,
  quarters_with_overpass,
  periods24_with_overpass,
  eez_iso3
from
  raw_detections a
  left join
scene_quality
  using(scene_id)
left join
  repeated_objects_center180_100m 
using(detect_id)
left join
  repeated_objects_forward180_100m
using(detect_id)
left join
  repeated_objects_back180_100m
using(detect_id)
left join
  yearly_overpasses
using(lat_index, lon_index, year)
left join
  overpasses_2017_2021_table
using(lat_index, lon_index)
left join
  quarterly_overpasses
using(lat_index, lon_index, quarter)
left join
  quarters_with_overpass_table
using(lat_index, lon_index)
left join
  every24day_overpasses
using(date_24, lat_index, lon_index)
left join 
  periods24_with_overpass_table
using(lat_index, lon_index)
left join
  years_with_overpasses_table
using(lat_index, lon_index)
left join
  eez_table
using(detect_id)


'''

# +
table_id = f'{project_id}.proj_global_sar.detections_w_overpasses_v20220805'
query_to_table(q, table_id)

description = '''detections with overpasses, including overpasses by quarter and by 24 days

The query to make it: 
''' + q

update_table_description(table_id, 
                        description)
# -

# ## version 20220929
#
# This version updates the table with a column `periods24_with_overpass_june2017_dec2021`, which is the number of 24 day periods between june 18 2017 and december 11 2021 that have a least one overpass. It also includes a column of having at least 30 overpasses each year between 2018 adn 2021.

q = '''with 
yearly_overpasses as (
  select lat_index, lon_index, overpasses, year from 
  `proj_global_sar.overpasses_200_by_year_filtered_v20220508`
),

every24day_overpasses as (
  select date as date_24, lat_index, lon_index, overpasses from 
  proj_global_sar.overpasses_200_every24days_v20220805
),

periods24_with_overpass_table as 
(
  select 
    count(*) periods24_with_overpass_june2017_dec2021, 
    lat_index, lon_index 
  from 
    every24day_overpasses
  where date_24 between "2017-06-18" and "2021-12-11"
  group by lat_index, lon_index
),

at_least_30overpasses_2018_2021 as (
select 
  lat_index, lon_index, count(*) number
from 
  yearly_overpasses
where 
  overpasses >= 30 
and 
  year between 2018 and 2021
group by 
  lat_index, lon_index
having 
  number = 4 -- at least four years with > 30 overpasses
)


select 
  * except(number, lat_index, lon_index),
  b.lat_index is not null as over_30_overpasses_eachyear_2018_2021
from 
  proj_global_sar.detections_w_overpasses_v20220805 a
left join 
  at_least_30overpasses_2018_2021 b
on 
  floor(detect_lat*200) = b.lat_index
  and floor(detect_lon*200) = b.lon_index
left join 
  periods24_with_overpass_table c
on 
  floor(detect_lat*200) = c.lat_index
  and floor(detect_lon*200) = c.lon_index'''

# +
table_id = f'{project_id}.proj_global_sar.detections_w_overpasses_v20220929'
query_to_table(q, table_id)

description = '''detections with overpasses, including overpasses by quarter and by 24 days. Added the number of 24 day periods with a detection from mid 2017 to dec 2021

The query to make it: 
''' + q

update_table_description(table_id, 
                        description)
# -
# ## Version 20230215
#
# updated version that uses new matches, updates EEZs, and adds in doppler regions

# +
q = f'''with 
yearly_overpasses as (
  select lat_index, lon_index, overpasses, year from 
  `proj_global_sar.overpasses_200_by_year_filtered_v20220508`
),

quarterly_overpasses as (
select 
  quarter,
  date as date_quarter,
  overpasses,
  lat_index, 
  lon_index 
from
  proj_global_sar.overpasses_200_byquarter_v20220805
),

quarters_with_overpass_table as (
  select 
    lat_index,
    lon_index,
    count(*) quarters_with_overpass,
  from 
    quarterly_overpasses
  group by
    lat_index,
    lon_index
),

every24day_overpasses as (
  select date as date_24, lat_index, lon_index, overpasses from 
  proj_global_sar.overpasses_200_every24days_v20220805
),

periods24_with_overpass_table as 
(
  select 
    count(*) periods24_with_overpass, 
    lat_index, lon_index 
  from 
    every24day_overpasses
  group by lat_index, lon_index
),

years_with_overpasses_table as (
  select 
    count(*) years_with_overpasses, 
    lat_index, lon_index 
  from 
    yearly_overpasses
  group by lat_index, lon_index
),

-- get overpasses across five years
overpasses_2017_2021_table as 
(
  select 
    lat_index, lon_index, 
    sum(overpasses) overpasses_2017_2021
  from 
    yearly_overpasses
  where 
    year between 2017 and 2021
  group by 
    lat_index, lon_index
),

score_ave_table as (
select
  detect_id,
  score as score_ave,
  ssvid as ssvid_ave,
  confidence as confidence_ave
from proj_global_sar.matched_ave
),

score_mult_table as (
select
  detect_id,
  score as score_mult,
  ssvid as ssvid_mult,
  confidence as confidence_mult
from proj_global_sar.matched_mult
),

score_mult_recall_table as (
select
  detect_id,
  score as score_mult_recall,
  ssvid as ssvid_mult_recall,
  confidence as confidence_mult_recall
from proj_global_sar.matched_mult_recall
),


score_mult_recall_length_table as (
select
  detect_id,
  score as score_mult_recall_length,
  ssvid as ssvid_mult_recall_length,
  confidence as confidence_mult_recall_length
from proj_global_sar.matched_mult_recall_length
),

road_doppler_table as (
  select 
  detect_id,
  in_road_doppler,
  in_road_doppler2
  from
  proj_global_sar.doppler_road_area_detections
),


-- get lengths and presence for each detection
raw_detections as (
  select 
    _partitiontime date,
    extract(year from _partitiontime)*10 + floor(extract(dayofyear from _partitiontime)/366.5*4) quarter,
    timestamp_add( timestamp("2017-01-01"),
       interval
        cast( floor(timestamp_diff(_partitiontime,  
                                    timestamp("2017-01-01"),
                                     day)/24)*24 as int64) day) as date_24,
    *,
      format("lon:%+07.2f_lat:%+07.2f", 
    detect_lon, 
    detect_lat) as gridcode, -- for matching
    floor(detect_lat * 200) lat_index,
    floor(detect_lon * 200) lon_index,
    st_geogpoint(detect_lon, detect_lat) pos,
    extract(year from _partitiontime) year,
    extract(month from _partitiontime) month
  FROM 
    `{project_id}.proj_sentinel1_v20210924.detect_scene_match` 
  join
    (select 
      detect_id, avg(presence) presence, 
      avg(length_m) length_m 
    from 
      `{project_id}.proj_sentinel1_v20210924.detect_scene_pred_*`
    group by detect_id
    )
  using(detect_id)
    -- where _partitiontime between "2017-01-01" and "2021-12-31"
 ),

regions as (
SELECT 
  regions.eez as eez_array, 
  if(array_length(regions.eez)>0, regions.eez[ordinal(1)], null) MRGID,
  gridcode 
FROM 
  `{project_id}.pipe_static.spatial_measures_20201105` 
),

eez_table as (
select 
  detect_id,
  eez_array,
  ISO_TER1 as eez_iso3
from 
  raw_detections
join
  regions
using(gridcode)
left join
  proj_global_sar.eez_info_v11
using(MRGID)
),

-- scene quality table
scene_quality as (
  select 
    scene_id,
    count(*) scene_detections,
    sum(if(presence < .7 or presence is null,1,0)) bad_detections
  from 
    raw_detections
  group by 
    scene_id
 ),

repeated_objects_center180_100m as (
  select 
  a.detect_id,
  count(*) repeats_100m_180days_center
  from 
    raw_detections a
  cross join 
    raw_detections b
  where st_distance(a.pos, b.pos) < 100
    and a.detect_id != b.detect_id
    and abs(timestamp_diff(a.date, b.date, day)) <= 90 
  group by a.detect_id
 -- having repeats >=3
),

repeated_objects_forward180_100m as (
  select 
  a.detect_id,
  count(*) repeats_100m_180days_forward
  from 
    raw_detections a
  cross join 
    raw_detections b
  where st_distance(a.pos, b.pos) < 100
    and a.detect_id != b.detect_id
    and timestamp_diff(b.date, a.date, day) between 0 and 180 
  group by a.detect_id
),

-- Get number of repeats... with 6 month window looking backwards,
-- centered, and looking forward. 

repeated_objects_back180_100m as (
  select 
    a.detect_id,
    count(*) repeats_100m_180days_back
  from 
    raw_detections a
  cross join 
    raw_detections b
  where 
    st_distance(a.pos, b.pos) < 100
    and a.detect_id != b.detect_id
    and timestamp_diff(a.date, b.date, day) between 0 and 180 
  group by a.detect_id
),

repated_detections as 
(
  select distinct detect_id 
    from (   
      select detect_id from 
      repeated_objects_center180_100m
    union all
    select detect_id from 
      repeated_objects_back180_100m
    union all
    select detect_id from 
      repeated_objects_forward180_100m
    )
),

final_table as 
(select
  date,
  year,
  date_quarter,
  date_24,
  detect_id,
  scene_id,
  detect_timestamp,
  detect_lat,
  detect_lon,
  score,
  ssvid,
  confidence,
  score_ave,
  ssvid_ave,
  confidence_ave,
  score_mult,
  ssvid_mult,
  confidence_mult,
  score_mult_recall,
  ssvid_mult_recall,
  confidence_mult_recall,
  score_mult_recall_length,
  ssvid_mult_recall_length,
  confidence_mult_recall_length,
  presence,
  length_m,
  scene_detections,
  bad_detections,
  1 - bad_detections/scene_detections scene_quality,
  ifnull(repeats_100m_180days_back, 0) repeats_100m_180days_back,
  ifnull(repeats_100m_180days_center, 0) repeats_100m_180days_center,
  ifnull(repeats_100m_180days_forward, 0) repeats_100m_180days_forward,
  overpasses_2017_2021,
  yearly_overpasses.overpasses as overpasses_year,
  quarterly_overpasses.overpasses as overpasses_quarter,
  every24day_overpasses.overpasses as overpasses_24day,
  years_with_overpasses,
  quarters_with_overpass,
  periods24_with_overpass,
  eez_iso3,
  eez_array,
  ifnull(in_road_doppler, False) in_road_doppler,
  ifnull(in_road_doppler2, False) in_road_doppler2,
from
  raw_detections a
  left join
scene_quality
  using(scene_id)
left join
  repeated_objects_center180_100m 
using(detect_id)
left join
  repeated_objects_forward180_100m
using(detect_id)
left join
  repeated_objects_back180_100m
using(detect_id)
left join
  yearly_overpasses
using(lat_index, lon_index, year)
left join
  overpasses_2017_2021_table
using(lat_index, lon_index)
left join
  quarterly_overpasses
using(lat_index, lon_index, quarter)
left join
  quarters_with_overpass_table
using(lat_index, lon_index)
left join
  every24day_overpasses
using(date_24, lat_index, lon_index)
left join 
  periods24_with_overpass_table
using(lat_index, lon_index)
left join
  years_with_overpasses_table
using(lat_index, lon_index)
left join
  eez_table
using(detect_id)
left join
  score_ave_table
using(detect_id)
left join
  score_mult_table
using(detect_id)
left join
  score_mult_recall_table
using(detect_id)
left join
  score_mult_recall_length_table
using(detect_id)
left join
  road_doppler_table
using(detect_id)
)

select * from final_table'''

table_id = f'{project_id}.proj_global_sar.detections_w_overpasses_v20230215'
query_to_table(q, table_id)

description = '''detections with overpasses, uses new matches, updates EEZs, and adds in doppler regions from roads. 

The query to make it: 
''' + q

update_table_description(table_id, 
                        description)
# -
# ## Version 20230420
#
# exactly the same as version 20230215, but adding in two different ways to score the 

# +
q = f"""with 
yearly_overpasses as (
  select lat_index, lon_index, overpasses, year from 
  `proj_global_sar.overpasses_200_by_year_filtered_v20220508`
),

quarterly_overpasses as (
select 
  quarter,
  date as date_quarter,
  overpasses,
  lat_index, 
  lon_index 
from
  proj_global_sar.overpasses_200_byquarter_v20220805
),

quarters_with_overpass_table as (
  select 
    lat_index,
    lon_index,
    count(*) quarters_with_overpass,
  from 
    quarterly_overpasses
  group by
    lat_index,
    lon_index
),

every24day_overpasses as (
  select date as date_24, lat_index, lon_index, overpasses from 
  proj_global_sar.overpasses_200_every24days_v20220805
),

periods24_with_overpass_table as 
(
  select 
    count(*) periods24_with_overpass, 
    lat_index, lon_index 
  from 
    every24day_overpasses
  group by lat_index, lon_index
),

years_with_overpasses_table as (
  select 
    count(*) years_with_overpasses, 
    lat_index, lon_index 
  from 
    yearly_overpasses
  group by lat_index, lon_index
),

-- get overpasses across five years
overpasses_2017_2021_table as 
(
  select 
    lat_index, lon_index, 
    sum(overpasses) overpasses_2017_2021
  from 
    yearly_overpasses
  where 
    year between 2017 and 2021
  group by 
    lat_index, lon_index
),

score_ave_table as (
select
  detect_id,
  score as score_ave,
  ssvid as ssvid_ave,
  confidence as confidence_ave
from proj_global_sar.matched_ave
),

score_mult_table as (
select
  detect_id,
  score as score_mult,
  ssvid as ssvid_mult,
  confidence as confidence_mult
from proj_global_sar.matched_mult
),

score_mult_recall_table as (
select
  detect_id,
  score as score_mult_recall,
  ssvid as ssvid_mult_recall,
  confidence as confidence_mult_recall
from proj_global_sar.matched_mult_recall
),


score_mult_recall_length_table as (
select
  detect_id,
  score as score_mult_recall_length,
  ssvid as ssvid_mult_recall_length,
  confidence as confidence_mult_recall_length
from proj_global_sar.matched_mult_recall_length
),


score_ave_length_table as (
select
  detect_id,
  score as score_ave_length,
  ssvid as ssvid_ave_length,
  confidence as confidence_ave_length
from proj_global_sar.matched_ave_length
),


score_mult_length_table as (
select
  detect_id,
  score as score_mult_length,
  ssvid as ssvid_mult_length,
  confidence as confidence_mult_length
from proj_global_sar.matched_mult_length
),


road_doppler_table as (
  select 
  detect_id,
  in_road_doppler,
  in_road_doppler2
  from
  proj_global_sar.doppler_road_area_detections
),


-- get lengths and presence for each detection
raw_detections as (
  select 
    _partitiontime date,
    extract(year from _partitiontime)*10 + floor(extract(dayofyear from _partitiontime)/366.5*4) quarter,
    timestamp_add( timestamp("2017-01-01"),
       interval
        cast( floor(timestamp_diff(_partitiontime,  
                                    timestamp("2017-01-01"),
                                     day)/24)*24 as int64) day) as date_24,
    *,
      format("lon:%+07.2f_lat:%+07.2f", 
    detect_lon, 
    detect_lat) as gridcode, -- for matching
    floor(detect_lat * 200) lat_index,
    floor(detect_lon * 200) lon_index,
    st_geogpoint(detect_lon, detect_lat) pos,
    extract(year from _partitiontime) year,
    extract(month from _partitiontime) month
  FROM 
    `{project_id}.proj_sentinel1_v20210924.detect_scene_match` 
  join
    (select 
      detect_id, avg(presence) presence, 
      avg(length_m) length_m 
    from 
      `{project_id}.proj_sentinel1_v20210924.detect_scene_pred_*`
    group by detect_id
    )
  using(detect_id)
    -- where _partitiontime between "2017-01-01" and "2021-12-31"
 ),

regions as (
SELECT 
  regions.eez as eez_array, 
  if(array_length(regions.eez)>0, regions.eez[ordinal(1)], null) MRGID,
  gridcode 
FROM 
  `{project_id}.pipe_static.spatial_measures_20201105` 
),

eez_table as (
select 
  detect_id,
  eez_array,
  ISO_TER1 as eez_iso3
from 
  raw_detections
join
  regions
using(gridcode)
left join
  proj_global_sar.eez_info_v11
using(MRGID)
),

-- scene quality table
scene_quality as (
  select 
    scene_id,
    count(*) scene_detections,
    sum(if(presence < .7 or presence is null,1,0)) bad_detections
  from 
    raw_detections
  group by 
    scene_id
 ),

repeated_objects_center180_100m as (
  select 
  a.detect_id,
  count(*) repeats_100m_180days_center
  from 
    raw_detections a
  cross join 
    raw_detections b
  where st_distance(a.pos, b.pos) < 100
    and a.detect_id != b.detect_id
    and abs(timestamp_diff(a.date, b.date, day)) <= 90 
  group by a.detect_id
 -- having repeats >=3
),

repeated_objects_forward180_100m as (
  select 
  a.detect_id,
  count(*) repeats_100m_180days_forward
  from 
    raw_detections a
  cross join 
    raw_detections b
  where st_distance(a.pos, b.pos) < 100
    and a.detect_id != b.detect_id
    and timestamp_diff(b.date, a.date, day) between 0 and 180 
  group by a.detect_id
),

-- Get number of repeats... with 6 month window looking backwards,
-- centered, and looking forward. 

repeated_objects_back180_100m as (
  select 
    a.detect_id,
    count(*) repeats_100m_180days_back
  from 
    raw_detections a
  cross join 
    raw_detections b
  where 
    st_distance(a.pos, b.pos) < 100
    and a.detect_id != b.detect_id
    and timestamp_diff(a.date, b.date, day) between 0 and 180 
  group by a.detect_id
),

repated_detections as 
(
  select distinct detect_id 
    from (   
      select detect_id from 
      repeated_objects_center180_100m
    union all
    select detect_id from 
      repeated_objects_back180_100m
    union all
    select detect_id from 
      repeated_objects_forward180_100m
    )
),

final_table as 
(select
  date,
  year,
  date_quarter,
  date_24,
  detect_id,
  scene_id,
  detect_timestamp,
  detect_lat,
  detect_lon,
  score,
  ssvid,
  confidence,
  score_ave,
  ssvid_ave,
  confidence_ave,
  score_mult,
  ssvid_mult,
  confidence_mult,
  score_mult_recall,
  ssvid_mult_recall,
  confidence_mult_recall,
  score_mult_recall_length,
  ssvid_mult_recall_length,
  confidence_mult_recall_length,
  score_mult_length,
  ssvid_mult_length,
  confidence_mult_length,
  score_ave_length,
  ssvid_ave_length,
  confidence_ave_length,
  presence,
  length_m,
  scene_detections,
  bad_detections,
  1 - bad_detections/scene_detections scene_quality,
  ifnull(repeats_100m_180days_back, 0) repeats_100m_180days_back,
  ifnull(repeats_100m_180days_center, 0) repeats_100m_180days_center,
  ifnull(repeats_100m_180days_forward, 0) repeats_100m_180days_forward,
  overpasses_2017_2021,
  yearly_overpasses.overpasses as overpasses_year,
  quarterly_overpasses.overpasses as overpasses_quarter,
  every24day_overpasses.overpasses as overpasses_24day,
  years_with_overpasses,
  quarters_with_overpass,
  periods24_with_overpass,
  eez_iso3,
  eez_array,
  ifnull(in_road_doppler, False) in_road_doppler,
  ifnull(in_road_doppler2, False) in_road_doppler2,
from
  raw_detections a
  left join
scene_quality
  using(scene_id)
left join
  repeated_objects_center180_100m 
using(detect_id)
left join
  repeated_objects_forward180_100m
using(detect_id)
left join
  repeated_objects_back180_100m
using(detect_id)
left join
  yearly_overpasses
using(lat_index, lon_index, year)
left join
  overpasses_2017_2021_table
using(lat_index, lon_index)
left join
  quarterly_overpasses
using(lat_index, lon_index, quarter)
left join
  quarters_with_overpass_table
using(lat_index, lon_index)
left join
  every24day_overpasses
using(date_24, lat_index, lon_index)
left join 
  periods24_with_overpass_table
using(lat_index, lon_index)
left join
  years_with_overpasses_table
using(lat_index, lon_index)
left join
  eez_table
using(detect_id)
left join
  score_ave_table
using(detect_id)
left join
  score_mult_table
using(detect_id)
left join
  score_mult_recall_table
using(detect_id)
left join
  score_mult_recall_length_table
using(detect_id)
left join
  score_ave_length_table
using(detect_id)
left join
  score_mult_length_table
using(detect_id)
left join
  road_doppler_table
using(detect_id)
)

select * from final_table"""

table_id = f"{project_id}.proj_global_sar.detections_w_overpasses_v20230420"
query_to_table(q, table_id)

description = (
    """detections with overpasses, uses new matches, updates EEZs, and adds in doppler regions from roads. 

The query to make it: 
"""
    + q
)

update_table_description(table_id, description)
# -

# # Version 20230803
#
# include potential azimuth ambiguities and fixed infrastructure

# +
q = f"""with detections as (
  select 
    * except(row) 
  from 
   (select *, row_number() over (partition by detect_id order by rand()) as row
   from proj_global_sar.detections_w_overpasses_v20230420 )
   where row = 1 -- oops, previous version had duplicates
),

infrastructure as 
(
  select 
    st_geogpoint(lon, lat) as pos 
  from (
      SELECT distinct cluster_number, st_y(clust_centr_final) as lat, 
      st_x(clust_centr_final) as lon FROM proj_sentinel1_v20210924.detect_comp_cluster_locations_20230623
    )
),


bad_detects as (
  select 
    distinct 
    detect_id, 
  from 
    detections a
  cross join
    infrastructure b
  where
    st_distance(st_geogpoint(detect_lon, detect_lat), b.pos) < 200
),

potential_ambiguities as (
  select 
    distinct detect_id from proj_global_sar.potential_S1_amgibuities_2017_2021
  where 
  line_dist < 200
  and ambiguity_length < source_length
)

select 
  * ,
  bad_detects.detect_id is not null as close_to_infra,
  potential_ambiguities.detect_id is not null as potential_ambiguity
from 
  detections 
left join
  bad_detects
using(detect_id)
left join
  potential_ambiguities
using(detect_id)"""



table_id = f"{project_id}.proj_global_sar.detections_w_overpasses_v20230803"
query_to_table(q, table_id)

description = (
    """detections with overpasses, using the same table as detections_w_overpasses_v20230420 except with 
    the field potential_ambiguity, which highlights likely azimuthal ambiguities, and the field close_to_infra,
    which flags detections that are 200m within fixed infrastructure

The query to make it: 
"""
    + q
)

update_table_description(table_id, description)
# -



# # Now, an exciting table... this one is expensive 
#
# $65
#
# build a rolling average in sql

q = f'''with 
    dates_table as (
      select * from unnest(GENERATE_TIMESTAMP_ARRAY(timestamp("2017-01-01"),
                                      timestamp("2021-12-31"),
                                      INTERVAL 1 day)) as rolling_date),


every24day_overpasses as (
  select date as date_24, lat_index, lon_index, overpasses from 
  proj_global_sar.overpasses_200_every24days_v20220805
),

periods24_with_overpass_table as 
(
  select 
    count(*) periods24_with_overpass, 
    lat_index, lon_index 
  from 
    every24day_overpasses
  group by lat_index, lon_index
),


rolling_24 as 

(select 
date as rolling_date,
overpasses,
lat_index,
lon_index 
from
`proj_global_sar.overpasses_200_24dayrolling_v20220802_*`),

-- get lengths and presence for each detection
raw_detections as (
  select 
    _partitiontime date,
    extract(year from _partitiontime)*10 + floor(extract(dayofyear from _partitiontime)/366.5*4) quarter,
    timestamp_add( timestamp("2017-01-01"),
       interval
        cast( floor(timestamp_diff(_partitiontime,  
                                    timestamp("2017-01-01"),
                                     day)/24)*24 as int64) day) as date_24,
    *,
    floor(detect_lat * 200) lat_index,
    floor(detect_lon * 200) lon_index,
    st_geogpoint(detect_lon, detect_lat) pos,
    extract(year from _partitiontime) year,
    extract(month from _partitiontime) month
  FROM 
    `{project_id}.proj_sentinel1_v20210924.detect_scene_match` 
  join
    (select 
      detect_id, avg(presence) presence, 
      avg(length_m) length_m 
    from 
      `{project_id}.proj_sentinel1_v20210924.detect_scene_pred_*`
    group by detect_id
    )
  using(detect_id)
    -- where _partitiontime between "2017-01-01" and "2021-12-31"
 ),

 expand_raw_detections as (

select
 detect_id,
 date,
 rolling_date,
 detect_lat,
 detect_lon,
 lat_index,
 lon_index,
from 
  raw_detections
cross join
  dates_table
where 
  rolling_date between 
              timestamp_sub(date, interval 12 day) 
              and timestamp_add(date, interval 11 day))

select 
  rolling_date,
  detect_id,
  overpasses
from 
  expand_raw_detections
join 
  rolling_24
using(lat_index, lon_index, rolling_date)
'''

# +
table_id = f'{project_id}.proj_global_sar.detections_w_overpasses_rolling24_v20220805'
# query_to_table(q, table_id)

description = '''This table contains just detect_id, a date that is within 12 days back or forward of that detect_id, and the \
number of overpasses in that time period. Note that each detection is here 24 times (about) -- once for every overpass it is in.

The query to make it: 
''' + q

update_table_description(table_id, 
                        description)
# -


