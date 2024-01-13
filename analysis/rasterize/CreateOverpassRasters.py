# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Create Overpass Rasters
#
# This notebook takes the rasterized version of the Sentinel-1 footprints an combines them to create tables that have the number of overpasses a given cell (1/200th of a degree by 1/200th of a degree) had over a given time period. These overpass rasters were important for estimating vessel density by accounting for the number of times each part of the ocean was imaged. 
#
#  - all time (2017-2022)
#  - corresponding to the infrastructre time periods (six month moving time windows)
#  - yearly
#  - by quarter
#  - by 24 day period
#  - then, ambitiously, by 24 day period *with a moving window* -- this is important to create the smooth curve for figure 4
import sys
sys.path.append('../utils')
from bigquery_helper_functions import (
    update_table_description,
    query_to_table
)

# ### by year
#

# +
# by year, filtered

q = '''
with raw_detections as (

select 
  _partitiontime date,
  *,
  st_geogpoint(detect_lon, detect_lat) pos,
  extract(year from _partitiontime) year,
  extract(month from _partitiontime) month
  FROM `project-id.proj_sentinel1_v20210924.detect_scene_match` 
  join
  ( select 
      detect_id, avg(presence) presence, 
      avg(length_m) length_m 
    from 
      `project-id.proj_sentinel1_v20210924.detect_scene_pred_*`
    group by 
      detect_id)
  using(detect_id)
 ),

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

scenes_to_ignore as (
  select 
    scene_id 
  from 
    scene_quality 
  where 
    bad_detections/scene_detections > .5 and scene_detections > 5 
)

SELECT 
  extract(year from _partitiontime) year,
  lat_index,
  lon_index,
  count(*) overpasses
 FROM 
   `project-id.proj_sentinel1_v20210924.detect_foot_raster_200` 
 WHERE 
   scene_id not in (select * from scenes_to_ignore)
 group by 
   lat_index, lon_index, year '''

# expensive!!!
# query_to_table(q, 'project-id.proj_global_sar.overpasses_200_by_year_filtered_v20220508')

description = '''A table with the number of overpasses by year, with scenes eliminated that have too many bad detections
lat_index and lon_index are floor(lat*200) and floor(lon*200) respectively

query to generate: 
''' + q

update_table_description('project-id.proj_global_sar.overpasses_200_by_year_filtered_v20220508', 
                        description)

# +
## by year, unfiltered


q = '''
SELECT 
  extract(year from _partitiontime) year,
  lat_index,
  lon_index,
  count(*) overpasses
 FROM 
   `project-id.proj_sentinel1_v20210924.detect_foot_raster_200` 
 group by 
   lat_index, lon_index, year '''

# expensive!!!
query_to_table(q, 'project-id.proj_global_sar.overpasses_200_by_year_v20221031')

description = '''A table with the number of overpasses by year, 
lat_index and lon_index are floor(lat*200) and floor(lon*200) respectively

query to generate: 
''' + q

update_table_description('project-id.proj_global_sar.overpasses_200_by_year_v20221031', 
                        description)
# -

# ### By Quarter

# +
q = '''select
  extract(year from _partitiontime)*10 + floor(extract(dayofyear from _partitiontime)/366.5*4) quarter,
  min(_partitiontime) date,
  lat_index,
  lon_index,
  count(*) overpasses
from 
  `project-id.proj_sentinel1_v20210924.detect_foot_raster_200`
group by 
  lat_index,lon_index, quarter'''

table_id = 'project-id.proj_global_sar.overpasses_200_byquarter_v20220805'

# -

# expensive!!!
# query_to_table(q, table_id)


# +
description = '''Overpasses by quarter. 

The query to make it: 
''' + q

update_table_description(table_id, 
                        description)
# -



# ### By Quarter Smoothed

q = '''WITH
dates_table as (
  select * from unnest(GENERATE_TIMESTAMP_ARRAY(timestamp("2017-01-01"),
                                  timestamp("2021-12-31"),
                                  INTERVAL 24 day)) as date)

select 
  date,
  lat_index,
  lon_index,
  count(*) overpasses
from 
  dates_table
cross join
  `project-id.proj_sentinel1_v20210924.detect_foot_raster_200`
where 
  _partitiontime between
  timestamp_sub(date, interval 48 day) and timestamp_add(date, interval 48 day)
group by 
  date, lat_index, lon_index
'''

table_id = 'project-id.proj_global_sar.overpasses_200_quartersmoothed_v20220617'


q = '''
extract(year from _partitiontime)*10 + floor(extract(dayofyear from _partitiontime)/366.5*4) quarter,
min(_partitiontime) date,
lat_index,
lon_index,
count(*) overpasses
from 
  `project-id.proj_sentinel1_v20210924.detect_foot_raster_200`
group by lat_index,lon_index, quarter
'''
# query_to_table(q, "project-id.proj_global_sar.overpasses_200_quartersmoothed_v20220617")



# +
description = '''Every 24 days, awindow of 48 days back and 48 days forward, \
which is roughly 1/4th of a year. So it produces a roughly a moving 1/4 year window. 

The query to make it: 
''' + q

update_table_description(table_id, 
                        description)
# -

# ### Every 24 days
#
# The repeat time of Sentinel-1 is 12 days, so 24 is twice this

q = '''WITH
dates_table as (
  select * from unnest(GENERATE_TIMESTAMP_ARRAY(timestamp("2017-01-01"),
                                  timestamp("2021-12-31"),
                                  INTERVAL 24 day)) as date)

select 
  date,
  lat_index,
  lon_index,
  count(*) overpasses
from 
  dates_table
cross join
  `project-id.proj_sentinel1_v20210924.detect_foot_raster_200`
where 
  _partitiontime between date and timestamp_add(date, interval 23 day)
group by 
  date, lat_index, lon_index
'''

table_id = 'project-id.proj_global_sar.overpasses_200_every24days_v20220805'


# query_to_table(q, table_id)


# +
description = '''overpasses for each 24 day period, starting at a given 24 hour time period.  

the date can also be calculated as:


timestamp_add( timestamp("2017-01-01"),
       interval
        cast(
        floor(timestamp_diff(timestamp_to_compare,  
                timestamp("2017-01-01"), day)/24)*24 as int64) day)

The query to make it: 
''' + q

update_table_description(table_id, 
                        description)
# -

# ### Every day with 24 day moving average
#
#

for year in [2017,2018,2019,2020,2021]:
    q = f'''WITH
    dates_table as (
      select * from unnest(GENERATE_TIMESTAMP_ARRAY(timestamp("{year}-01-01"),
                                      timestamp("{year}-12-31"),
                                      INTERVAL 1 day)) as date)

    select 
      date,
      lat_index,
      lon_index,
      count(*) overpasses
    from 
      dates_table
    cross join
      `project-id.proj_sentinel1_v20210924.detect_foot_raster_200`
    where 
      _partitiontime between timestamp_sub("{year}-01-01", interval 12 day) 
                    and timestamp_add("{year}-12-31", interval 11 day)
     and _partitiontime between timestamp_sub(date, interval 12 day) and timestamp_add(date, interval 11 day)
    group by 
      date, lat_index, lon_index

    '''
    table_id = f"project-id.proj_global_sar.overpasses_200_24dayrolling_v20220802_{year}0101"
#     query_to_table(q, table_id")

    description = '''For each day, the number of overpasses in a period 11 days back to 12 days forward (24 total days).  

    The query to make it: 
    ''' + q

    update_table_description(table_id, 
                            description)


