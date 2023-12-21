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

# %matplotlib inline
import numpy as np
import pandas as pd
from IPython.core.display import display, HTML
import subprocess
from subprocess import Popen
from tabulate import tabulate
from datetime import datetime, timedelta
from pathlib import Path
import os

import matplotlib.pyplot as plt
import matplotlib.colors as mpcolors
import pyseas.maps as psm
import pyseas.contrib as psc
import pyseas.cm
import matplotlib as mpl
from concurrent.futures import ThreadPoolExecutor
import time

# # Create Grid of Cells to Look In
#
# We are going to build a table of grid cells that have the following characteristics:
#  - In an EEZ with a high use of AIS by fishing vessels, such that most boats in the region should be included. 
#  - More than 20km from shore.
#  - In AIS data, in a 0.1 by 0.1 degree square, there was under 10 hours of activity in AIS data in 2018. This is a proxy for areas with very low density fof vessels in the AIS data. 
#  - Class A reception is over 60 positions per day.
#
#
# These are areas were we expect to see very few vessels. The number of detections in this area can serve as the upper bound for the number of false positives per unit area.

dataset='proj_global_sar'
low_density_table = "cells_10kmfromshore_under10hours_200thdegree_eez_v20220612"
# out = !bq mk --time_partitioning_type=DAY {dataset}.{low_density_table}
print(out)


def make_raster_far_from_shore_date_range(
    thedate,
    low_density_table=low_density_table,
    dataset=dataset
):
    q = f'''   #standardSQL 
    WITH
      --
      reception_quality AS (
      SELECT
        lat_bin lat_bin_1degree,
        lon_bin AS lon_bin_1degree,
        positions_per_day
      FROM
        `proj_ais_gaps_catena.sat_reception_one_degree_v20200806`
      WHERE
        year = 2019
        AND class = "A"
        AND month = 1),
      --
      --
      eez_info_table AS (
      SELECT
        gridcode,
        territory1_iso3 AS iso3
      FROM (
        SELECT
          eez AS eez,
          gridcode
        FROM
          `pipe_static.regions`
        CROSS JOIN
          UNNEST(regions.eez) AS eez ) a
      JOIN
        `gfw_research.eez_info` b
      ON
        a.eez = CAST(eez_id AS string)),
      --
      --
      overpasses AS (
      SELECT
        lat_index,
        lon_index,
        scene_id
      FROM
        `proj_sentinel1_v20210924.detect_foot_raster_200`
      WHERE
        _partitiontime = TIMESTAMP("{thedate}")
        AND scene_id != "" ),
      --
      --
      joined_distance_from_shore AS (
      SELECT
        lat_index,
        lon_index,
        scene_id,
        distance_from_shore_m
      FROM
        overpasses a
      JOIN
        `pipe_static.distance_from_shore` b
      ON
        CAST((a.lat_index/2) AS int64) = CAST((b.lat*100)AS int64)
        AND CAST((a.lon_index/2)AS int64)=CAST((b.lon*100)AS int64) ),
      --
      --
      joined_with_density AS (
      SELECT
        a.lat_index lat_index,
        a.lon_index lon_index,
        ifnull(f0_,
          0) AS vessel_hours,
        distance_from_shore_m,
        scene_id
      FROM
        joined_distance_from_shore a
      LEFT JOIN
       gfw_research_precursors.vessel_density_10thdegree_2018  b
      ON
        CAST((a.lat_index/20) AS int64) = CAST((b.lat_bin) AS int64)
        AND CAST((a.lon_index/20)AS int64)=CAST((b.lon_bin) AS int64) )
      --
      --
    SELECT
      lat_index,
      lon_index,
      scene_id,
      distance_from_shore_m,
      ifnull(iso3,
        "high seas") iso3,
    FROM
      joined_with_density
    LEFT JOIN
      eez_info_table
    ON
      FORMAT("lon:%+07.2f_lat:%+07.2f", ROUND(lon_index/200/0.01+1/400)*0.01, ROUND(lat_index/200/0.01+1/400)*0.01) = gridcode
    JOIN
      reception_quality
    ON
      FLOOR(lat_index/200) = lat_bin_1degree
      AND FLOOR(lon_index/200) = lon_bin_1degree
    WHERE
      distance_from_shore_m > 20*1000 -- more than 20km from shore
      AND vessel_hours < 10 -- less than 10 vessel hours in 2018
      AND positions_per_day > 60 -- more than 60 class A positions per day
      AND ((iso3 IS NULL
          AND distance_from_shore_m > 1852*180 ) -- in the high seas
        OR iso3 IN ( -- in an EEZ with high use of AIS by fishing vessels
          'ARG',
          'CAN',
          'DNK',
          'FRA',
          'DEU',
          'GRC',
          'ISL',
          'IRL',
          'ITA',
          'JPN',
          'NLD',
          'NOR',
          'POL',
          'SYC',
          'KOR',
          'ESP',
          'TWN',
          'GBR',
          'USA',
          'VUT',
          'PER',
          'CHL',
          'ZAF',
          'ECU',
          'NZL')) 
      AND
      -- select regions of the world to include...
        ((lat_index/200 BETWEEN -55
          AND 45) -- avoid ice
        OR (lon_index/200 BETWEEN -143.12
          AND -121.17
          AND lat_index/200 BETWEEN 43.82
          AND 55.11 ) -- western Canada and NW US
        OR (lon_index/200 BETWEEN -84.1
          AND -54.4
          AND lat_index/200 BETWEEN -57.4
          AND -45.2 ) -- south america
        )
      -- select regions to exclude
      AND NOT (lon_index/200 BETWEEN 127.11
        AND 165.92
        AND lon_index/200 BETWEEN 35.94
        AND 63.01)  -- sea of Japan and Okhotsk'''

    thedate_ = thedate.replace("-", "")
    cmd = f"""
    bq query --replace \
       --destination_table={dataset}.{low_density_table}${thedate_} \
       --allow_large_results \
       --use_legacy_sql=false  \
       --max_rows=0
    """.strip().split()
    
    subprocess.run(cmd, input=bytes(q, "utf-8"))




the_dates = np.arange(datetime(2017,4,11), datetime(2022,1,1), timedelta(days=1)).astype(datetime)


# +
start_time = time.time()

with ThreadPoolExecutor(max_workers=16) as e:
    for d in the_dates:
        e.submit(make_raster_far_from_shore_date_range, d.strftime("%Y-%m-%d"))

print("Per query, this took", (time.time() - start_time)/len(the_dates), "to run")
# -

# # Map Grid Cells

# +
scale = 10

q = f'''
select 
  floor(lat_index/200*{scale}) lat_index,
  floor(lon_index/200*{scale}) lon_index,
  count(*)/(200/{scale}*200/{scale}) overpasses 
from 
  {dataset}.{low_density_table}
group by lat_index, lon_index
'''
df = pd.read_gbq(q)
# -

overpass_grid = psm.rasters.df2raster(df,
                'lon_index', 'lat_index',
               'overpasses', xyscale=scale, 
                per_km2=False, origin = 'lower')

df.overpasses.max()

# +

norm = mpcolors.LogNorm(vmin=1, vmax=df.overpasses.max()/2)

with psm.context(psm.styles.dark):
    fig = plt.figure(figsize=(15, 15))
    ax, im = psm.plot_raster(
        overpass_grid,
        cmap= "presence",
        norm=norm,
        origin="lower",
    )
    fig.colorbar(
    im,
    ax=ax,orientation="horizontal",
        fraction=0.02,
        aspect=40,
        pad=0.04)

plt.title("Regions assessed for false positives")
# -





# # Now count detections in this region
#
# - that have a presence < .7
# - that do not match to AIS
# - that are not in "bad scenes" -- scenes with lots of bad detections, which we are eliminating

# +
q = f'''#standardSQL

---------------------------------------------------------------
-- User defined JS helper functions
---------------------------------------------------------------

  CREATE TEMP FUNCTION startdate() AS (DATE('2017-01-01'));
  CREATE TEMP FUNCTION enddate() AS (DATE('2021-12-31'));

  # Define some utility functions to make things more readable
  CREATE TEMP FUNCTION YYYYMMDD(d DATE) AS (
    # Format a date as YYYYMMDD
    # e.g. DATE('2018-01-01') => '20180101'
    FORMAT_DATE('%Y%m%d',
      d) );


with 

detection_table as (
  select 
  extract(date from _partitiontime) date,
  substring(scene_id, 1,3) as sat,
  scene_id, 
  detect_lat,
  detect_lon,
  score, presence, length_m 
from 
  `proj_sentinel1_v20210924.detect_scene_match`
join 
  `proj_sentinel1_v20210924.detect_scene_pred_*` 
using(detect_id)
where  
  DATE(_PARTITIONTIME) between startdate() and enddate()
  AND _table_suffix between YYYYMMDD(startdate()) and YYYYMMDD(enddate())

),

bad_scenes as (
  select scene_id from 
  (
    select 
      scene_id, 
      count(*) scene_detections,
      sum(if(presence>.7,1,0))/count(*) scene_quality
    from 
      detection_table
    group by 
      scene_id
  )
  where 
    -- get rid of all scenes that have more than five detections
    -- where over half of the detections have a presence under .7
    -- This should eliminate some bad scenes
    (scene_detections >=5 and scene_quality <= .5)
),


grid_table as (
  select 
    *, extract(date from _partitiontime) as date
  from 
    {dataset}.{low_density_table}
  where 
    date(_partitiontime) between startdate() and enddate()
    and scene_id not in (select scene_id from bad_scenes)
),


area_table as
(
select
  date,
  substring(scene_id, 1,3) as sat,
  SUM(111*111/(200*200)*COS(lat_index/200*3.14/180) ) area_km2,
  -- area of each cell in km2
  COUNT(*) num
FROM
  grid_table
GROUP by sat, date
)



select 
  a.date date,
  a.sat sat,
  count(*) detections,
  area_km2,
  count(*)/area_km2 as detections_per_km2
from 
  detection_table a
JOIN
  grid_table b
ON
  FLOOR(detect_lat*200) = lat_index
  AND FLOOR(detect_lon*200) = lon_index
  and a.scene_id = b.scene_id
join
  area_table c
on(a.sat=c.sat and a.date=c.date)
  where presence > .7
  and score < 1e-2
  and a.scene_id not in (select scene_id from bad_scenes)
  group by date,sat, area_km2 order by date, sat
'''

df = pd.read_gbq(q)
# -

df.head()







plt.figure(figsize=(10,3))
d = df[df.sat=="S1A"]
plt.plot(d.date,d.detections_per_km2)
d = df[df.sat=="S1B"]
plt.plot(d.date,d.detections_per_km2)

# overall density for S1A
d = df[df.sat=="S1A"]
d.detections.sum()/d.area_km2.sum()

# overall density for S1B
d = df[df.sat=="S1B"]
d.detections.sum()/d.area_km2.sum()

#overall
df.detections.sum()/df.area_km2.sum()

# ### False Positive Denisty
# Estimated to be 5.4 * 10^-5 vessels per km^2
#

# total area in million km2
df.area_km2.sum()/1e6

# total detections
df.detections.sum()

# before 2018-3-12
d = df[df.date<datetime(2018,3,12)]
d.detections.sum()/d.area_km2.sum()

# between 2018-3-12 and 2020-01-12
d = df[(df.date>datetime(2018,3,12))&(df.date<datetime(2020,1,12))]
d.detections.sum()/d.area_km2.sum()

# after 2020-01-12
d = df[(df.date>datetime(2020,1,12))]
d.detections.sum()/d.area_km2.sum()





# # Now compare this to our total number of detections and the potential false positive rate

# +
# create a number of bounding boxes to elinate areas with ice

boxes = []
boxes.append([-120.0,50.5,-46.8,80.5]) # huson bay, canada, etc.
boxes.append([-120.0,50.5,-46.8,80.5]) # huson bay, canada, etc.
boxes.append([39.5,65.0,-46.8,90]) # arctic except n. atlantic
boxes.append([15.95,59.02,36.23,66.57]) # North Baltic sea
boxes.append([-173.7,62.0,-158.4,66.8]) # north beiring sea
boxes.append([130.5,50.6,-174.2,67.8]) #sea of okhotsk
boxes.append([3.5,78.1,31.9,85.0]) #north of Salvbard
boxes.append([-179.8,57.4,-156.5,62.2]) #beiring sea, more southern, because it didn't work
boxes.append([-44.82,-57.93,-29.05,-50.61]) ## south georgia island
boxes.append([31.4,61.4,60.3,73.1])## far northeast russia -- a small area
boxes.append([-27.61,68,-19.47,68.62]) # tiny piece of ice near ne iceland that annoyed me

eliminated_locations = '''and not 
  (
'''

for b in boxes:
    min_lon, min_lat, max_lon, max_lat = b
    if min_lon > max_lon:
        bounding_string = \
'''   ( 
      (lon_index/10 > {min_lon} or lon_index/10 < {max_lon} ) and 
      lat_index/10> {min_lat} and lat_index/10 < {max_lat} ) 
  or'''
    else:
        bounding_string = \
'''   ( lon_index/10 > {min_lon} and lon_index/10 < {max_lon} 
      and lat_index/10 > {min_lat} and lat_index/10 < {max_lat} ) or 
'''
    eliminated_locations+=bounding_string.format(min_lon=min_lon,
                                                         max_lon=max_lon, 
                                                         max_lat=max_lat, 
                                                         min_lat=min_lat)
eliminated_locations = eliminated_locations[:-4] + ")\n"

# print(eliminated_locations)

# +
q = f'''

CREATE TEMP FUNCTION startdate() AS (DATE('2017-01-01'));
CREATE TEMP FUNCTION enddate() AS (DATE('2021-12-31'));

# Define some utility functions to make things more readable
CREATE TEMP FUNCTION YYYYMMDD(d DATE) AS (
  # Format a date as YYYYMMDD
  # e.g. DATE('2018-01-01') => '20180101'
  FORMAT_DATE('%Y%m%d',
    d) );


with 
detection_table as (
  select 
  extract(date from _partitiontime) date,
  substring(scene_id, 1,3) as sat,
  scene_id, 
  detect_lat,
  detect_lon,
  score, presence, length_m 
from 
  `proj_sentinel1_v20210924.detect_scene_match`
join 
  `proj_sentinel1_v20210924.detect_scene_pred_*` 
using(detect_id)
where  
  DATE(_PARTITIONTIME) between startdate() and enddate()
  AND _table_suffix between YYYYMMDD(startdate()) and YYYYMMDD(enddate())

),

bad_scenes as (
  select scene_id from 
  (
    select 
      scene_id, 
      count(*) scene_detections,
      sum(if(presence>.7,1,0))/count(*) scene_quality
    from 
      detection_table
    group by 
      scene_id
  )
  where 
    -- get rid of all scenes that have more than five detections
    -- where over half of the detections have a presence under .7
    -- This should eliminate some bad scenes
    (scene_detections >=5 and scene_quality <= .5)
)

SELECT 
  extract(date from _partitiontime) date,
  sum(overpasses * pow(cos(lat_index/10*3.1415/180),2)*111*111/10/10)  area_km2,
FROM 
  `proj_sentinel1_v20210924.detect_foot_raster_10` 
WHERE 
  DATE(_PARTITIONTIME) between startdate() and enddate()
  and scene_id not in (select scene_id from bad_scenes)
  {eliminated_locations}
group by date order by date'''

df_tot_area = pd.read_gbq(q)
# -

plt.figure(figsize=(10,3))
plt.plot(df_tot_area.date, df_tot_area.area_km2/1e6)
plt.ylim(0,11)

# +
# total number of detections that could be bad

total_bad_detects = df_tot_area.area_km2.sum()*d.detections.sum()/d.area_km2.sum()
total_bad_detects
# -

# # How many detections total?

# +
eliminated_locations = '''and not 
  (
'''

for b in boxes:
    min_lon, min_lat, max_lon, max_lat = b
    if min_lon > max_lon:
        bounding_string = \
'''   ( 
      (detect_lon > {min_lon} or detect_lon < {max_lon} ) and 
      detect_lat> {min_lat} and detect_lat < {max_lat} ) 
  or'''
    else:
        bounding_string = \
'''   ( detect_lon > {min_lon} and detect_lon < {max_lon} and detect_lat> {min_lat} and detect_lat < {max_lat} ) or 
'''
    eliminated_locations+=bounding_string.format(min_lon=min_lon,
                                                         max_lon=max_lon, 
                                                         max_lat=max_lat, 
                                                         min_lat=min_lat)
eliminated_locations = eliminated_locations[:-4] + ")\n"

# print(eliminated_locations)

# +
q = f'''

with 

predictions_table as
(
select detect_id, fishing as fishing_score from 
`proj_global_sar.rf_predictions_v20220515_*`
where _table_suffix between "20170101" and "20211231"
),

vessel_info as (
select ssvid, on_fishing_list_best
from `gfw_research.vi_ssvid_v20220101`
),

eez_table as

        (select detect_id, 
           array_agg(ifnull(MRGID,0)) as MRGID, 
           array_agg(ifnull(ISO_TER1,"")) as ISO_TER1, 
           array_agg(ifnull(TERRITORY1,"")) AS TERRITORY1
        from proj_global_sar.detections_w_overpasses_v20220509
        CROSS JOIN
      (select wkt, MRGID, ISO_TER1, TERRITORY1 from `minderoo.marine_regions_v11`)
      WHERE
      ST_CONTAINS(SAFE.ST_GEOGFROMTEXT(wkt),ST_GEOGPOINT(detect_lon,detect_lat) )
      group by detect_id

      ),
      
final_table as (
select 
  count(*) detections,
  sum(if( repeats_100m_180days_forward < 3 and 
  repeats_100m_180days_back < 3 and
  repeats_100m_180days_center < 3 , 1,0)) non_stationary
from 
  proj_global_sar.detections_w_overpasses_v20220509 a
left join
  vessel_info
using(ssvid)
left join
  predictions_table
using(detect_id)
left join
 eez_table
using(detect_id)
where
  (scene_detections <=5 or scene_quality > .5)
  and extract(date from detect_timestamp) between "2017-01-01" and "2021-12-31"
  and presence > .7
  {eliminated_locations}

)

select * from final_table


'''

df_detects = pd.read_gbq(q)
# -

df_detects

total_detections = df_detects.detections.values[0]
total_non_repeated = df_detects.non_stationary.values[0]

total_bad_detects/total_detections

total_bad_detects/total_non_repeated

# ### False Positives
#
# According this estimate, about 2-3% of detections might be false positives if all of the detections in these regions are false positives and the false positive density is the same in rest of the world. 


