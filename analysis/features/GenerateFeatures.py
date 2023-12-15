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

# # Generate Features 
#
# This notebook produces features for the random forest and the neural net classifier that determines if a vessel is a fishing vessel. Note that for the paper we did not use the random forest classifer.
#

# +
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mpcolors
import pandas as pd
import pyseas.maps as psm
import pyseas.contrib as psc
import pyseas.cm
# %matplotlib inline
from datetime import datetime, timedelta
import matplotlib as mpl
import cartopy
import cartopy.crs as ccrs

from google.cloud import bigquery
from google.cloud.exceptions import NotFound



# Construct a BigQuery client object.
client = bigquery.Client()


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
# -



# ## Vessel Density
#
# Notes on vessel denisity:
#  - we include only vessels where on_fishing_list_best and on_fishing_list_nn agree -- so we don't include "ambigous vessels"
#  - the output values are the sum of hours at a 100th of a degree across 2017 to 2021

q = '''
with good_vessels as (

select 
  ssvid,
  on_fishing_list_best,
  best.best_vessel_class as vessel_class,
  best.best_length_m as length_m
from
  `project-id.gfw_research.vi_ssvid_byyear_v20220101`
where 
  on_fishing_list_nn = on_fishing_list_best
  and best.best_length_m is not null
and not activity.offsetting
and activity.active_hours > 200
and activity.overlap_hours < 24
),

good_segs as (
  select seg_id from `project-id.gfw_research.pipe_v20201001_segs`
  where good_seg and not overlapping_and_short
  and positions > 100
),

gridded_activity as (
select 
  count(*)*5/60 hours,
  vessel_class,
  on_fishing_list_best as is_fishing_vessels,
  vessel_class in ("cargo","tanker","cargo_or_tanker") is_cargo_tanker,
  length_m < 50 as under_50m,
  floor(lon*100) lon_index,
  floor(lat*100) lat_index
from
  `project-id.gfw_research.pipe_v20201001_5min`
join
  good_vessels
using(ssvid)
join 
  good_segs
using(seg_id)
where 
  _partitiontime between '2017-01-01' and '2021-12-31'
group by 
  vessel_class, lat_index, lon_index, is_fishing_vessels, is_cargo_tanker, under_50m
)


select * from gridded_activity'''

# +
# query_to_table(q, 'project-id.proj_global_sar.ais_vessel_density_2017_2021_100th_degree')
# -









# +
## Make cargo and tanker and non-fishing at a few different resolutions
##

q = '''with 
grid_100 as 
(
select 
lat_index,
lon_index,
sum(if(not is_fishing_vessels,hours/(111*111/100/100*cos(3.14/180*lat_index/100)),0)) non_fishing_100,
sum(if(is_fishing_vessels,hours/(111*111/100/100*cos(3.14/180*lat_index/100)),0)) fishing_100,
sum(if(not is_fishing_vessels and under_50m,hours/(111*111/100/100*cos(3.14/180*lat_index/100)),0)) non_fishing_under50m_100, --under_50m
sum(if(is_cargo_tanker,hours/(111*111/100/100*cos(3.14/180*lat_index/100)),0)) cargo_tanker_100 
 from `project-id.proj_global_sar.ais_vessel_density_2017_2021_100th_degree`
 group by lat_index, lon_index
),

grid_2 as (
select 
a.lat_index,
a.lon_index,
avg(b.non_fishing_100) as non_fishing_2km,
avg(b.fishing_100) as fishing_2km,
avg(b.cargo_tanker_100) as cargo_tanker_2km,
avg(b.non_fishing_under50m_100) as non_fishing_under50m_2km,
from grid_100 a
join
grid_100 b
on st_distance(st_geogpoint(a.lon_index/100+.005,
a.lat_index/100+.005), st_geogpoint(b.lon_index/100+.005,
b.lat_index/100+.005)) < 2
group by lat_index, lon_index

),

grid_5 as (

select 
a.lat_index,
a.lon_index,
avg(b.fishing_100) as fishing_5km,
avg(b.non_fishing_100) as non_fishing_5km,
avg(b.cargo_tanker_100) as cargo_tanker_5km,
avg(b.non_fishing_under50m_100) as non_fishing_under50m_5km
from grid_100 a
join
grid_100 b
on st_distance(st_geogpoint(a.lon_index/100+.005,
a.lat_index/100+.005), st_geogpoint(b.lon_index/100+.005,
b.lat_index/100+.005)) < 5
group by lat_index, lon_index
),


grid_10 as (
select 
a.lat_index,
a.lon_index,
avg(b.fishing_100) as fishing_10km,
avg(b.non_fishing_100) as non_fishing_10km,
avg(b.cargo_tanker_100) as cargo_tanker_10km,
avg(b.non_fishing_under50m_100) as non_fishing_under50m_10km
from grid_100 a
join
grid_100 b
on st_distance(st_geogpoint(a.lon_index/100+.005,
a.lat_index/100+.005), st_geogpoint(b.lon_index/100+.005,
b.lat_index/100+.005)) < 10
group by lat_index, lon_index

)


select
fishing_100 as fishing,
non_fishing_100 as non_fishing,
cargo_tanker_100 as cargo_tanker,
non_fishing_under50m_100 as non_fishing_under50m,
* except(fishing_100,
           non_fishing_100,
              cargo_tanker_100,
              non_fishing_under50m_100 ) from 
grid_100
left join
grid_5
using(lon_index, lat_index)
left join
grid_2
using(lon_index, lat_index)
left join
grid_10
using(lon_index, lat_index)


'''

query_to_table(q, 'project-id.proj_global_sar.ais_vessel_density_2017_2021_multi_res')

# +
### make another version that 
# -- eliminates gear
# -- makes sure it is confident it is fishing or non-fishing

q = '''
with good_vessels as (

select 
  ssvid,
  on_fishing_list_best,
  best.best_vessel_class as vessel_class,
  best.best_length_m as length_m
from
  `project-id.gfw_research.vi_ssvid_v20220101`
where 
  on_fishing_list_nn = on_fishing_list_best
  and best.best_length_m is not null
and not activity.offsetting
and activity.active_hours > 200
and activity.overlap_hours < 24
and on_fishing_list_sr = on_fishing_list_best
and 
(inferred.fishing_class_score < .15 or inferred.fishing_class_score>.85)
and best.best_vessel_class != "gear"
),

good_segs as (
  select seg_id from `project-id.gfw_research.pipe_v20201001_segs`
  where good_seg and not overlapping_and_short
  and positions > 100
),

gridded_activity as (
select 
  count(*)*5/60 hours,
  vessel_class,
  on_fishing_list_best as is_fishing_vessels,
  vessel_class in ("cargo","tanker","cargo_or_tanker") is_cargo_tanker,
  length_m < 50 as under_50m,
  floor(lon*100) lon_index,
  floor(lat*100) lat_index
from
  `project-id.gfw_research.pipe_v20201001_5min`
join
  good_vessels
using(ssvid)
join 
  good_segs
using(seg_id)
where 
  _partitiontime between '2017-01-01' and '2021-12-31'
group by 
  vessel_class, lat_index, lon_index, is_fishing_vessels, is_cargo_tanker, under_50m
)


select * from gridded_activity'''

### $40 query!!!!
# query_to_table(q, 'project-id.proj_global_sar.ais_vessel_density_2017_2021_100th_degree_v2')

# +
## Make cargo and tanker and non-fishing at a few different resolutions
##

q = '''with 
grid_100 as 
(
select 
lat_index,
lon_index,
sum(if(not is_fishing_vessels,hours/(111*111/100/100*cos(3.14/180*lat_index/100)),0)) non_fishing_100,
sum(if(is_fishing_vessels,hours/(111*111/100/100*cos(3.14/180*lat_index/100)),0)) fishing_100,
sum(if(not is_fishing_vessels and under_50m,hours/(111*111/100/100*cos(3.14/180*lat_index/100)),0)) non_fishing_under50m_100, --under_50m
sum(if(is_cargo_tanker,hours/(111*111/100/100*cos(3.14/180*lat_index/100)),0)) cargo_tanker_100 
 from `project-id.proj_global_sar.ais_vessel_density_2017_2021_100th_degree_v2`
 group by lat_index, lon_index
),

grid_2 as (
select 
a.lat_index,
a.lon_index,
avg(b.non_fishing_100) as non_fishing_2km,
avg(b.fishing_100) as fishing_2km,
avg(b.cargo_tanker_100) as cargo_tanker_2km,
avg(b.non_fishing_under50m_100) as non_fishing_under50m_2km,
from grid_100 a
join
grid_100 b
on st_distance(st_geogpoint(a.lon_index/100+.005,
a.lat_index/100+.005), st_geogpoint(b.lon_index/100+.005,
b.lat_index/100+.005)) < 2
group by lat_index, lon_index

),

grid_5 as (

select 
a.lat_index,
a.lon_index,
avg(b.fishing_100) as fishing_5km,
avg(b.non_fishing_100) as non_fishing_5km,
avg(b.cargo_tanker_100) as cargo_tanker_5km,
avg(b.non_fishing_under50m_100) as non_fishing_under50m_5km
from grid_100 a
join
grid_100 b
on st_distance(st_geogpoint(a.lon_index/100+.005,
a.lat_index/100+.005), st_geogpoint(b.lon_index/100+.005,
b.lat_index/100+.005)) < 5
group by lat_index, lon_index
),


grid_10 as (
select 
a.lat_index,
a.lon_index,
avg(b.fishing_100) as fishing_10km,
avg(b.non_fishing_100) as non_fishing_10km,
avg(b.cargo_tanker_100) as cargo_tanker_10km,
avg(b.non_fishing_under50m_100) as non_fishing_under50m_10km
from grid_100 a
join
grid_100 b
on st_distance(st_geogpoint(a.lon_index/100+.005,
a.lat_index/100+.005), st_geogpoint(b.lon_index/100+.005,
b.lat_index/100+.005)) < 10
group by lat_index, lon_index

)


select
fishing_100 as fishing,
non_fishing_100 as non_fishing,
cargo_tanker_100 as cargo_tanker,
non_fishing_under50m_100 as non_fishing_under50m,
* except(fishing_100,
           non_fishing_100,
              cargo_tanker_100,
              non_fishing_under50m_100 ) from 
grid_100
left join
grid_5
using(lon_index, lat_index)
left join
grid_2
using(lon_index, lat_index)
left join
grid_10
using(lon_index, lat_index)


'''

query_to_table(q, 'project-id.proj_global_sar.ais_vessel_density_2017_2021_multi_res_v2')
# -







# ## clip AIS vessel density to SAR footprint

q = '''select 
  lat_index,
  lon_index,
  non_fishing as non_fishing_ais_hours,
  cargo_tanker as cargo_tanker_ais_hours,
  non_fishing_under50m as non_fishing_under50m_hours,
 from 
   proj_global_sar.ais_vessel_density_2017_2021_multi_res_v2 
 join
  proj_global_sar.bathymetry_port_shore_metrics_100th_clipped
using(lat_index, lon_index)
 where elevation_m < 50'''
# this table was not actually created

# # Overpass 
#
# This counts, at 200th of a degree, the number of overpasses in each cell

# +
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
  -- floor(timestamp_diff(_partitiontime, timestamp("2017-01-01"), day)/12) twelve_day_period,
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
# -



# # Vessel denisty, average size, standard deviation of size

# +
q = '''with 
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
    `project-id.proj_sentinel1_v20210924.detect_scene_match` 
  join
    (select 
      detect_id, avg(presence) presence, 
      avg(length_m) length_m 
    from 
      `project-id.proj_sentinel1_v20210924.detect_scene_pred_*`
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
using(lat_index, lon_index)
'''

query_to_table(q, 'project-id.proj_global_sar.detections_w_overpasses_v20220509')
# -

# # How does it look?

# +
eliminated_locations = ""


boxes = []
boxes.append([-120.0,50.5,-46.8,80.5]) # huson bay, canada, etc.
# boxes.append([-155.77,59.29,-148.44,61.73]) # near anchorage
boxes.append([-120.0,50.5,-46.8,80.5]) # huson bay, canada, etc.
boxes.append([39.5,65.0,-46.8,90]) # arctic except n. atlantic
boxes.append([15.95,59.02,36.23,66.57]) # North Baltic sea
boxes.append([-173.7,62.0,-158.4,66.8]) # north beiring sea
boxes.append([130.5,50.6,-174.2,67.8]) #sea of okhotsk
# boxes.append([-72.84,7.75,-69.62,10.86]) #lake maracaibo
boxes.append([3.5,78.1,31.9,85.0]) #north of Salvbard
boxes.append([-179.8,57.4,-156.5,62.2]) #beiring sea, more southern, because it didn't work
boxes.append([-44.82,-57.93,-29.05,-50.61]) ## south georgia island
boxes.append([31.4,61.4,60.3,73.1])## far northeast russia -- a small area
boxes.append([-27.61,68,-19.47,68.62]) # tiny piece of ice near ne iceland that annoyed me

eliminated_locations+= '''and not (
'''

for b in boxes:
    min_lon, min_lat, max_lon, max_lat = b
    if min_lon > max_lon:
        bounding_string = '''     ( (detect_lon > {min_lon} or detect_lon < {max_lon} ) 
        and detect_lat> {min_lat} and detect_lat < {max_lat} ) or '''
    else:
        bounding_string = '''     ( detect_lon > {min_lon} and detect_lon < {max_lon}  
        and detect_lat> {min_lat} and detect_lat < {max_lat} ) or '''
    eliminated_locations+=bounding_string.format(min_lon=min_lon,
                                                         max_lon=max_lon, 
                                                         max_lat=max_lat, 
                                                         min_lat=min_lat)
eliminated_locations = eliminated_locations[:-3] + ")\n"

print(eliminated_locations)
# -

scale = 10
q = f'''
select 
floor(detect_lat*{scale}) lat_index,
floor(detect_lon*{scale}) lon_index,
sum(1/overpasses_2017_2021) detections,
avg(length_m) average_length,
from 
proj_global_sar.detections_w_overpasses_v20220509
where
repeats_100m_180days_forward <3 and 
       repeats_100m_180days_back < 3 and
       repeats_100m_180days_center < 3 
      and (scene_detections <=5 or scene_quality > .5)
      {eliminated_locations}
      and extract(date from detect_timestamp) between "2017-01-01" and "2021-12-31"
      and overpasses_2017_2021 > 40
      and presence > .7
group by lat_index, lon_index
'''
df = pd.read_gbq(q)

# +
raster = psm.rasters.df2raster(df,
                               'lon_index', 'lat_index',
               'detections', xyscale=scale, 
                per_km2=True, origin = 'lower')


norm = mpcolors.LogNorm(vmin=.01, vmax=100)

with psm.context(psm.styles.dark):
    fig = plt.figure(figsize=(15, 8))
    ax = psm.create_map()
#         psm.add_land(ax)
#         psm.add_eezs()
#         psm.add_eezs(ax, edgecolor=eez_color, linewidth=eez_linewidth)
    psm.add_raster(
        raster*1000,
        norm=norm,
        cmap='presence',
        origin="lower",
    )
#     psm.add_colorbar(ax, label=r"hours per $\mathregular{km^2}$", width=0.5)

plt.title(f"Vessels per km2 detected by Sentinel-1, 2017-2021")

plt.savefig(f"images/vessels_normalized.png",dpi=300, bbox_inches="tight")

# +
raster = psm.rasters.df2raster(df,
                               'lon_index', 'lat_index',
               'average_length', xyscale=scale, 
                per_km2=False, origin = 'lower')


norm = mpcolors.Normalize(vmin=0, vmax=200)

with psm.context(psm.styles.dark):
    fig = plt.figure(figsize=(15, 8))
    ax = psm.create_map()
#         psm.add_land(ax)
#         psm.add_eezs()
#         psm.add_eezs(ax, edgecolor=eez_color, linewidth=eez_linewidth)
    psm.add_raster(
        raster,
        norm=norm,
        cmap='presence',
        origin="lower",
    )
#     psm.add_colorbar(ax, label=r"hours per $\mathregular{km^2}$", width=0.5)

plt.title(f"Vessels per km2 detected by Sentinel-1, 2017-2021")

plt.savefig(f"images/vessels_length.png",dpi=300, bbox_inches="tight")
# -

# # Create length, standard deviation of length, and density of vessel detections (non infrastructure) computed over different windows

# +
q = '''select 
floor(detect_lat*100) lat_index,
floor(detect_lon*100) lon_index,
sum(1/overpasses_2017_2021) detections_normalized,
count(*) detections,
avg(length_m) avg_length_m,
stddev(length_m) stddev_length_m
from 
proj_global_sar.detections_w_overpasses_v20220509
where
repeats_100m_180days_forward <3 and 
       repeats_100m_180days_back < 3 and
       repeats_100m_180days_center < 3 
      and (scene_detections <=5 or scene_quality > .5)
      and extract(date from detect_timestamp) between "2017-01-01" and "2021-12-31"
      and presence > .7
group by lat_index, lon_index


'''

query_to_table(q, 'project-id.proj_global_sar.vessel_density_statistics_100th_2017_2021')
# +
q = '''with source_table as (


select 
floor(detect_lat*20) lat_index,
floor(detect_lon*20) lon_index,
1/overpasses_2017_2021 detections_norm,
st_geogpoint(detect_lon, detect_lat) pos,
detect_lat,
detect_lon,
length_m
from 
proj_global_sar.detections_w_overpasses_v20220509
where
repeats_100m_180days_forward <3 and 
       repeats_100m_180days_back < 3 and
       repeats_100m_180days_center < 3 
      and (scene_detections <=5 or scene_quality > .5)
      and extract(date from detect_timestamp) between "2017-01-01" and "2021-12-31"
      and overpasses_2017_2021 > 40
      and presence > .7
),

distinct_cells as (
select distinct lat_index, lon_index
from source_table
),


stats_5km as (
select 
lat_index,
lon_index, 
sum(detections_norm/(111*111/20/20/cos(detect_lat*3.1416/180))) detections_km2_5km,
stddev(length_m) length_m_stddev_5km,
avg(length_m) length_m_ave_5km
from 
source_table
group by lat_index, lon_index
),

stats_5km_radius as (
select 
  a.lat_index, 
  a.lon_index,
  sum(detections_norm)/(5*5*3.1416) detections_km2_5km_radius,
  stddev(length_m) length_m_stddev_5km_radius,
  avg(length_m) length_m_ave_5km_radius
from distinct_cells a
cross join
  source_table b
where st_distance(st_geogpoint(a.lon_index/20 + .5/20, a.lat_index/20 + .5/20), pos) < 5000
  group by lat_index, lon_index
),


stats_10km_radius as (
select 
  a.lat_index, 
  a.lon_index,
  sum(detections_norm)/(10*10*3.1416) detections_km2_10km_radius,
  stddev(length_m) length_m_stddev_10km_radius,
  avg(length_m) length_m_ave_10km_radius
from distinct_cells a
cross join
  source_table b
where st_distance(st_geogpoint(a.lon_index/20 + .5/20, a.lat_index/20 + .5/20), pos) < 10000
  group by lat_index, lon_index
),

stats_20km_radius as (
select 
  a.lat_index, 
  a.lon_index,
  sum(detections_norm)/(10*10*3.1416) detections_km2_20km_radius,
  stddev(length_m) length_m_stddev_20km_radius,
  avg(length_m) length_m_ave_20km_radius
from distinct_cells a
cross join
  source_table b
where st_distance(st_geogpoint(a.lon_index/20 + .5/20, a.lat_index/20 + .5/20), pos) < 20000
  group by lat_index, lon_index
)

select 
  lat_index/20 + .5/20 as lat,
  lon_index/20 + .5/20 as lon,
  *
from 
  stats_5km
left join
  stats_5km_radius
using(lat_index, lon_index)
left join
  stats_10km_radius
using(lat_index, lon_index)
left join
  stats_20km_radius
using(lat_index, lon_index)

'''

query_to_table(q, 'project-id.proj_global_sar.vessel_density_statistics_2017_2021')
# -

# # Bathymetry, Distance from Port, Distance to Shore

q = '''with bathymetry_table as (
  SELECT 
    round(lon*100) lon_index,
    round(lat*100) lat_index, 
    avg(elevation_m) elevation_m -- unlikley there are 
    -- actual multiple values for this, but avg to make
    -- sure we don't have duplicates
  FROM 
    `project-id.pipe_static.bathymetry` 
  group by
    lat_index, lon_index
),

distance_from_port_table as (
  SELECT 
    round(lon*100) lon_index,
    round(lat*100) lat_index, 
    avg(distance_from_port_m) distance_from_port_m -- unlikley there are 
    -- actual multiple values for this, but avg to make
    -- sure we don't have duplicates
  FROM 
    `project-id.pipe_static.distance_from_port_20201105`
  group by
    lat_index, lon_index
),


distance_from_shore_table as (
  SELECT 
    round(lon*100) lon_index,
    round(lat*100) lat_index, 
    avg(distance_from_shore_m) distance_from_shore_m -- unlikley there are 
    -- actual multiple values for this, but avg to make
    -- sure we don't have duplicates
  FROM 
    `project-id.pipe_static.distance_from_shore`
  group by
    lat_index, lon_index
)

select 
  *
from 
  distance_from_shore_table
full outer join
  distance_from_port_table
using(lat_index, lon_index)
full outer join
  bathymetry_table
using(lat_index, lon_index)

'''
query_to_table(q, 'project-id.proj_global_sar.bathymetry_port_shore_metrics_100th')

# # Clip bathyemtry, port, and shore to the area imaged


# +
q = '''
with

raster_table as (

select 
    lat_index, 
    lon_index,
    overpasses,
    first_value(scene_id) over (partition by cast(lat_index as int64),
                                         cast(lon_index as int64)
                                          order by scene_id) scene_id
from 
  `project-id.proj_sentinel1_v20210924.detect_foot_raster_10`
),

enough_overpasses as (
select 
  lat_index, lon_index, 
  sum(overpasses) overpasses 
from 
  raster_table
group by 
  lat_index, lon_index 
having overpasses>10),

limit_scene_table as (
select 
  distinct scene_id 
from 
  raster_table
join
  enough_overpasses
using(lat_index,lon_index)),

scene_outlines as 
(SELECT 
st_buffer(
    st_union_agg(
        st_simplify(
            st_geogfromtext(footprint_wkt_5km)
            ,20*1000 -- simplify each by 20 km
        )
    ),
    1000*100) as unioned_outline -- buffer the unioned set by 100km
 FROM 
   `project-id.proj_sentinel1_v20210924.detect_foot_raw_*`
 join
   limit_scene_table
  using
   (scene_id)
 where safe.st_geogfromtext(footprint_wkt_5km) is not null
)

select 
  lat_index,
  lon_index,
  distance_from_shore_m,
  distance_from_port_m,
  elevation_m
from 
  proj_global_sar.bathymetry_port_shore_metrics_100th
cross join
  scene_outlines
where 
  lat_index/100 between -80 and 85
  and st_contains(unioned_outline, 
              st_geogpoint(lon_index/100 + .5/100, 
                           lat_index/100 + .5/100))'''

query_to_table(q, 'project-id.proj_global_sar.bathymetry_port_shore_metrics_100th_clipped')
# -



# # Slope 

# +
q_temp = '''bath_grid{i} AS (
SELECT
  lat,
  lon,
  elevation_m,
  LEAD( elevation_m,{i}) OVER (PARTITION BY lat ORDER BY lon) ev_11,
  LAG( elevation_m,{i}) OVER (PARTITION BY lat ORDER BY lon) ev_10,
  LEAD( elevation_m,{i}) OVER (PARTITION BY lon ORDER BY lat) ev_01,
  LAG( elevation_m,{i}) OVER (PARTITION BY lon ORDER BY lat) ev_00
FROM elevation_table),

slope{i} AS (
    SELECT
      lat,
      lon,
      elevation_m,
      SQRT(s1*s1 + s2*s2) slope{i2}_km -- this is totally not right... but should be close
    FROM (
      SELECT
        lat,
        lon,
        elevation_m,
        ABS(ev_11-ev_10)/cos(lat/100*3.1416/180)/({i}*2) s1,
        ABS(ev_01-ev_00)/({i}*2) s2
      FROM
        bath_grid{i} )),
'''


q = '''with
elevation_table as (select 
    CAST(lat*100 AS int64) lat,
    CAST(lon*100 AS int64) AS lon,
    elevation_m
  FROM
    `project-id.pipe_static.bathymetry` ),
'''
# average_depth_2km as (
# select 
# avg(b.elevation_m) elevation_m_2km,
# a.lat lat,
# a.lon lon
# from elevation_table a
# cross join
# elevation_table b
# where st_distance(st_geogpoint(a.lon/100,a.lat/100), 
#               st_geogpoint(b.lon/100,b.lat/100))<=2000
# group by lat, lon
# ),

# average_depth_5km as (
# select 
# avg(b.elevation_m) elevation_m_5km,
# a.lat lat,
# a.lon lon
# from elevation_table a
# cross join
# elevation_table b
# where st_distance(st_geogpoint(a.lon/100,a.lat/100), 
#               st_geogpoint(b.lon/100,b.lat/100))<=5000
# group by lat,lon
# ),
# '''
for i in range(1,11):
    q+=q_temp.format(i=i,i2=i*2)

q = q[:-2]
q+= '''

select
lat as lat_index,
lon as lon_index,
slope1.elevation_m as elevation_m,
* except (elevation_m, lat, lon)
from 
slope1'''
for i in range(2,11):
    q+=f'''
left join
slope{i}
using(lat, lon)'''


# ## copy to clipboard to identify it
import pyperclip
pyperclip.copy(q)
# -

query_to_table(q, 'project-id.proj_global_sar.slope')

# # Average Depth

# +
q = '''with
elevation_table as (select 
    CAST(lat*100 AS int64) as lat,
    CAST(lon*100 AS int64) AS lon,
    elevation_m
  FROM
    `project-id.pipe_static.bathymetry`
    where lat between -80 and 80
    and elevation_m < 500
    )


select 
avg(b.elevation_m) elevation_m_4km,
STDDEV(b.elevation_m) elevation_m_stddev_4km,
a.lat lat_index,
a.lon lon_index,
a.elevation_m
from elevation_table a
cross join
elevation_table b
where st_distance(st_geogpoint(a.lon/100 + .005, a.lat/100 +.005), 
              st_geogpoint(b.lon/100 + .005, b.lat/100 + .005))<=4000
group by lat_index, lon_index, elevation_m


'''

query_to_table(q, 'project-id.proj_global_sar.average_depth_4km_radius')

# +
q = '''
with
elevation_table as (
  select 
  CAST(lat*100 AS int64) as lat,
  CAST(lon*100 AS int64) AS lon,
  elevation_m
FROM
  `project-id.pipe_static.bathymetry`
where 
  lat between -80 and 85
  and elevation_m < 1000
    )

select 
avg(b.elevation_m) elevation_m_2km,
STDDEV(b.elevation_m) elevation_m_stddev_2km,
a.lat lat_index,
a.lon lon_index,
a.elevation_m
from elevation_table a
cross join
elevation_table b
where st_distance(st_geogpoint(a.lon/100 + .005, a.lat/100 +.005), 
              st_geogpoint(b.lon/100 + .005, b.lat/100 + .005))<=2000
group by lat_index, lon_index, elevation_m


'''

query_to_table(q, 'project-id.proj_global_sar.average_depth_2km_radius')
# -



# # Organize Environmental Data
#

# +
q = '''with base_table as (
  select * from 
  `project-id.environmental_data.copernicus_global_analysis_forecast_phy_001_024_hourly_t_u_v_ssh` 
  where date(time) between "2017-01-01" and "2019-12-31"),


stddev_alltime as (
select lat, lon,
stddev(pow(pow(uo,2)+pow(vo,2),.5)) stddev_current
from base_table
group by lat, lon

),


daily_stats as (


SELECT 
lat, lon,
avg(the_tao) avg_temp,
stddev(the_tao) stddev_daily_avg_temp,
avg(the_tao_stddev) avg_daily_stddev_temp,
avg(avg_current) avg_current,
stddev(avg_current) stdev_daily_avg_current,
avg(stddev_current) avg_daily_stddev_current,
from
(select 
lat, 
lon,
date(time) date,
avg(thetao) the_tao,
stddev(thetao) the_tao_stddev,
avg(pow(pow(uo,2)+pow(vo,2),.5)) avg_current,
stddev(pow(pow(uo,2)+pow(vo,2),.5)) stddev_current
FROM 
  base_table
group by lat, lon, date)
group by lat, lon)


select * 
from daily_stats
join stddev_alltime
using(lat, lon)'''

query_to_table(q, 
    'project-id.proj_global_sar.temp_currents_12thdegree_2017_2019')

# +
# bilinear interpolation!!!!

q = '''
CREATE TEMP FUNCTION bilinear_interpolation(Q11 float64,
Q12 float64, Q22 float64, Q21 float64, x1 float64, x2 float64,
y1 float64, y2 float64, x float64, y float64) AS (
  # see https://en.wikipedia.org/wiki/Bilinear_interpolation
  # Q11 is the value at x1, y1, Q12 is the value at x1, y2, etc.
  # x and y are the coordinates we want the value for
  1/( (x2 - x1) * (y2 - y1)) *
    (
       Q11 * (x2 - x) * (y2 - y)
     + Q21 * (x - x1) * (y2 - y)
     + Q12 * (x2 - x) * (y - y1)
     + Q22 * (x - x1) * (y - y1)
    )
  );


with base_primary as 
(select
  lat as lat0, 
  lon as lon0,
  lat + 1 lat1, 
  lon + 1 lon1,
  avg_temp,
  avg_daily_stddev_temp,
  avg_current,
  avg_daily_stddev_current,
  stddev_current,
  1 as v,
from 
  proj_global_sar.temp_currents_12thdegree_2017_2019
),

base_augmented as 
-- okay, this is to help with the padding...
(
select
  lat -1 as lat0, 
  lon -1  as lon0,
  lat as lat1, 
  lon as lon1,
  avg_temp,
  avg_daily_stddev_temp,
  avg_current,
  avg_daily_stddev_current,
  stddev_current,
  2 as v
from 
  proj_global_sar.temp_currents_12thdegree_2017_2019
),

base as (
select * from 
  (
  select 
    *, row_number() over (partition by lat0, lon0 order by v) row
  from
    (select * from base_primary
      union all
    select * from base_augmented)
  )
where row = 1
),


with_temp_grid as (
select 
  a.lat0 as lat12th,
  a.lon0 as lon12th,
  a.lat0 as lat_bin0, 
  a.lon0 as lon_bin0,
  a.lat1 as lat_bin1,
  a.lon1 as lon_bin1,
  a.lat0/12  - 80 as lat0, 
  a.lon0/12  - 180 lon0,
  a.lat1/12  - 80 as lat1,
  a.lon1/12  - 180 as lon1,

  a.avg_temp as avg_temp00,
  ifnull(b.avg_temp, a.avg_temp) as avg_temp10,
  ifnull(c.avg_temp, a.avg_temp) as avg_temp11,
  ifnull(d.avg_temp, a.avg_temp) as avg_temp01,

  a.avg_current as avg_current00,
  ifnull(b.avg_current, a.avg_current) as avg_current10,
  ifnull(c.avg_current, a.avg_current) as avg_current11,
  ifnull(d.avg_current,a.avg_current)  as avg_current01,

  a.avg_daily_stddev_temp as avg_daily_stddev_temp00,
  ifnull(b.avg_daily_stddev_temp, a.avg_daily_stddev_temp) as avg_daily_stddev_temp10,
  ifnull(c.avg_daily_stddev_temp, a.avg_daily_stddev_temp) as avg_daily_stddev_temp11,
  ifnull(d.avg_daily_stddev_temp, a.avg_daily_stddev_temp) as avg_daily_stddev_temp01,

  a.stddev_current as stddev_current00,
  ifnull(b.stddev_current, a.stddev_current) as stddev_current10,
  ifnull(c.stddev_current, a.stddev_current) as stddev_current11,
  ifnull(d.stddev_current, a.stddev_current) as stddev_current01,

from 
  base a
left join 
  base b
on 
  a.lon1 = b.lon0
  and a.lat0 = b.lat0
left join
  base c
on 
  a.lon1 = c.lon0
  and a.lat1 = c.lat0
left join
  base d
on 
  a.lon0 = d.lon0
  and a.lat1 = d.lat0
),

lat_lon_grid as (
select 
  round(lat*100)/100 + 1/200 as lat, 
  round(lon*100)/100 + 1/200 as lon, 
  round(lat*100) as lat_index, 
  round(lon*100) as lon_index 
from
  `project-id.pipe_static.bathymetry`
)

select
  cast(lat_index as int64) lat_index,
  cast(lon_index as int64) lon_index,
  bilinear_interpolation(avg_temp00,
    avg_temp01, avg_temp11, avg_temp10, lon0, lon1,
    lat0, lat1, lon, lat) avg_temp,
  bilinear_interpolation(avg_current00,
    avg_current01, avg_current11, avg_current10, lon0, lon1,
    lat0, lat1, lon, lat) avg_current,
  bilinear_interpolation(avg_daily_stddev_temp00,
    avg_daily_stddev_temp01, avg_daily_stddev_temp11, avg_daily_stddev_temp10, lon0, lon1,
    lat0, lat1, lon, lat) avg_daily_stddev_temp,
  bilinear_interpolation(stddev_current00,
    stddev_current01, stddev_current11, stddev_current10, lon0, lon1,
    lat0, lat1, lon, lat) stddev_current
from 
  lat_lon_grid
join
  with_temp_grid
on
  floor((lat_index/100 +80 +1/200 )*12) = lat12th
  and floor((lon_index/100+180 +1/200 )*12) = lon12th
'''


query_to_table(q, 
    'project-id.proj_global_sar.temperature_currents_100')
# -



# +
q = '''


CREATE TEMP FUNCTION bilinear_interpolation(Q11 float64,
Q12 float64, Q22 float64, Q21 float64, x1 float64, x2 float64,
y1 float64, y2 float64, x float64, y float64) AS (
  # see https://en.wikipedia.org/wiki/Bilinear_interpolation
  # Q11 is the value at x1, y1, Q12 is the value at x1, y2, etc.
  # x and y are the coordinates we want the value for
  1/( (x2 - x1) * (y2 - y1)) *
    (
       Q11 * (x2 - x) * (y2 - y)
     + Q21 * (x - x1) * (y2 - y)
     + Q12 * (x2 - x) * (y - y1)
     + Q22 * (x - x1) * (y - y1)
    )
  );


with 

source_table as (
select 
  lat, lon, avg(chl) chl
from 
  proj_global_sar.chlorophyll_24thdegree_yearly
group by lat, lon
),

base_primary as 
(select
  lat as lat0, 
  lon as lon0,
  lat + 1 lat1, 
  lon + 1 lon1,
  chl,
  1 as v,
from 
  source_table
),

base_augmented as 
-- okay, this is to help with the padding...
(
select
  lat -1 as lat0, 
  lon -1  as lon0,
  lat as lat1, 
  lon as lon1,
  chl,
  2 as v
from 
  source_table
),

base as (
select * from 
  (
  select 
    *, 
    row_number() over (partition by lat0, lon0 order by v) row
 from
    (
      select * from base_primary
      union all
      select * from base_augmented
    )
  )
where row = 1
),

with_temp_grid as (
select 
  a.lat0 as lat12th,
  a.lon0 as lon12th,
  a.lat0 as lat_bin0, 
  a.lon0 as lon_bin0,
  a.lat1 as lat_bin1,
  a.lon1 as lon_bin1,
  a.lat0/24 - 90 as lat0, 
  a.lon0/24 - 180 lon0,
  a.lat1/24 - 90 as lat1,
  a.lon1/24 - 180 as lon1,
  a.chl as chl00,
  ifnull(b.chl, a.chl) as chl10,
  ifnull(c.chl, a.chl) as chl11,
  ifnull(d.chl, a.chl) as chl01,
 from 
  base a
 left join 
  base b
 on 
   a.lon1 = b.lon0
   and a.lat0 = b.lat0
 left join
   base c
 on 
   a.lon1 = c.lon0
   and a.lat1 = c.lat0
 left join
   base d
 on 
   a.lon0 = d.lon0
   and a.lat1 = d.lat0
),

lat_lon_grid as (
select 
  round(lat*100)/100 + 1/200 as lat, 
  round(lon*100)/100 + 1/200 as lon, 
  round(lat*100) as lat_index, 
  round(lon*100) as lon_index 
from
  `project-id.pipe_static.bathymetry`)
  

select
  cast(lat_index as int64) lat_index,
  cast(lon_index as int64) lon_index,
  bilinear_interpolation(chl00,
  chl01, chl11, chl10, lon0, lon1,
  lat0, lat1, lon, lat) chl,
from 
  lat_lon_grid
join
  with_temp_grid
on
  floor((lat_index/100 +90 +1/200 )*24)   = lat12th
  and floor((lon_index/100+180 +1/200 )*24) = lon12th'''

query_to_table(q, 
    'project-id.proj_global_sar.chlorophyll_100')
# +
q = """

with 

vessel_info as (
select 
  ssvid,
  best.best_vessel_class,
  on_fishing_list_best,
  on_fishing_list_nn
from 
  `project-id.gfw_research.vi_ssvid_v20220101` 
),


detections_table as (
select 
  detect_id,
  scene_id,
  source,
  detect_timestamp,
  detect_lat,
  detect_lon,
  presence,
  length_m,
  ssvid,
  score,
  confidence,
  month,
  scene_detections,
  bad_detections,
  scene_quality,
  repeats_100m_180days_back,
  repeats_100m_180days_center,
  repeats_100m_180days_forward,
  overpasses_year,
  overpasses_2017_2021
from 
  proj_global_sar.detections_w_overpasses_v20220509
),

eez_table as (
  select 
  detect_id, 
  array_agg(ifnull(MRGID,0)) as MRGID, 
  array_agg(ifnull(ISO_TER1,"")) as ISO_TER1, 
  array_agg(ifnull(TERRITORY1,"")) AS TERRITORY1
from detections_table
  CROSS JOIN
(select wkt, MRGID, ISO_TER1, TERRITORY1 from `project-id.minderoo.marine_regions_v11`)
WHERE
  ST_CONTAINS(SAFE.ST_GEOGFROMTEXT(wkt),ST_GEOGPOINT(detect_lon,detect_lat) )
group by 
  detect_id
),
--
-- vessel density at 100th of a degree bins, and averaged
ais_vessel_density_2017_2021_multi_res as (
select 
  lon_index, lat_index, -- at 100th of degree
  non_fishing, cargo_tanker,  non_fishing_5, cargo_tanker_5, 
  non_fishing_2, cargo_tanker_2, non_fishing_10, cargo_tanker_10
from 
  proj_global_sar.ais_vessel_density_2017_2021_multi_res),
--
--
vessel_density_statistics_2017_2021 as (
select 
-- lat_index and lon_index are at 20th of a degree, so floor(lat/20)
  lat_index, lon_index, detections_km2_5km, length_m_stddev_5km, length_m_ave_5km, 
  detections_km2_5km_radius, length_m_stddev_5km_radius, length_m_ave_5km_radius, 
  detections_km2_10km_radius, length_m_stddev_10km_radius, length_m_ave_10km_radius,
  detections_km2_20km_radius, length_m_stddev_20km_radius, length_m_ave_20km_radius
 from 
   proj_global_sar.vessel_density_statistics_2017_2021
),

slope_table as (
select 
  lat_index, lon_index, -- lat and lon_index at 100th of a degree
  elevation_m, slope2_km, slope4_km, slope6_km, 
  slope8_km, slope10_km, slope12_km, slope14_km, 
  slope16_km, slope18_km, slope20_km
from 
  proj_global_sar.slope
),

average_depth_4km_radius as (
select 
  elevation_m_4km,
  elevation_m_stddev_4km,
  lat_index, -- 100th of a degree
  lon_index
from
  proj_global_sar.average_depth_4km_radius
),

average_depth_2km_radius as (
select 
  elevation_m_2km,
  elevation_m_stddev_2km,
  lat_index, -- 100th of a degree
  lon_index
from
  proj_global_sar.average_depth_2km_radius
),

distance_from_shore_table  as (
select 
  distance_from_shore_m,
  floor(lat*100) as lat_index,
  floor(lon*100) as lon_index,
from 
  `pipe_static.distance_from_shore`
),

distance_from_port_table as (
select 
  distance_from_port_m,
  floor(lat*100) as lat_index,
  floor(lon*100) as lon_index
from
  `pipe_static.distance_from_port_20201105`
)


select 
  * except(lat_index, lon_index) 
from 
  detections_table a 
left join 
  vessel_info
using(ssvid)
left join
  eez_table b 
using(detect_id)
left join
  ais_vessel_density_2017_2021_multi_res c
on floor(detect_lat*100) = c.lat_index
and floor(detect_lon*100) = c.lon_index
left join 
  vessel_density_statistics_2017_2021 e
on floor(detect_lat*20) = e.lat_index
and floor(detect_lon*20) = e.lon_index
left join 
  slope_table f
on floor(detect_lat*100) = f.lat_index
and floor(detect_lon*100) = f.lon_index
left join 
  average_depth_2km_radius g
on floor(detect_lat*100) = g.lat_index
and floor(detect_lon*100) = g.lon_index
left join 
  average_depth_4km_radius h
on floor(detect_lat*100) = h.lat_index
and floor(detect_lon*100) = h.lon_index
left join
  distance_from_shore_table i
ON floor( detect_lat*100) = i.lat_index
AND floor(detect_lon*100) = i.lon_index
left join
  distance_from_port_table j
ON floor(detect_lat*100) = j.lat_index
AND floor(detect_lon*100) =j.lon_index


"""

query_to_table(q, "project-id.proj_global_sar.detections_w_features_v20220509")
# -



# # Add Features

# +
# v20220509 -- this does not include environmental data

q = '''

with 

vessel_info as (
select 
  ssvid,
  best.best_vessel_class,
  on_fishing_list_best,
  on_fishing_list_nn
from 
  `project-id.gfw_research.vi_ssvid_v20220101` 
),


detections_table as (
select 
  detect_id,
  scene_id,
  source,
  detect_timestamp,
  detect_lat,
  detect_lon,
  presence,
  length_m,
  ssvid,
  score,
  confidence,
  month,
  scene_detections,
  bad_detections,
  scene_quality,
  repeats_100m_180days_back,
  repeats_100m_180days_center,
  repeats_100m_180days_forward,
  overpasses_year,
  overpasses_2017_2021
from 
  proj_global_sar.detections_w_overpasses_v20220509
),

eez_table as (
  select 
  detect_id, 
  array_agg(ifnull(MRGID,0)) as MRGID, 
  array_agg(ifnull(ISO_TER1,"")) as ISO_TER1, 
  array_agg(ifnull(TERRITORY1,"")) AS TERRITORY1
from detections_table
  CROSS JOIN
(select wkt, MRGID, ISO_TER1, TERRITORY1 from `project-id.minderoo.marine_regions_v11`)
WHERE
  ST_CONTAINS(SAFE.ST_GEOGFROMTEXT(wkt),ST_GEOGPOINT(detect_lon,detect_lat) )
group by 
  detect_id
),
--
-- vessel density at 100th of a degree bins, and averaged
ais_vessel_density_2017_2021_multi_res as (
select 
  lon_index, lat_index, -- at 100th of degree
  non_fishing, cargo_tanker,  non_fishing_5, cargo_tanker_5, 
  non_fishing_2, cargo_tanker_2, non_fishing_10, cargo_tanker_10
from 
  proj_global_sar.ais_vessel_density_2017_2021_multi_res),
--
--
vessel_density_statistics_2017_2021 as (
select 
-- lat_index and lon_index are at 20th of a degree, so floor(lat/20)
  lat_index, lon_index, detections_km2_5km, length_m_stddev_5km, length_m_ave_5km, 
  detections_km2_5km_radius, length_m_stddev_5km_radius, length_m_ave_5km_radius, 
  detections_km2_10km_radius, length_m_stddev_10km_radius, length_m_ave_10km_radius,
  detections_km2_20km_radius, length_m_stddev_20km_radius, length_m_ave_20km_radius
 from 
   proj_global_sar.vessel_density_statistics_2017_2021
),

slope_table as (
select 
  lat_index, lon_index, -- lat and lon_index at 100th of a degree
  elevation_m, slope2_km, slope4_km, slope6_km, 
  slope8_km, slope10_km, slope12_km, slope14_km, 
  slope16_km, slope18_km, slope20_km
from 
  proj_global_sar.slope
),

average_depth_4km_radius as (
select 
  elevation_m_4km,
  elevation_m_stddev_4km,
  lat_index, -- 100th of a degree
  lon_index
from
  proj_global_sar.average_depth_4km_radius
),

average_depth_2km_radius as (
select 
  elevation_m_2km,
  elevation_m_stddev_2km,
  lat_index, -- 100th of a degree
  lon_index
from
  proj_global_sar.average_depth_2km_radius
),

distance_from_shore_table  as (
select 
  distance_from_shore_m,
  floor(lat*100) as lat_index,
  floor(lon*100) as lon_index,
from 
  `pipe_static.distance_from_shore`
),

distance_from_port_table as (
select 
  distance_from_port_m,
  floor(lat*100) as lat_index,
  floor(lon*100) as lon_index
from
  `pipe_static.distance_from_port_20201105`
)


select 
  * except(lat_index, lon_index) 
from 
  detections_table a 
left join 
  vessel_info
using(ssvid)
left join
  eez_table b 
using(detect_id)
left join
  ais_vessel_density_2017_2021_multi_res c
on floor(detect_lat*100) = c.lat_index
and floor(detect_lon*100) = c.lon_index
left join 
  vessel_density_statistics_2017_2021 e
on floor(detect_lat*20) = e.lat_index
and floor(detect_lon*20) = e.lon_index
left join 
  slope_table f
on floor(detect_lat*100) = f.lat_index
and floor(detect_lon*100) = f.lon_index
left join 
  average_depth_2km_radius g
on floor(detect_lat*100) = g.lat_index
and floor(detect_lon*100) = g.lon_index
left join 
  average_depth_4km_radius h
on floor(detect_lat*100) = h.lat_index
and floor(detect_lon*100) = h.lon_index
left join
  distance_from_shore_table i
ON floor( detect_lat*100) = i.lat_index
AND floor(detect_lon*100) = i.lon_index
left join
  distance_from_port_table j
ON floor(detect_lat*100) = j.lat_index
AND floor(detect_lon*100) =j.lon_index


'''

query_to_table(q, 'project-id.proj_global_sar.detections_w_features_v20220509')
# -


# # Features for v20220509
#
#  - elevation_m_2km: average depth, meters, of each cell, averaged over a circle 2km radius
#  - elevation_m_stddev_2km: standard deviation of depth, meters, over a circle 2km radius
#  - elevation_m_4km: over 4km radius
#  - elevation_m_stddev_4km: 4km radius
#  - non_fishing: hours of non-fishing vessels per km2 in AIS, 2017 to 2022
#  - cargo_tanker: hours of cargo or tanker hours per km2 in AIS, 2017 to 2022	
#  - non_fishing_5km: average hours of non-fishing vessels per km2 in AIS, 2012 to 2022, averaged over a circle with radius of 5km	
#  - cargo_tanker_5km: average hours of cargo tanker vesesls, averaged over a a radius of 5km  	
#  - non_fishing_2km: 	
#  - cargo_tanker_2km:	
#  - non_fishing_10km:
#  - cargo_tanker_10km: 
#  - elevation_m: depth in meters
#  - slope2_km: slope, calculated at 2km 
#  - slope4_km: slope, calculated over 4km
#  - slope6_km:
#  - slope8_km:
#  - slope10_km:
#  - slope12_km:
#  - slope14_km:
#  - slope16_km:
#  - slope18_km:
#  - slope20_km:
#  - length_m: inferred length from the neural net of the detection
#  - distance_from_shore_m: distance from shore in meters
#  - distance_from_port_m: distance from port in meters
#  - detections_km2_5km: density of detections averaged over a 20th of a degree
#  - length_m_stddev_5km: standard deviation of length of detections over a 20th of a degree
#  - length_m_ave_5km: average length of detections over a 20th of a degree
#  - detections_km2_5km_radius: denisty of detections over a circle with a 5km radius
#  - length_m_stddev_5km_radius: standard deviation of detections over a circle with a 5km radius
#  - length_m_ave_5km_radius: average length of detections over a circle with a 5km radius
#  - detections_km2_10km_radius	
#  - length_m_stddev_10km_radius	
#  - length_m_ave_10km_radius	
#  - detections_km2_20km_radius	
#  - length_m_stddev_20km_radius	
#  - length_m_ave_20km_radius
#

# +
# v20220720 -- includes environmental data



q = '''
with 

vessel_info as (
select 
  ssvid,
  best.best_vessel_class,
  on_fishing_list_best,
  on_fishing_list_nn
from 
  `project-id.gfw_research.vi_ssvid_v20220101` 
),


detections_table as (
select 
  detect_id,
  scene_id,
  source,
  detect_timestamp,
  detect_lat,
  detect_lon,
  presence,
  length_m,
  ssvid,
  score,
  confidence,
  month,
  scene_detections,
  bad_detections,
  scene_quality,
  repeats_100m_180days_back,
  repeats_100m_180days_center,
  repeats_100m_180days_forward,
  overpasses_year,
  overpasses_2017_2021
from 
  proj_global_sar.detections_w_overpasses_v20220509
),

eez_table as (
  select 
  detect_id, 
  array_agg(ifnull(MRGID,0)) as MRGID, 
  array_agg(ifnull(ISO_TER1,"")) as ISO_TER1, 
  array_agg(ifnull(TERRITORY1,"")) AS TERRITORY1
from detections_table
  CROSS JOIN
(select wkt, MRGID, ISO_TER1, TERRITORY1 from `project-id.minderoo.marine_regions_v11`)
WHERE
  ST_CONTAINS(SAFE.ST_GEOGFROMTEXT(wkt),ST_GEOGPOINT(detect_lon,detect_lat) )
group by 
  detect_id
),
--
-- vessel density at 100th of a degree bins, and averaged
ais_vessel_density_2017_2021_multi_res as (
select 
  lon_index, lat_index, -- at 100th of degree
  non_fishing, cargo_tanker, non_fishing_under50m
from 
  proj_global_sar.ais_vessel_density_2017_2021_multi_res),
--
--
vessel_density_statistics_2017_2021 as (
select 
-- lat_index and lon_index are at 20th of a degree, so floor(lat/20)
  lat_index, lon_index, detections_km2_5km, length_m_stddev_5km, length_m_ave_5km, 
  detections_km2_5km_radius, length_m_stddev_5km_radius, length_m_ave_5km_radius, 
  detections_km2_10km_radius, length_m_stddev_10km_radius, length_m_ave_10km_radius,
  detections_km2_20km_radius, length_m_stddev_20km_radius, length_m_ave_20km_radius
 from 
   proj_global_sar.vessel_density_statistics_2017_2021
),

slope_table as (
select 
  lat_index, lon_index, -- lat and lon_index at 100th of a degree
  elevation_m, slope2_km, slope4_km, slope6_km, 
  slope8_km, slope10_km, slope12_km, slope14_km, 
  slope16_km, slope18_km, slope20_km
from 
  proj_global_sar.slope
),

average_depth_4km_radius as (
select 
  elevation_m_4km,
  elevation_m_stddev_4km,
  lat_index, -- 100th of a degree
  lon_index
from
  proj_global_sar.average_depth_4km_radius
),

average_depth_2km_radius as (
select 
  elevation_m_2km,
  elevation_m_stddev_2km,
  lat_index, -- 100th of a degree
  lon_index
from
  proj_global_sar.average_depth_2km_radius
),

distance_from_shore_table  as (
select 
  distance_from_shore_m,
  floor(lat*100) as lat_index,
  floor(lon*100) as lon_index,
from 
  `pipe_static.distance_from_shore`
),

distance_from_port_table as (
select 
  distance_from_port_m,
  floor(lat*100) as lat_index,
  floor(lon*100) as lon_index
from
  `pipe_static.distance_from_port_20201105`
),

currents_temperature as (
  select 
    lat_index,
    lon_index,
    avg_temp,
    avg_current,
    avg_daily_stddev_temp,
    stddev_current
  from 
    proj_global_sar.temperature_currents_100
),

chlorphyll as (
  select 
    lat_index,
    lon_index,
    chl
  from 
    proj_global_sar.chlorophyll_100
)


select 
  * except(lat_index, lon_index) 
from 
  detections_table a 
left join 
  vessel_info
using(ssvid)
left join
  eez_table b 
using(detect_id)
left join
  ais_vessel_density_2017_2021_multi_res c
on floor(detect_lat*100) = c.lat_index
and floor(detect_lon*100) = c.lon_index
left join 
  vessel_density_statistics_2017_2021 e
on floor(detect_lat*20) = e.lat_index
and floor(detect_lon*20) = e.lon_index
left join 
  slope_table f
on floor(detect_lat*100) = f.lat_index
and floor(detect_lon*100) = f.lon_index
left join 
  average_depth_2km_radius g
on floor(detect_lat*100) = g.lat_index
and floor(detect_lon*100) = g.lon_index
left join 
  average_depth_4km_radius h
on floor(detect_lat*100) = h.lat_index
and floor(detect_lon*100) = h.lon_index
left join
  distance_from_shore_table i
ON floor( detect_lat*100) = i.lat_index
AND floor(detect_lon*100) = i.lon_index
left join
  distance_from_port_table j
ON floor(detect_lat*100) = j.lat_index
AND floor(detect_lon*100) = j.lon_index
left join 
  currents_temperature k
ON floor(detect_lat*100) = k.lat_index
AND floor(detect_lon*100) = k.lon_index
left join
  chlorphyll l
ON floor(detect_lat*100) = l.lat_index
AND floor(detect_lon*100) = l.lon_index


'''

query_to_table(q, 'project-id.proj_global_sar.detections_w_features_v20220720')

# +
# v20220802 -- This is exactly the same as detections_w_features_v20220720, but it
# uses proj_global_sar.ais_vessel_density_2017_2021_multi_res_v2 instead of
# proj_global_sar.ais_vessel_density_2017_2021_multi_res



q = '''
with 

vessel_info as (
select 
  ssvid,
  best.best_vessel_class,
  on_fishing_list_best,
  on_fishing_list_nn
from 
  `project-id.gfw_research.vi_ssvid_v20220701` 
),


detections_table as (
select 
  detect_id,
  scene_id,
  detect_timestamp,
  detect_lat,
  detect_lon,
  presence,
  length_m,
  ssvid,
  score,
  confidence,
  scene_detections,
  bad_detections,
  scene_quality,
  repeats_100m_180days_back,
  repeats_100m_180days_center,
  repeats_100m_180days_forward,
  overpasses_year,
  overpasses_2017_2021
from 
  proj_global_sar.detections_w_overpasses_v20220805
),

eez_table as (
  select 
  detect_id, 
  array_agg(ifnull(MRGID,0)) as MRGID, 
  array_agg(ifnull(ISO_TER1,"")) as ISO_TER1, 
  array_agg(ifnull(TERRITORY1,"")) AS TERRITORY1
from detections_table
  CROSS JOIN
(select wkt, MRGID, ISO_TER1, TERRITORY1 from `project-id.minderoo.marine_regions_v11`)
WHERE
  ST_CONTAINS(SAFE.ST_GEOGFROMTEXT(wkt),ST_GEOGPOINT(detect_lon,detect_lat) )
group by 
  detect_id
),
--
-- vessel density at 100th of a degree bins, and averaged
ais_vessel_density_2017_2021_multi_res as (
select 
  lon_index, lat_index, -- at 100th of degree
  non_fishing, cargo_tanker, non_fishing_under50m
from 
  proj_global_sar.ais_vessel_density_2017_2021_multi_res_v2),
--
--
vessel_density_statistics_2017_2021 as (
select 
-- lat_index and lon_index are at 20th of a degree, so floor(lat/20)
  lat_index, lon_index, detections_km2_5km, length_m_stddev_5km, length_m_ave_5km, 
  detections_km2_5km_radius, length_m_stddev_5km_radius, length_m_ave_5km_radius, 
  detections_km2_10km_radius, length_m_stddev_10km_radius, length_m_ave_10km_radius,
  detections_km2_20km_radius, length_m_stddev_20km_radius, length_m_ave_20km_radius
 from 
   proj_global_sar.vessel_density_statistics_2017_2021
),

slope_table as (
select 
  lat_index, lon_index, -- lat and lon_index at 100th of a degree
  elevation_m, slope2_km, slope4_km, slope6_km, 
  slope8_km, slope10_km, slope12_km, slope14_km, 
  slope16_km, slope18_km, slope20_km
from 
  proj_global_sar.slope
),

average_depth_4km_radius as (
select 
  elevation_m_4km,
  elevation_m_stddev_4km,
  lat_index, -- 100th of a degree
  lon_index
from
  proj_global_sar.average_depth_4km_radius
),

average_depth_2km_radius as (
select 
  elevation_m_2km,
  elevation_m_stddev_2km,
  lat_index, -- 100th of a degree
  lon_index
from
  proj_global_sar.average_depth_2km_radius
),

distance_from_shore_table  as (
select 
  distance_from_shore_m,
  round(lat*100)+1 as lat_index,
  round(lon*100) as lon_index,
from 
  `pipe_static.distance_from_shore`
),

distance_from_port_table as (
select 
  distance_from_port_m,
  round(lat*100)-1 as lat_index,
  round(lon*100) as lon_index
from
  `pipe_static.distance_from_port_20201105`
),

currents_temperature as (
  select 
    lat_index,
    lon_index,
    avg_temp,
    avg_current,
    avg_daily_stddev_temp,
    stddev_current
  from 
    proj_global_sar.temperature_currents_100
),

chlorphyll as (
  select 
    lat_index,
    lon_index,
    chl
  from 
    proj_global_sar.chlorophyll_100
)


select 
  * except(lat_index, lon_index) 
from 
  detections_table a 
left join 
  vessel_info
using(ssvid)
left join
  eez_table b 
using(detect_id)
left join
  ais_vessel_density_2017_2021_multi_res c
on floor(detect_lat*100) = c.lat_index
and floor(detect_lon*100) = c.lon_index
left join 
  vessel_density_statistics_2017_2021 e
on floor(detect_lat*20) = e.lat_index
and floor(detect_lon*20) = e.lon_index
left join 
  slope_table f
on floor(detect_lat*100) = f.lat_index
and floor(detect_lon*100) = f.lon_index
left join 
  average_depth_2km_radius g
on floor(detect_lat*100) = g.lat_index
and floor(detect_lon*100) = g.lon_index
left join 
  average_depth_4km_radius h
on floor(detect_lat*100) = h.lat_index
and floor(detect_lon*100) = h.lon_index
left join
  distance_from_shore_table i
ON floor( detect_lat*100) = i.lat_index
AND floor(detect_lon*100) = i.lon_index
left join
  distance_from_port_table j
ON floor(detect_lat*100) = j.lat_index
AND floor(detect_lon*100) = j.lon_index
left join 
  currents_temperature k
ON floor(detect_lat*100) = k.lat_index
AND floor(detect_lon*100) = k.lon_index
left join
  chlorphyll l
ON floor(detect_lat*100) = l.lat_index
AND floor(detect_lon*100) = l.lon_index


'''

query_to_table(q, 'project-id.proj_global_sar.detections_w_features_v20220812')
# -

# # Features for v20220720 and v20220802
#
# Can we predict if it is a fishing vessel using these features? 
#
#  - elevation_m_2km: average depth, meters, of each cell, averaged over a circle 2km radius
#  - elevation_m_stddev_2km: standard deviation of depth, meters, over a circle 2km radius
#  - elevation_m_4km: over 4km radius
#  - elevation_m_stddev_4km: 4km radius
#  - non_fishing: hours of non-fishing vessels per km2 in AIS, 2017 to 2022
#  - cargo_tanker: hours of cargo or tanker hours per km2 in AIS, 2017 to 2022		
#  - non_fishing_under50m: hours of non-fishing vessels per km2 in AIS for vessels < 50m, 2017 to 2022
#  - elevation_m: depth in meters
#  - slope2_km: slope, calculated at 2km 
#  - slope4_km: slope, calculated over 4km
#  - slope6_km:
#  - slope8_km:
#  - slope10_km:
#  - slope12_km:
#  - slope14_km:
#  - slope16_km:
#  - slope18_km:
#  - slope20_km:
#  - length_m: inferred length from the neural net of the detection
#  - distance_from_shore_m: distance from shore in meters
#  - distance_from_port_m: distance from port in meters
#  - detections_km2_5km: density of detections averaged over a 20th of a degree
#  - length_m_stddev_5km: standard deviation of length of detections over a 20th of a degree
#  - length_m_ave_5km: average length of detections over a 20th of a degree
#  - detections_km2_5km_radius: denisty of detections over a circle with a 5km radius
#  - length_m_stddev_5km_radius: standard deviation of detections over a circle with a 5km radius
#  - length_m_ave_5km_radius: average length of detections over a circle with a 5km radius
#  - detections_km2_10km_radius	
#  - length_m_stddev_10km_radius	
#  - length_m_ave_10km_radius	
#  - detections_km2_20km_radius	
#  - length_m_stddev_20km_radius	
#  - length_m_ave_20km_radius
#  - chl: average chlorophyll concentration between 2012 and 2021
#  - avg_temp: average temperature across 2017 to 2019
#  - avg_current: average current across 2017 to 2019,
#  - avg_daily_stddev_temp: average of the daily standard deviation of temperature, 2017 to 2019
#  - stddev_current: standard deviation of current, 2017 to 2019
#



# # Layers for a neural net
#
# These are the queries needed to make the rasters for the neural net raster images

q = '''with sar_detection_table as (
select 
  lat_index,
  lon_index,
  detections_normalized,
  detections,
  avg_length_m,
  stddev_length_m
 from 
   proj_global_sar.vessel_density_statistics_100th_2017_2021 
),

ais_tale as (
select 
  lat_index,
  lon_index,
  non_fishing as non_fishing_ais_hours,
  cargo_tanker as cargo_tanker_ais_hours
 from proj_global_sar.ais_vessel_density_2017_2021_multi_res_v2 
 ),

pipe_static_metrics_table as (
 select 
 lat_index,
 lon_index,
 distance_from_shore_m,
 distance_from_port_m,
 elevation_m
  from proj_global_sar.bathymetry_port_shore_metrics_100th )

  '''

# +
# currents, temperature:

q = '''  
  select 
    lat_index,
    lon_index,
    avg_temp,
    avg_current,
    avg_daily_stddev_temp,
    stddev_current
  from 
    proj_global_sar.temperature_currents_100  
 join
  proj_global_sar.bathymetry_port_shore_metrics_100th_clipped 
 using(lat_index, lon_index)
 where elevation_m < 50'''
# -

q = '''  
  select 
    lat_index,
    lon_index,
    chld
  from 
    proj_global_sar.chlorophyll_100  
 join
  proj_global_sar.bathymetry_port_shore_metrics_100th_clipped 
 using(lat_index, lon_index)
 where elevation_m < 50'''
