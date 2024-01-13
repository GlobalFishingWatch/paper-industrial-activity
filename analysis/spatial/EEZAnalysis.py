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

# # EEZ Analysis
#
# This produces a table with information about each EEZ, and the fraction of total activity that is under 200 m in depth. It also provides the estimates of the number of vessels active in the eastern and western Noth Korean EEZs.

# +
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import pyperclip
# use the standard for eliminating ice locations.

import sys
sys.path.append('../utils')
from eliminate_ice_string import *
from proj_id import project_id
eliminated_locations = eliminate_ice_string()
# -

# # Update the EEZ info table

# Download marine reagions EEZ layer version 11 from https://www.marineregions.org/
# make sure data is in the right place
# !ls ../data/World_EEZ_v11_20191118/

# +
### upload EEZ info table to bigquery. uncomment to redo

# # https://www.marineregions.org/eezattribute.php
# import fiona

# attributes = []
# with fiona.open("../../data/World_EEZ_v11_20191118/eez_v11.shp", "r") as source:
#     print( len(source))
#     for s in source:
#         attributes.append(s['properties'])
        
# df = pd.DataFrame(attributes).sort_values("MRGID_TER1")
# df['MRGID'] = df.MRGID.apply(lambda x: str(int(x)))

# # upload to bigquery
# df.to_gbq('proj_global_sar.eez_info_v11', if_exists='replace')
# -

df_eez = pd.read_gbq("select * from proj_global_sar.eez_info_v11")


df_eez['MRGID'] = df_eez.MRGID.apply(lambda x: str(int(x)))
df_eez = df_eez.set_index('MRGID')

df_eez.head()



# ## For each eez, get:
#  - area analyzed
#  - area with 200m depth
#  - how much of 200m depth has a deteciton once every 10 days and every 100
#  - time series
#  - map all time
#  - map by year
#  - detections by category
#  - detections by category by year

# ## Area Analyzed and Area wihtin 200 depth

# make the ice string elmination work with different variables
e = eliminated_locations.replace("detect_lon","(lon_index/200 + 1/400)")
e = e.replace("detect_lat","(lat_index/200 + 1/400)")

# +
q = f'''with depth_table as 
(select 
elevation_m,
lon,
lat,
format("lon:%+07.2f_lat:%+07.2f", round(lon/0.01)*0.01, round(lat/0.01)*0.01) as gridcode
from pipe_static.bathymetry
where elevation_m <= 0
),

footprints as
(SELECT 
  lon_index, 
  lat_index,
  format("lon:%+07.2f_lat:%+07.2f", 
  round((lon_index/200+1/400)/0.01)*0.01, 
  round((lat_index/200+1/400)/0.01)*0.01) as gridcode,
  sum(overpasses) overpasses, 
FROM 
  `proj_global_sar.overpasses_200_by_year_filtered_v20220508`
where 
  year between 2017 and 2021 
  {e}
group by lat_index, lon_index
having overpasses >= 30
),

regions as 
(SELECT eez , 
gridcode FROM `pipe_static.spatial_measures_20201105` 
cross join unnest( regions.eez) as eez

),


combined as (
select 
  lat_index,
  lon_index,
  lat,
  lon,
  elevation_m,
  eez
from 
  footprints
left join
  regions
using(gridcode)
left join
  depth_table
using(gridcode)
),

area_imaged as 

(select 
eez,
sum(111*111/200/200*cos(lat*3.1416/180)) area_km2_imaged,
sum(if(elevation_m >= -200, 
    111*111/200/200*cos(lat*3.1416/180),0)) area_km2_under200_imaged
from combined
group by eez
),

total_area as

(
select
ifnull(eez,'highseas') eez,
sum(111*111/100/100*cos(lat*3.1416/180)) area_km2,
sum(if(elevation_m >= -200, 
    111*111/100/100*cos(lat*3.1416/180),0)) area_km2_under200
from
depth_table
left join
regions
using(gridcode)
group by eez
)

select * from 
total_area
left join
area_imaged
using(eez)

'''

df_area = pd.read_gbq(q)
# -

df_area.head()

df_area.area_km2.sum()/1e6

# fraction of study area that is under 200 meters
df_area.area_km2_under200_imaged.sum()/df_area.area_km2_imaged.sum()

# fraction of ocean that is under 200 meters
df_area.area_km2_under200.sum()/df_area.area_km2.sum()

# +
q = f'''

with
predictions_table as
(


  select 
    detect_id, 
    avg(fishing_33) fishing_score_low,
    avg( fishing_50) fishing_score, 
    avg(fishing_66) fishing_score_high
  from
    (select detect_id, fishing_33, fishing_50, fishing_66 from 
    `proj_sentinel1_v20210924.fishing_pred_even_v5*`
    union all
    select detect_id, fishing_33, fishing_50, fishing_66 from 
    `proj_sentinel1_v20210924.fishing_pred_odd_v5*`
    )
  group by 
    detect_id

),

depth_table as (
select 
  elevation_m, detect_id 
from 
  `proj_global_sar.detections_w_features_v20220812`
),

vessel_info as (
select
  ssvid,
  if(on_fishing_list_known is not null, on_fishing_list_known, on_fishing_list_nn) as on_fishing_list
from
   `gfw_research.vi_ssvid_v20221001`
  -- don't do anything with identity spoofing vessels!
  where activity.overlap_hours_multinames < 24
),

regions as (
  SELECT 
    eez , 
    gridcode FROM `pipe_static.spatial_measures_20201105` 
cross join unnest( regions.eez) as eez

),

detections_table as

(
  select
  floor(detect_lat*5) as lat_index_d,
  floor(detect_lon*5) as lon_index_d,
  format("lon:%+07.2f_lat:%+07.2f", 
          round((detect_lon)/0.01)*0.01,
          round((detect_lat)/0.01)*0.01) as gridcode,
  detect_lat,
  detect_lon,
  detect_id,
  ssvid_mult_recall_length as ssvid,
  score_mult_recall_length as score,
  confidence_mult_recall_length,
  overpasses_2017_2021,
  date_24,
  7.4e-6 as dd_perkm2
  from
  `proj_global_sar.detections_w_overpasses_v20230803`

  where
  -- the following is very restrictive on repeated objects
  repeats_100m_180days_forward < 3 and
  repeats_100m_180days_back < 3 and
  repeats_100m_180days_center < 3
  -- get rid of scenes where more than half the detections
  -- are likely noise
  and (scene_detections <=5 or scene_quality > .5)
  and extract(date from detect_timestamp)
     between "2017-01-01" and "2021-12-31"
  -- at least 10 overpasses
  and overpasses_2017_2021 > 10
  -- our cutoff for noise -- this could be adjusted down, but makes
  -- very little difference between .5 and .7
  and presence > .7
  and not close_to_infra
  and not potential_ambiguity
  {eliminated_locations}
 and not in_road_doppler
  
  ),




final_table as (
select
  detect_id,
  lat_index_d,
  lon_index_d,
  date_24,
  detect_lat,
  detect_lon,
  overpasses_2017_2021,
  ifnull(eez,'highseas') eez,
  fishing_score,
  fishing_score_low,
  fishing_score_high,
  case when score > dd_perkm2 and on_fishing_list  then "matched_fishing"
   when score > dd_perkm2 and not on_fishing_list  then "matched_nonfishing"
   when score > dd_perkm2 and  on_fishing_list is null  then "matched_unknown"
   when score < dd_perkm2 then "unmatched" end as matched_category,
from
  detections_table a
left join
  vessel_info
using(ssvid)
left join
  predictions_table
using(detect_id)
left join
regions
using(gridcode)

),

detections_grid as 
(
select
  eez,
  floor(detect_lon*10) as lon_index,
  floor(detect_lat*10) as lat_index,
  sum(1/overpasses_2017_2021) detections,
  
  sum(if(matched_category = "matched_fishing",1/overpasses_2017_2021,0)) + 
  sum(if(matched_category = "matched_unknown", fishing_score/overpasses_2017_2021,0)) matched_fishing,
  
  sum(if(matched_category = "matched_nonfishing",1/overpasses_2017_2021,0)) + 
  sum(if(matched_category = "matched_unknown", (1-fishing_score)/overpasses_2017_2021,0)) matched_nonfishing,
  
  sum(if(matched_category = "unmatched", (fishing_score)/overpasses_2017_2021,0)) dark_fishing,
  
  sum(if(matched_category = "unmatched", (1-fishing_score)/overpasses_2017_2021,0)) dark_nonfishing
from
  final_table
join
  depth_table
using(detect_id)
group by lat_index, lon_index, eez
),

avg_bathymetry as 
(select 
floor(lat*10) lat_index,
floor(lon*10) lon_index,
avg(elevation_m) elevation_m
from 
`pipe_static.bathymetry`
where {eliminated_locations[3:].replace("detect_lon","lon").replace("detect_lat","lat")}
group by lat_index, lon_index
having elevation_m <=0
),

overpasses as (
SELECT 
lat_index,
lon_index,
sum(overpasses) as overpasses
 FROM `proj_sentinel1_v20210924.detect_foot_raster_10` 
 WHERE DATE(_PARTITIONTIME) between "2017-01-01" and "2021-12-31"
 group by lat_index, lon_index
 having overpasses >= 30),

overpasses_w_depth as (
 select lat_index, lon_index, elevation_m 
 from overpasses
 join
 avg_bathymetry
 using(lat_index, lon_index) 
)

select lat_index, lon_index, eez,
ifnull(matched_nonfishing + dark_nonfishing ,0) nonfishing_detections, 
ifnull(matched_fishing + dark_fishing,0) fishing_detections,
ifnull(matched_fishing,0) matched_fishing,
ifnull(matched_nonfishing,0) matched_nonfishing,
ifnull(dark_fishing,0) dark_fishing,
ifnull(dark_nonfishing,0) dark_nonfishing,
-elevation_m as depth
from overpasses_w_depth
left join
detections_grid
using(lat_index, lon_index)

'''

df_d = pd.read_gbq(q)
# pyperclip.copy(q)
# -


df_d.head()

# check the fraction of fishing that is dark
# this is also calculated in another notebook
df_d.dark_fishing.sum()/df_d.fishing_detections.sum()


# +
def get_area(row):
    return 111*111/10/10*np.cos(row.lat_index/10 / 180 * np.pi)**2

df_d['area'] = df_d.apply(get_area, axis=1)
df_d.head()
# -

# area in the high seas, in million square km
df_d[df_d.eez.isna()].area.sum()/1e6



# +
d = df_d[(df_d.depth < 200)&(df_d.fishing_detections>0)].groupby('eez').sum()
d['area_fish_under_200'] = d['area']
d['fishing_under_200'] = d['fishing_detections']
d_area_fish_200m = d[['area_fish_under_200','fishing_under_200']]


d = df_d[(df_d.depth < 200)&(df_d.nonfishing_detections>0)].groupby('eez').sum()
d['area_nonfish_under_200'] = d['area']
d['nonfishing_under_200'] = d['nonfishing_detections']
d_area_nonfish_200m = d[['area_nonfish_under_200','nonfishing_under_200']]


d = df_d[(df_d.depth < 200)&(df_d.fishing_detections>.1)].groupby('eez').sum()
d['area_fish_under_200m_10day'] = d['area']
d['fishing_under_200m_10day'] = d['fishing_detections']
d_area_fish_200m_10day = d[['area_fish_under_200m_10day','fishing_under_200m_10day']]

d = df_d[(df_d.fishing_detections>0)].groupby('eez').sum()
d['area_fish'] = d['area']
d['fishing'] = d['fishing_detections']
d_area_fish = d[['area_fish','fishing']]

d = df_d[(df_d.nonfishing_detections>0)].groupby('eez').sum()
d['area_nonfish'] = d['area']
d['nonfishing'] = d['nonfishing_detections']
d_area_nonfish = d[['area_nonfish','nonfishing']]

d = df_d[(df_d.fishing_detections>.1)].groupby('eez').sum()
d['area_fish_10day'] = d['area']
d['fishing_10day'] = d['fishing_detections']
d_area_fish_10day = d[['fishing_10day','area_fish_10day']]

d = df_d.groupby('eez').sum()
d_fishnonfish = d[['matched_fishing','matched_nonfishing','dark_fishing','dark_nonfishing']]
# -
d_area = df_area.set_index('eez').fillna(0)
d_area.head()



d = df_eez.join(d_area, how='outer')
d = d.join(d_area_fish_200m, how='outer')
d = d.join(d_area_fish_200m_10day, how='outer')
d = d.join(d_area_fish, how='outer')
d = d.join(d_area_fish_10day, how='outer')
d = d.join(d_fishnonfish,how='outer')
d = d.join(d_area_nonfish_200m,how='outer')
d = d.join(d_area_nonfish,how='outer')
d = d.fillna(0)
df = d

df.head()



# +
import pycountry
from pycountry_convert import (
    country_alpha2_to_continent_code,
    country_name_to_country_alpha2,
)

continents = {
    "NA": "North America",
    "SA": "South America",
    "AS": "Asia",
    "OC": "Australia",
    "AF": "Africa",
    "EU": "Europe",
}


def get_continent(x):
    try:
        return continents[
            country_alpha2_to_continent_code(pycountry.countries.get(alpha_3=x).alpha_2)
        ]
    except:
        "None"


# -

df['continent'] = df.ISO_TER1.apply(get_continent)

df.to_csv("../data/activity_by_eez.csv",index=False)

# # calcluate some statistics on area under 200 meters depth

# what fraction of the global area of the ocean under 200 meters is in our study area?
# note that we eliminated enormous amounts of shallow water in the far north
df.area_km2_under200_imaged.sum()/df.area_km2_under200.sum()

# what fraction of the imaged area under 200m had fishing activity?
df.area_fish_under_200.sum()/df.area_km2_under200_imaged.sum()

# what fraction of the area imaged had fishing activity?
df.area_fish.sum()/df.area_km2_imaged.sum()

# What fraction of the area under 200m had a vessel at least once every 10 days?
# That is, a density of 0.1 fishing vessels per 10th degree square
df.area_fish_under_200m_10day.sum()/df.area_km2_under200_imaged.sum()

# What is the total number of fishing vessels per km2 in areas under 200m
# by continent?
d = df[['fishing_under_200','area_km2_under200_imaged','continent']].groupby("continent").sum()
d['concentration_200'] = d.fishing_under_200/d.area_km2_under200_imaged
d['concentration_200']

# What is the area under 200m depth by continent that was imaged?
d.area_km2_under200_imaged

# # How much fishig is in European countries on the Mediterranean as compared to African countries on the Mediterranean?

med_eu = ['FRA','ESP','ITA','GRC','MCO','HRV','ALB','MNE']
med_afr = ['MAR','DZA','TUN','LBY','EGY']


# Europe
df[df.ISO_TER1.isin(med_eu)].fishing.sum()

# Africa
df[df.ISO_TER1.isin(med_afr)].fishing.sum()

# Just AIS, Europe
df[df.ISO_TER1.isin(med_eu)].matched_fishing.sum()

# Just AIS, Africa
df[df.ISO_TER1.isin(med_afr)].matched_fishing.sum()

# ratio of AIS fishing in med EU versus afirca 
df[df.ISO_TER1.isin(med_eu)].matched_fishing.sum() / df[df.ISO_TER1.isin(med_afr)].matched_fishing.sum()

# +
# # ratio of total fishing 
# df[df.ISO_TER1.isin(med_afr)].fishing.sum()/df[df.ISO_TER1.isin(med_eu)].matched_fishing.sum()
# -


# What fraction of the ocean area is under 200m? 
df.area_km2_under200.sum()/df.area_km2.sum()

# what fraciton of fishing is in asia?
df[df.continent=='Asia'].fishing.sum()/df.fishing.sum()



# What fraction of fishing is under 200 meters in depth?
df.fishing_under_200.sum()/df.fishing.sum()

# What fraction of nonfishing is under 200 meters in depth?
df.nonfishing_under_200.sum()/df.nonfishing.sum()

# # Analysis for North Korea by Year
#
# Rebuild the same dataframes as above, but divide the North Korean EEZ into east and west.
# Note that these are the same queries with very small modificaitons to divide North Korea into east and west.

# +
q = f'''with depth_table as 
(select 
elevation_m,
lon,
lat,
format("lon:%+07.2f_lat:%+07.2f", round(lon/0.01)*0.01, round(lat/0.01)*0.01) as gridcode
from pipe_static.bathymetry
where elevation_m <= 0
),

footprints as
(SELECT 
  lon_index, 
  lat_index,
  format("lon:%+07.2f_lat:%+07.2f", 
  round((lon_index/200+1/400)/0.01)*0.01, 
  round((lat_index/200+1/400)/0.01)*0.01) as gridcode,
  sum(overpasses) overpasses, 
FROM 
  `{project_id}.proj_global_sar.overpasses_200_by_year_filtered_v20220508`
where 
  year between 2017 and 2021 
  {e}
group by lat_index, lon_index
having overpasses >= 30
),

regions as 
( select * from 
  (SELECT eez , 
    gridcode FROM `{project_id}.pipe_static.spatial_measures_20201105` 
    cross join unnest( regions.eez) as eez)
  where eez = '8328' -- only north korea
),

combined as
(select 
lat_index,
lon_index,
lat,
lon,
elevation_m,
eez
from footprints
left join
regions
using(gridcode)
left join
depth_table
using(gridcode)),

area_imaged as 
(select 
eez,
if(lon < 127, "west","east") eez_region,
sum(111*111/200/200*cos(lat*3.1416/180)) area_km2_imaged,
sum(if(elevation_m >= -200, 
    111*111/200/200*cos(lat*3.1416/180),0)) area_km2_under200_imaged
from combined
group by eez, eez_region
),

total_area as
(
select
ifnull(eez,'highseas') eez,
if(lon < 127, "west","east") eez_region,
sum(111*111/100/100*cos(lat*3.1416/180)) area_km2,
sum(if(elevation_m >= -200, 
    111*111/100/100*cos(lat*3.1416/180),0)) area_km2_under200
from
depth_table
left join
regions
using(gridcode)
group by eez, eez_region
)

select * from 
total_area
left join
area_imaged
using(eez, eez_region)
where eez = '8328'

'''

df_area_nk = pd.read_gbq(q)
# -





# +
q = f'''

with
predictions_table as
(


  select 
    detect_id, 
    avg(fishing_33) fishing_score_low,
    avg( fishing_50) fishing_score, 
    avg(fishing_66) fishing_score_high
  from
    (select detect_id, fishing_33, fishing_50, fishing_66 from 
    `proj_sentinel1_v20210924.fishing_pred_even_v5*`
    union all
    select detect_id, fishing_33, fishing_50, fishing_66 from 
    `proj_sentinel1_v20210924.fishing_pred_odd_v5*`
    )
  group by 
    detect_id

),

depth_table as (
select 
  elevation_m, detect_id 
from 
  `proj_global_sar.detections_w_features_v20220812`
),

vessel_info as (
select
  ssvid,
  if(on_fishing_list_known is not null, on_fishing_list_known, on_fishing_list_nn) as on_fishing_list
from
   `gfw_research.vi_ssvid_v20221001`
  -- don't do anything with identity spoofing vessels!
  where activity.overlap_hours_multinames < 24
),

regions as (
 select * from  (SELECT 
    eez, 
    gridcode FROM `pipe_static.spatial_measures_20201105` 
cross join unnest( regions.eez) as eez
)  where eez = '8328' -- only north korea

),

detections_table as

(
  select
  floor(detect_lat*5) as lat_index_d,
  floor(detect_lon*5) as lon_index_d,
  format("lon:%+07.2f_lat:%+07.2f", 
          round((detect_lon)/0.01)*0.01,
          round((detect_lat)/0.01)*0.01) as gridcode,
  detect_lat,
  detect_lon,
  extract(year from detect_timestamp) year,
  if(detect_lon < 127, "west","east") eez_region,
  detect_id,
  ssvid,
  score_mult_recall_length as score,
  confidence,
  overpasses_year,
  date_24,
  7.4e-6 as dd_perkm2
  from
  `proj_global_sar.detections_w_overpasses_v20230803`

  where
  -- the following is very restrictive on repeated objects
  repeats_100m_180days_forward < 3 and
  repeats_100m_180days_back < 3 and
  repeats_100m_180days_center < 3
  -- get rid of scenes where more than half the detections
  -- are likely noise
  and (scene_detections <=5 or scene_quality > .5)
  and extract(date from detect_timestamp)
     between "2017-01-01" and "2021-12-31"
  -- at least 10 overpasses
  and overpasses_2017_2021 > 10
  -- our cutoff for noise -- this could be adjusted down, but makes
  -- very little difference between .5 and .7
  and presence > .7
  and not close_to_infra  
  and not potential_ambiguity
 {eliminated_locations}
 -- eliminate points near roads that could be trucks
 and not in_road_doppler
  
  ),




final_table as (
select
  detect_id,
  lat_index_d,
  lon_index_d,
  date_24,
  detect_lat,
  detect_lon,
  year,
  eez_region,
  overpasses_year,
  ifnull(eez,'highseas') eez,
  fishing_score,
  fishing_score_low,
  fishing_score_high,
  case when score > dd_perkm2 and on_fishing_list  then "matched_fishing"
   when score > dd_perkm2 and not on_fishing_list  then "matched_nonfishing"
   when score > dd_perkm2 and  on_fishing_list is null  then "matched_unknown"
   when score < dd_perkm2 then "unmatched" end as matched_category,
   
from
  detections_table a
left join
  vessel_info
using(ssvid)
left join
  predictions_table
using(detect_id)
join
regions
using(gridcode)

),

detections_grid as 
(
select
  eez,
  eez_region,
  year,
  floor(detect_lon*10) as lon_index,
  floor(detect_lat*10) as lat_index,
  sum(1/overpasses_year) detections,
  
  sum(if(matched_category = "matched_fishing",1/overpasses_year,0)) + 
  sum(if(matched_category = "matched_unknown", fishing_score/overpasses_year,0)) matched_fishing,
  
  sum(if(matched_category = "matched_nonfishing",1/overpasses_year,0)) + 
  sum(if(matched_category = "matched_unknown", (1-fishing_score)/overpasses_year,0)) matched_nonfishing,
  
  sum(if(matched_category = "unmatched", (fishing_score)/overpasses_year,0)) dark_fishing,
  
  sum(if(matched_category = "unmatched", (1-fishing_score)/overpasses_year,0)) dark_nonfishing,
  
from
  final_table
join
  depth_table
using(detect_id)
group by lat_index, lon_index, eez, eez_region, year
),

avg_bathymetry as 
(select 
floor(lat*10) lat_index,
floor(lon*10) lon_index,
avg(elevation_m) elevation_m
from 
`pipe_static.bathymetry`
where {eliminated_locations[3:].replace("detect_lon","lon").replace("detect_lat","lat")}
group by lat_index, lon_index
having elevation_m <=0
),

overpasses as (
SELECT 
lat_index,
lon_index,
sum(overpasses) as overpasses
 FROM `proj_sentinel1_v20210924.detect_foot_raster_10` 
 WHERE DATE(_PARTITIONTIME) between "2017-01-01" and "2021-12-31"
 group by lat_index, lon_index
 having overpasses >= 30),

overpasses_w_depth as (
 select lat_index, lon_index, elevation_m 
 from overpasses
 join
 avg_bathymetry
 using(lat_index, lon_index) 
)

select 
year,
lat_index, 
lon_index, 
eez, 
eez_region,
ifnull(matched_nonfishing + dark_nonfishing ,0) nonfishing_detections, 
ifnull(matched_fishing + dark_fishing,0) fishing_detections,
ifnull(matched_fishing,0) matched_fishing,
ifnull(matched_nonfishing,0) matched_nonfishing,
ifnull(dark_fishing,0) dark_fishing,
ifnull(dark_nonfishing,0) dark_nonfishing,
-elevation_m as depth
from overpasses_w_depth
left join
detections_grid
using(lat_index, lon_index)
where eez = '8328'

'''

df_d_nk = pd.read_gbq(q)
# pyperclip.copy(q)
# -

df_d_nk.head()

d = df_d_nk.groupby(["eez_region",'year']).sum().reset_index()
d.sort_values(['eez_region','year'])



# how many dark vessels per 1000 km2 in 2017-2019?
dark_fishing = d[(d.eez_region=='west')&(d.year.isin([2017,2018,2019]))].dark_fishing.mean()
area = df_area_nk[df_area_nk.eez_region=='west']['area_km2_imaged'].values[0]
dark_fishing/area*1000

# how many dark vessels per 1000 km2 in 2020-2021?
dark_fishing = d[(d.eez_region=='west')&(d.year.isin([2020,2021]))].dark_fishing.mean()
area = df_area_nk[df_area_nk.eez_region=='west']['area_km2_imaged'].values[0]
dark_fishing/area*1000



df.columns


