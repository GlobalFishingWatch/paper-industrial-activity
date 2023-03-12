# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Plot distribution of vessels and infrastructure as a function of distance from shore

import pandas as pd
import matplotlib.pyplot as plt
import proplot as pplt
import numpy as np

# +
q = """with

predictions_table as
(
  select 
    detect_id, 
    avg(fishing_33) fishing_score_low,
    avg( fishing_50) fishing_score, 
    avg(fishing_66) fishing_score_high
  from
    (select detect_id, fishing_33, fishing_50, fishing_66 from 
    `world-fishing-827.proj_sentinel1_v20210924.fishing_pred_even_v5*`
    union all
    select detect_id, fishing_33, fishing_50, fishing_66 from 
    `world-fishing-827.proj_sentinel1_v20210924.fishing_pred_odd_v5*`
    )
  group by 
    detect_id
),

vessel_info as (
select 
  ssvid, 
  if(on_fishing_list_known is not null, on_fishing_list_known, on_fishing_list_nn) as on_fishing_list
from 
  `gfw_research.vi_ssvid_v20220701`
  -- don't do anything with identity spoofing vessels!
  where activity.overlap_hours_multinames < 24
),

detections_table as 

(
  select 
  floor(detect_lat*5) as lat_index,
  floor(detect_lon*5) as lon_index,
  date_24,
  periods24_with_overpass,
  periods24_with_overpass_june2017_dec2021,
  overpasses_24day,
  detect_lat,
  detect_lon,
  detect_id,
  ssvid,
  eez_iso3,
  score,
  confidence,
  length_m,
  overpasses_2017_2021,  
  from
  proj_global_sar.detections_w_overpasses_v20220929
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
  and not
      (
       ( detect_lon > -120.0 and detect_lon < -46.8 and detect_lat> 50.5 and detect_lat < 80.5 ) or 
   ( detect_lon > -120.0 and detect_lon < -46.8 and detect_lat> 50.5 and detect_lat < 80.5 ) or 
   (
          (detect_lon > 39.5 or detect_lon < -46.8 ) and
          detect_lat> 65.0 and detect_lat < 90 )
      or   ( detect_lon > 15.95 and detect_lon < 36.23 and detect_lat> 59.02 and detect_lat < 66.57 ) or 
   ( detect_lon > -173.7 and detect_lon < -158.4 and detect_lat> 62.0 and detect_lat < 66.8 ) or 
   (
          (detect_lon > 130.5 or detect_lon < -174.2 ) and
          detect_lat> 50.6 and detect_lat < 67.8 )
      or   ( detect_lon > 3.5 and detect_lon < 31.9 and detect_lat> 78.1 and detect_lat < 85.0 ) or 
   ( detect_lon > -179.8 and detect_lon < -156.5 and detect_lat> 57.4 and detect_lat < 62.2 ) or 
   ( detect_lon > -44.82 and detect_lon < -29.05 and detect_lat> -57.93 and detect_lat < -50.61 ) or 
   ( detect_lon > 31.4 and detect_lon < 60.3 and detect_lat> 61.4 and detect_lat < 73.1 ) or 
   ( detect_lon > -27.61 and detect_lon < -19.47 and detect_lat> 68 and detect_lat < 68.62 ) )

  ),


-- get the density of dark vessels at a 5th of a degree square

dark_vessel_density as (

select 
lat_index, 
lon_index,
 sum(1/overpasses_2017_2021) / pow(111/5 * cos((lat_index+.5)/5*3.1416/180),2) dd_perkm2 -- dark detects per km2
from detections_table
where score < 1e-3 -- this is very insensitive to the exact cut off
group by lat_index, lon_index
),



final_table as (
select
  detect_id,
  lat_index,
  lon_index,
  overpasses_2017_2021 as overpasses,
  date_24,
  eez_iso3,
  fishing_score,
  fishing_score_low,
  fishing_score_high,
  distance_from_shore_m,
  length_m,
  case when score > dd_perkm2 and on_fishing_list and confidence > .5 then "matched_fishing"
   when score > dd_perkm2 and not on_fishing_list and confidence > .5 then "matched_nonfishing"
   when score > dd_perkm2 and ( on_fishing_list is null or confidence < .5) then "matched_unknown"
   when score < dd_perkm2 then "unmatched" end as matched_category   
from
  detections_table a
left join
  vessel_info
using(ssvid)
left join
  predictions_table
using(detect_id)
left join
 dark_vessel_density
 using(lat_index, lon_index) 
left join
 `world-fishing-827.pipe_static.distance_from_shore`
on floor(detect_lat*100) + 1 = round(lat*100)
and floor(detect_lon*100) = round(lon*100)
where
overpasses_2017_2021 > 30

)

select 
  floor(distance_from_shore_m/1000) distance_from_shore_km,
  sum(if( matched_category = 'matched_fishing', 1/overpasses, 0)) matched_fishing,
  sum(if( matched_category = 'matched_nonfishing', 1/overpasses, 0)) matched_nonfishing,
  sum(if( matched_category = 'matched_unknown', 1/overpasses, 0)) matched_unknown,
  sum(if( matched_category = 'matched_unknown',
               fishing_score/overpasses, 0)) matched_unknown_likelyfish,
  sum(if( matched_category = 'matched_unknown',
               (1-fishing_score)/overpasses, 0)) matched_unknown_likelynonfish,               
  sum(if( matched_category = 'unmatched', fishing_score/overpasses, 0)) unmatched_fishing,
  sum(if( matched_category = 'unmatched', (1-fishing_score)/overpasses, 0)) unmatched_nonfishing,
  sum(1/overpasses) detections,
from 
  final_table
group by distance_from_shore_km
order by distance_from_shore_km
"""

df = pd.read_gbq(q)
# -

df["AIS fishing"] = df.matched_fishing + df.matched_unknown_likelyfish
df["AIS non-fishing"] = (
    df.matched_nonfishing + df.matched_unknown_likelynonfish
)
df["dark fishing"] = df.unmatched_fishing
df["dark non-fishing"] = df.unmatched_nonfishing
df["tot_fishing"] = df["dark fishing"] + df["AIS fishing"]
df["tot_nonfishing"] = df["dark non-fishing"] + df["AIS non-fishing"]

plt.figure(figsize=(10, 4))
plt.plot(df.distance_from_shore_km, df.tot_fishing, label="fishing")
plt.plot(df.distance_from_shore_km, df.tot_nonfishing, label="non-fishing")
plt.xlim(2, 200)
plt.legend()
plt.title("vessels per km")
plt.xlabel("distance km")

# ## Get Fixed Infrastructure

# +
## This is relatively quick, and doesn't use the polygons
## or the elimination string

## It also doesn't use the updated classifications for infrastructure or eliminate lake maracaibo

q = """with 

detections_labeled as (
SELECT 
  detect_id,
  detect_lon as lon,
  detect_lat as lat,
  extract( date from midpoint) detect_date,
  monthly_label as label,
  floor(distance_from_shore_m/1000) distance_from_shore_km
FROM 
  `world-fishing-827.proj_global_sar.infrastructure_repeat_cat_6m_v20220805` 
join
  `world-fishing-827.proj_global_sar.composite_ids_labeled_v20220708`
  using(detect_id)
left join
 `world-fishing-827.pipe_static.distance_from_shore`
on floor(detect_lat*100) + 1 = round(lat*100)
and floor(detect_lon*100) = round(lon*100)  
 where label in ('oil', 'other', 'wind')
 and midpoint = "2021-10-01"
-- {elimination_string}
)


select

distance_from_shore_km,
sum(if(label='oil',1,0)) oil,
sum(if(label='other',1,0)) other,
sum(if(label='wind',1,0)) wind,
from detections_labeled
group by distance_from_shore_km
order by distance_from_shore_km"""

df2 = pd.read_gbq(q)
# -

plt.figure(figsize=(10, 4))
plt.plot(df2.distance_from_shore_km, df2.oil, label="oil")
plt.plot(df2.distance_from_shore_km, df2.wind, label="wind")
plt.plot(df2.distance_from_shore_km, df2.other, label="other")
plt.legend()
plt.xlim(2, 100)
plt.title("fixed infrastructure")
plt.xlabel("distance from shore, km")
plt.ylabel("objects per km on average day")

# +
plt.figure(figsize=(10, 4))
plt.plot(df2.distance_from_shore_km, df2.oil, label="oil")
plt.plot(df2.distance_from_shore_km, df2.wind, label="wind")
# plt.plot(df2.distance_from_shore_km, df2.other, label = 'other')

plt.plot(df.distance_from_shore_km, df.tot_fishing, label="fishing")
plt.plot(df.distance_from_shore_km, df.tot_nonfishing, label="non-fishing")

plt.legend()
plt.xlim(2, 100)
plt.title("Objects and Vessels")
plt.xlabel("distance from shore, km")
plt.ylabel("objects or vessels per km on average day")

df.to_feather("data/distance_from_shore_vessel.feather")
df2.to_feather("data/distance_from_shore_infra.feather")

# +
plt.figure(figsize=(10, 4))
plt.plot(df2.distance_from_shore_km, df2.oil, label="oil")
plt.plot(df2.distance_from_shore_km, df2.wind, label="wind")
plt.plot(df2.distance_from_shore_km, df2.other, label="other")
plt.plot(df2.distance_from_shore_km, df2.other + df2.oil, label="other + oil")

# plt.plot(df2.distance_from_shore_km, df2.other, label = 'other')

plt.plot(df.distance_from_shore_km, df.tot_fishing, label="fishing")
plt.plot(df.distance_from_shore_km, df.tot_nonfishing, label="non-fishing")

plt.legend()
plt.xlim(2, 100)
plt.title("Objects and Vessels")
plt.xlabel("distance from shore, km")
plt.ylabel("objects or vessels per km on average day")
# -
