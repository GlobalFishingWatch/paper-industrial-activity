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

# +

# Get the query templates
import sys

sys.path.append("../utils")
from vessel_queries import *

# project id
from proj_id import project_id

# ice string elimination
from eliminate_ice_string import eliminate_ice_string

eliminated_locations = eliminate_ice_string()

# biguquery helper functions
from bigquery_helper_functions import query_to_table, update_table_description

import pandas as pd
from google.cloud import bigquery
from google.cloud.exceptions import NotFound

client: bigquery.Client = bigquery.Client(project=project_id)



# +
vessel_info_table = "gfw_research.vi_ssvid_v20230701"


predictions_table = """

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
"""
# -


query = f'''with 

detection_source_table as 
(select * from `proj_global_sar.detections_w_overpasses_v20230803`),


all_dates as
-- a table of all date_24s in our time period
(SELECT 
  distinct(date_24) date 
FROM 
  detection_source_table
where 
  date_24 between "2017-01-01" and "2021-12-31"
),

cells_worth_considering as (
select 
  distinct
  floor(detect_lat*200) as lat_index,
  floor(detect_lon*200) as lon_index
from 
  detection_source_table
where 	
  periods24_with_overpass >=70
  and date_24 between "2017-01-01" and "2021-12-31" 
),


expensive_table as 

(SELECT 
  lat_index,
  lon_index,
  overpasses,
  date
FROM 
  `proj_global_sar.overpasses_200_every24days_v20220805` 
),


cells_with_dates as (
select 
  date,
  lat_index,
  lon_index
from 
  cells_worth_considering
cross join
  all_dates
),

joined_with_expensive_table as (
select 
  date as date_24,
  a.lat_index,
  a.lon_index,
  b.lat_index is null as is_missing,
  ifnull(overpasses,0) overpasses
from 
  cells_with_dates  a
left join
  expensive_table b
using(lat_index, lon_index, date)),


predictions_table as
(
{predictions_table}
),

vessel_info as (
select
  ssvid,
  if(on_fishing_list_known is not null, on_fishing_list_known, on_fishing_list_nn) as on_fishing_list
from
   {vessel_info_table}
  -- don't do anything with identity spoofing vessels!
  where activity.overlap_hours_multinames < 24
),

detections_table as

(
  select
    detect_lat,
    detect_lon,
    overpasses_24day,
    detect_id,
    ssvid_mult_recall_length as ssvid,
    score_mult_recall_length as score,
    confidence,
    overpasses_2017_2021,
    date_24,
    length_m,
    7.4e-6 as dd_perkm2
  from
    detection_source_table
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
    and overpasses_2017_2021 > 30
    -- our cutoff for noise -- this could be adjusted down, but makes
    -- very little difference between .5 and .7
    and presence > .7
    and not in_road_doppler
    and not close_to_infra
    and not potential_ambiguity
    {eliminated_locations}
  ),



detections_matched_unmatched as (
select
  date_24,
  overpasses_24day,
  detect_lat,
  detect_lon,
  length_m,
  fishing_score,
  fishing_score_low,
  fishing_score_high,
  case when score > dd_perkm2 and on_fishing_list then "matched_fishing"
   when score > dd_perkm2 and not on_fishing_list  then "matched_nonfishing"
   when score > dd_perkm2 and on_fishing_list is null then "matched_unknown"
   when score < dd_perkm2 then "unmatched" end as matched_category
from
  detections_table a
left join
  vessel_info
using(ssvid)
left join
  predictions_table
using(detect_id)
),


detections_without_zeros as 
(select
  date_24,
  overpasses_24day,
  floor(detect_lat*200) lat_index,
  floor(detect_lon*200) lon_index,
  sum(if( matched_category = 'matched_fishing', 1/overpasses_24day, 0)) matched_fishing,
  sum(if( matched_category = 'matched_nonfishing', 1/overpasses_24day, 0)) matched_nonfishing,
  sum(if( matched_category = 'matched_unknown', 1/overpasses_24day, 0)) matched_unknown,
  sum(if( matched_category = 'matched_unknown',
               fishing_score/overpasses_24day, 0)) matched_unknown_likelyfish,
  sum(if( matched_category = 'matched_unknown',
               (1-fishing_score)/overpasses_24day, 0)) matched_unknown_likelynonfish,
  sum(if( matched_category = 'unmatched', fishing_score/overpasses_24day, 0)) unmatched_fishing,
  sum(if( matched_category = 'unmatched', (1-fishing_score)/overpasses_24day, 0)) unmatched_nonfishing,
  
  sum(if( matched_category = 'unmatched' and length_m > 100,
          (1-fishing_score)/overpasses_24day, 0)) unmatched_nonfishing_100,
  sum(if( matched_category = 'matched_nonfishing' and length_m > 100,
             1/overpasses_24day, 0)) matched_nonfishing_100,
  sum(if( matched_category = 'matched_unknown' and length_m > 100,
               (1-fishing_score)/overpasses_24day, 0)) matched_unknown_likelynonfish_100,  
  sum(1/overpasses_24day) detections
from
  detections_matched_unmatched
group by
  lat_index, lon_index, date_24,overpasses_24day)

select
  lat_index,
  lon_index,
  date_24,
  overpasses,
  matched_fishing,
  matched_nonfishing,
  matched_nonfishing_100,
  matched_unknown_likelyfish,
  matched_unknown_likelynonfish,
  matched_unknown_likelynonfish_100,
  unmatched_fishing,
  unmatched_nonfishing,
  unmatched_nonfishing_100,
  detections
from
  joined_with_expensive_table
left join
  detections_without_zeros
using(lat_index, lon_index, date_24)

'''

import pyperclip
pyperclip.copy(query)

query_to_table(query, f'{project_id}.proj_global_sar.detections_24_w_zeroes_v20230815')

# +

description = '''
This identifies where there are missing values in the 24 day time series. The query to generate it is

''' + query

update_table_description(f"{project_id}.proj_global_sar.detections_24_w_zeroes_v20230815", 
                        description)
# -



# # Now test it with Fernando's interpolated table

# +
q = '''with interpolated as (select 
date_24,
sum(ais_fishing + ais_nonfishing + dark_fishing + dark_nonfishing) as detections_interp
 from scratch_fernando.detections_24_w_interp 
 group by date_24
 order by date_24),

 seen_table as 
(
 select date_24, sum(detections) detections_seen
  from proj_global_sar.detections_24_w_zeroes_v20230815 
 group by date_24 order by date_24

)

select date_24, detections_seen,
ifnull(detections_interp,0) detections_interp
from seen_table
join
interpolated
using(date_24)
where date(date_24) < "2021-12-29"
order by date_24'''

df = pd.read_gbq(q)
# -

import matplotlib.pyplot as plt


df.head()

plt.plot(df.date_24.values, df.detections_seen.values, label = "detected")
plt.plot(df.date_24.values, (df.detections_seen+df.detections_interp).values, label = "detected + interpolated")
plt.legend()
plt.ylabel("total vessels per day")

plt.plot(df.date_24.values, df.detections_seen.values, label = "detected")
plt.plot(df.date_24.values, (df.detections_seen+df.detections_interp).values, label = "detected + interpolated")
plt.legend()
plt.ylabel("total vessels per day")

df.detections_seen.mean()

(df.detections_seen+df.detections_interp).mean()

df.detections_interp.sum()/(df.detections_seen+df.detections_interp).sum()



# +
q = f''' with 

detection_source_table as 
(select * from `proj_global_sar.detections_w_overpasses_v20220929`),

ssel_info as (
select
  ssvid,
  if(on_fishing_list_known is not null, on_fishing_list_known, on_fishing_list_nn) as on_fishing_list
from
   `gfw_research.vi_ssvid_v20221101`
  -- don't do anything with identity spoofing vessels!
  where activity.overlap_hours_multinames < 24
),

detections_table as

(
  select
    overpasses_24day,
    date_24,
    case when periods24_with_overpass >= 70 and periods24_with_overpass < 77 then "almost_enough"
    when periods24_with_overpass = 77 then "all"
    when periods24_with_overpass < 70 then "not_enough"
    end category_seen
  from
    detection_source_table
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
    {eliminated_locations}
  ),


detections_matched_unmatched as (
select
  date_24,
  overpasses_24day,
  category_seen,
from
  detections_table 
),

detections_without_zeros as 
(select
  date_24,
  category_seen,
  sum(1/overpasses_24day) detections
from
  detections_matched_unmatched
group by
   date_24,  category_seen )


select * from detections_without_zeros
order by date_24
 '''

df_c = pd.read_gbq(q)
# -

df_c.head()

# +
d1 = df_c[df_c.category_seen == 'all']
d2 = df_c[df_c.category_seen == 'almost_enough']
d3 = df_c[df_c.category_seen == 'not_enough']

# plt.plot(d1.date_24, d2.detections+d1.detections)
plt.plot(d2.date_24.values, d2.detections.values, label = "70 to 76 overpasses")
plt.plot(d1.date_24.values, d1.detections.values, label = "77 overpasses")
plt.plot(d3.date_24.values, d3.detections.values, label = "under 70")
plt.legend()
plt.title('detectiosn by 24 day period overpass category')
# -


