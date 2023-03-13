# ---
# jupyter:
#   jupytext:
#     custom_cell_magics: kql
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

# # Download All Vessel Detections
# This creates a table and downloads a dataframe to a compressed CSV that contains all likely vessel detections. These detections are in turn used to produce the figures that show individual vessels.

import pandas as pd
import pyperclip
from prj_global_sar_analysis.bigquery_helper_functions import query_to_table, update_table_description

# use the standard for eliminating ice locations.
from prj_global_sar_analysis.eliminate_ice_string import eliminate_ice_string
eliminated_locations = eliminate_ice_string()
pyperclip.copy(eliminated_locations)

# +
vessel_info_table = "gfw_research.vi_ssvid_v20221001"


predictions_table = """

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
"""


# -

min_lon, min_lat, max_lon, max_lat = -180,-90,180,90 # globe -- we are going to download *everything*

# # Download with more details on each detection

# +
# query to get every single vessel detection...

q = f'''with
predictions_table as
(
{predictions_table}
),

vessel_info as (
select
  ssvid,
  if(on_fishing_list_known is not null, on_fishing_list_known, on_fishing_list_nn) as on_fishing_list
from
   `world-fishing-827.{vessel_info_table}`
  -- don't do anything with identity spoofing vessels!
  where activity.overlap_hours_multinames < 24
),

detections_table as

(
  select
  floor(detect_lat*5) as lat_index_d,
  floor(detect_lon*5) as lon_index_d,
  extract(year from detect_timestamp) year,
  detect_lat,
  detect_lon,
  detect_id,
  ssvid_mult_recall_length as ssvid,
  eez_iso3,
  score_mult_recall_length as score,
  confidence_mult_recall_length as confidence,
  overpasses_2017_2021,
  date_24,
  7.4e-6 as dd_perkm2,
  in_road_doppler,
  in_road_doppler2
  from
  `world-fishing-827.proj_global_sar.detections_w_overpasses_v20230215`
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
  and detect_lon between {min_lon} and {max_lon}
  and detect_lat between {min_lat} and {max_lat}
  
  ),






final_table as (
select
  lat_index_d,
  lon_index_d,
  year,
  date_24,
  detect_lat,
  detect_lon,
  overpasses_2017_2021,
  eez_iso3,
  fishing_score,
  fishing_score_low,
  fishing_score_high,
  score,
  on_fishing_list,
  dd_perkm2,
  confidence,
  in_road_doppler,
  in_road_doppler2,
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

)

select
  detect_lat as lat,
  detect_lon as lon,
  year,
  in_road_doppler,
  in_road_doppler2,
  matched_category,
  fishing_score,
  confidence,
  score,
  on_fishing_list,
  dd_perkm2,
  overpasses_2017_2021
from
  final_table'''

# +
## Uncomment to run more

# table_id = 'world-fishing-827.proj_global_sar.detections_classified_v20230216'
# query_to_table(q, table_id)



# +
# description = '''
# This classifies each likely vessel detection into categories (matched fishing, matched non-fihsing, etc.
# ''' + q

# update_table_description(table_id, 
#                         description)
# -

# # Download this locally
#
# This is downloading > 1gb of data, so it takes a long time

df = pd.read_gbq('''
select 
    lat, lon, 
    case when matched_category = 'matched_nonfishing' then 'matched_nonfishing'
    when matched_category = 'matched_fishing' then 'matched_fishing'
    when matched_category = 'matched_unknown' and random > fishing_score then 'matched_fishing'
    when matched_category = 'matched_unknown' and random > fishing_score then 'matched_fishing'
    when matched_category = 'matched_unknown' and random < fishing_score then 'matched_fishing'
    when matched_category = 'matched_unknown' and random > fishing_score then 'matched_nonfishing'
    when matched_category = 'unmatched' and random < fishing_score then 'dark_fishing'
    when matched_category = 'unmatched' and random > fishing_score then 'dark_nonfishing'
    else "none" end as category_rand
from 
 (select *, rand()  as random from  proj_global_sar.detections_classified_v20230216)
''')

# save this *very large* data frame to a csv 
df.to_feather("../../data/all_detections_matched_rand.feather")

# +
df = pd.read_gbq('''
select lat, lon, 
    case when matched_category = 'matched_nonfishing' then 'matched_nonfishing'
    when matched_category = 'matched_fishing' then 'matched_fishing'
    when matched_category = 'matched_unknown' and random > fishing_score then 'matched_fishing'
    when matched_category = 'matched_unknown' and random > fishing_score then 'matched_fishing'
    when matched_category = 'matched_unknown' and random < fishing_score then 'matched_fishing'
    when matched_category = 'matched_unknown' and random > fishing_score then 'matched_nonfishing'
    when matched_category = 'unmatched' and random < fishing_score then 'dark_fishing'
    when matched_category = 'unmatched' and random> fishing_score then 'dark_nonfishing'
    else "none" end as category_rand
from 
 (select *, rand()  as random from  proj_global_sar.detections_classified_v20230216)
  where year = 2021
''')

df.to_feather("../../data/all_detections_matched_rand_2021.feather")
# -


