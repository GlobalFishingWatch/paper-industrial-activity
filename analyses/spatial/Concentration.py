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

import numpy as np
import pandas as pd
import proplot as pplt
import pyseas.cm
import pyseas.contrib as psc
import matplotlib.pyplot as plt

# +
scale = 10

from prj_global_sar_analysis.eliminate_ice_string import eliminate_ice_string
eliminated_locations = eliminate_ice_string()

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


# +
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
   {vessel_info_table}

),

detections_table as

(
  select
  detect_lat,
  detect_lon,
  detect_id,
  ssvid_mult_recall_length as ssvid,
  score_mult_recall_length as score,
  overpasses_2017_2021,
  7.4e-6 as dd_perkm2,
  eez_iso3
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
  -- at least 30 overpasses
  and overpasses_2017_2021 > 30
  -- our cutoff for noise -- this could be adjusted down, but makes
  -- very little difference between .5 and .7
  and presence > .7
  and not in_road_doppler
  {eliminated_locations}
),



final_table as (
select
  detect_lat,
  detect_lon,
  eez_iso3,
  overpasses_2017_2021,
  fishing_score,
  fishing_score_low,
  fishing_score_high,
  case when score > dd_perkm2 and on_fishing_list  then "matched_fishing"
   when score > dd_perkm2 and not on_fishing_list then "matched_nonfishing"
   when score > dd_perkm2 and on_fishing_list is null  then "matched_unknown"
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

gridded as (
select
  floor(detect_lat*{scale}) lat_index,
  floor(detect_lon*{scale}) lon_index,
  sum(if( matched_category = 'matched_fishing', 1/overpasses_2017_2021, 0)) matched_fishing,
  sum(if( matched_category = 'matched_nonfishing', 1/overpasses_2017_2021, 0)) matched_nonfishing,
  sum(if( matched_category = 'matched_unknown', 1/overpasses_2017_2021, 0)) matched_unknown,
  sum(if( matched_category = 'matched_unknown',
               fishing_score/overpasses_2017_2021, 0)) matched_unknown_likelyfish,
  sum(if( matched_category = 'matched_unknown',
               (1-fishing_score)/overpasses_2017_2021, 0)) matched_unknown_likelynonfish,
  sum(if( matched_category = 'unmatched', fishing_score/overpasses_2017_2021, 0)) unmatched_fishing,
  sum(if( matched_category = 'unmatched', (1-fishing_score)/overpasses_2017_2021, 0)) unmatched_nonfishing,
  
  sum(if( matched_category = 'matched_unknown',
               fishing_score_low/overpasses_2017_2021, 0)) matched_unknown_likelyfish_low,
  sum(if( matched_category = 'matched_unknown',
               (1-fishing_score_low)/overpasses_2017_2021, 0)) matched_unknown_likelynonfish_low,
  sum(if( matched_category = 'unmatched', 
               fishing_score_low/overpasses_2017_2021, 0)) unmatched_fishing_low,
  sum(if( matched_category = 'unmatched', 
                (1-fishing_score_low)/overpasses_2017_2021, 0)) unmatched_nonfishing_low,

  sum(if( matched_category = 'matched_unknown',
               fishing_score_high/overpasses_2017_2021, 0)) matched_unknown_likelyfish_high,
  sum(if( matched_category = 'matched_unknown',
               (1-fishing_score_high)/overpasses_2017_2021, 0)) matched_unknown_likelynonfish_high,
  sum(if( matched_category = 'unmatched', fishing_score_high/overpasses_2017_2021, 0)) unmatched_fishing_high,
  sum(if( matched_category = 'unmatched', (1-fishing_score_high)/overpasses_2017_2021, 0)) unmatched_nonfishing_high,
  
  
  sum(1/overpasses_2017_2021) detections
from
  final_table
group by
 lat_index, lon_index
)

 select 
 111*111/{scale}/{scale}*cos(3.1416/180*lat_index/{scale}) as area_km,
 matched_fishing + matched_unknown_likelyfish + unmatched_fishing as fishing,
 matched_nonfishing + matched_unknown_likelynonfish + unmatched_nonfishing as nonfishing,
 detections,
 from 
 gridded
 where detections >0
 order by fishing desc

'''

df = pd.read_gbq(q)
# -



# ## What is the total study area?

e = eliminated_locations.replace("detect_lon","(lon_index/200 + 1/400)")
e = e.replace("detect_lat","(lat_index/200 + 1/400)")

# +
q  = f'''
with
footprints as
(SELECT 
  lon_index, 
  lat_index,
  sum(overpasses) overpasses, 
FROM 
  `world-fishing-827.proj_global_sar.overpasses_200_by_year_filtered_v20220508`
where 
  year between 2017 and 2021 
  {e}
group by lat_index, lon_index
having overpasses >= 30
)

select 
sum(111*111/200/200*cos((lat_index/200+1/400)*3.1416/180)) area_km2_imaged,
from footprints

'''

area_km2_imaged = pd.read_gbq(q).area_km2_imaged.values[0]
area_km2_imaged/1e6
# -

df.head()

# +
d = df[df.fishing>0].sort_values('fishing', ascending=False)
d['fishing_cumsum'] = d.fishing.cumsum()
d['fishing_area_cumsum'] = d.area_km.cumsum()
tot_fishing = d.fishing.sum()


comparison_threshold=.25
for index, row in d.iterrows():
    if row.fishing_cumsum/tot_fishing > comparison_threshold:
        print(f"{comparison_threshold*100:.1f}% of fishing is in {row.fishing_area_cumsum/1e6:.2f}M km2, \
 {row.fishing_area_cumsum/area_km2_imaged*100:.1f}% of area imaged")
        break
        
comparison_threshold = .5
for index, row in d.iterrows():
    if row.fishing_cumsum/tot_fishing > comparison_threshold:
        print(f"{comparison_threshold*100:.1f}% of fishing is in {row.fishing_area_cumsum/1e6:.2f}M km2, \
 {row.fishing_area_cumsum/area_km2_imaged*100:.1f}% of area imaged")
        break

        
print(f"100% of fishing is in {d.fishing_area_cumsum.max()/1e6:.2f}M km2, \
 {d.fishing_area_cumsum.max()/area_km2_imaged*100:.1f}% of area imaged")

# -



# +
d = df[df.nonfishing>0].sort_values('nonfishing', ascending=False)
d['nonfishing_cumsum'] = d.nonfishing.cumsum()
d['nonfishing_area_cumsum'] = d.area_km.cumsum()

tot_nonfishing = d.nonfishing.sum()

comparison_threshold = .25
for index, row in d.iterrows():
    if row.nonfishing_cumsum/tot_nonfishing > comparison_threshold:
        print(f"{comparison_threshold*100:.1f}% of nonfishing is in {row.nonfishing_area_cumsum/1e6:.2f}M km2, \
 {row.nonfishing_area_cumsum/area_km2_imaged*100:.0f}% of area imaged")
        break

comparison_threshold = .5
for index, row in d.iterrows():
    if row.nonfishing_cumsum/tot_nonfishing > comparison_threshold:
        print(f"{comparison_threshold*100:.1f}% of nonfishing is in {row.nonfishing_area_cumsum/1e6:.2f}M km2, \
 {row.nonfishing_area_cumsum/area_km2_imaged*100:.0f}% of area imaged")
        break

print(f"100% of nonfishing is in {d.nonfishing_area_cumsum.max()/1e6:.2f}M km2, \
 {d.nonfishing_area_cumsum.max()/area_km2_imaged*100:.0f}% of area imaged")

# +
d = df[df.detections>0].sort_values('detections', ascending=False)
d['vessels_cumsum'] = d.detections.cumsum()
d['vessels_area_cumsum'] = d.area_km.cumsum()

tot_vessels = d.detections.sum()

comparison_threshold = .25
for index, row in d.iterrows():
    if row.vessels_cumsum/tot_vessels > comparison_threshold:
        print(f"{comparison_threshold*100:.1f}% of vessels are in {row.vessels_area_cumsum/1e6:.2f}M km2, \
 {row.vessels_area_cumsum/area_km2_imaged*100:.1f}% of area imaged")
        break

comparison_threshold = .5
for index, row in d.iterrows():
    if row.vessels_cumsum/tot_vessels > comparison_threshold:
        print(f"{comparison_threshold*100:.1f}% of vessels are in {row.vessels_area_cumsum/1e6:.2f}M km2, \
 {row.vessels_area_cumsum/area_km2_imaged*100:.1f}% of area imaged")
        break

print(f"100% of vessels is in {d.vessels_area_cumsum.max()/1e6:.2f}M km2, \
 {d.vessels_area_cumsum.max()/area_km2_imaged*100:.0f}% of area imaged")

# -




# +
plt.figure(figsize=(8,4),facecolor="white")
plt.plot(df.area_km.cumsum()/1e6, df.fishing.cumsum()/df.fishing.sum(),label = 'fishing')

d = df.sort_values('nonfishing', ascending=False)
plt.plot(d.area_km.cumsum()/1e6, d.nonfishing.cumsum()/d.nonfishing.sum(), label = 'nonfishing')
# plt.xlim(0,2e6)
plt.legend(frameon=False)
plt.ylabel("fraction of activity")
plt.xlabel("area, million km2")

