# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

from eliminate_ice_string import *
eliminated_locations = eliminate_ice_string()

matching_threshold = '7.4e-6'
vessel_info_table = "gfw_research.vi_ssvid_v20230701"


# +

predictions_table = f'''
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
    '''


# -

detections_table = f'''with
predictions_table as
(
{predictions_table}
),
vessel_info as (
select
  ssvid,
  if(on_fishing_list_known is not null, on_fishing_list_known, on_fishing_list_nn) as on_fishing_list
from
  `{vessel_info_table}`
  -- don't do anything with identity spoofing vessels!
  where activity.overlap_hours_multinames < 24
),


detections_table as
(
  select
    detect_lat,
    detect_lon,
    detect_id,
    ssvid_mult_recall_length as ssvid,
    eez_iso3,
    score_mult_recall_length as score,  
    confidence_mult_recall_length as confidence,  
    {matching_threshold} as dd_perkm2,
    overpasses_2017_2021,
    extract(year from detect_timestamp) year,
    date_24,
    in_road_doppler,
    in_road_doppler2,
    periods24_with_overpass,
    overpasses_24day,
    length_m
    
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
  and overpasses_2017_2021 > 30
  -- our cutoff for noise -- this could be adjusted down, but makes
  -- very little difference between .5 and .7
  and presence > .7
  and not in_road_doppler
  and not close_to_infra
  and not potential_ambiguity
  {eliminated_locations}
  ) '''

final_query_static = f'''

{detections_table},
  
final_table as (
select
  date_24,
  year,
  detect_lat,
  detect_lon,
  detect_id,
  overpasses_2017_2021,
  eez_iso3,
  fishing_score,
  fishing_score_low,
  fishing_score_high,
  case when score > dd_perkm2 and on_fishing_list then "matched_fishing"
   when score > dd_perkm2 and not on_fishing_list then "matched_nonfishing"
   when score > dd_perkm2 and on_fishing_list is null then "matched_unknown"
   when score < dd_perkm2 then "unmatched" end as matched_category,
  in_road_doppler,
  in_road_doppler2,
  confidence,
  score,
  on_fishing_list,
  ssvid, 
  length_m
from
  detections_table a
left join
  vessel_info
using(ssvid)
left join
  predictions_table
using(detect_id)
)'''

fields_static = '''  
  sum(if( matched_category = 'matched_fishing', 1/overpasses_2017_2021, 0)) matched_fishing,
  sum(if( matched_category = 'matched_nonfishing', 1/overpasses_2017_2021, 0)) matched_nonfishing,
  sum(if( matched_category = 'matched_unknown', 1/overpasses_2017_2021, 0)) matched_unknown,
  sum(if( matched_category = 'matched_unknown',
               fishing_score/overpasses_2017_2021, 0)) matched_unknown_likelyfish,
  sum(if( matched_category = 'matched_unknown',
               (1-fishing_score)/overpasses_2017_2021, 0)) matched_unknown_likelynonfish,
  sum(if( matched_category = 'unmatched', fishing_score/overpasses_2017_2021, 0)) unmatched_fishing,
  sum(if( matched_category = 'unmatched', (1-fishing_score)/overpasses_2017_2021, 0)) unmatched_nonfishing,
  
  -- use lower calibrated score
  sum(if( matched_category = 'matched_unknown',
               fishing_score_low/overpasses_2017_2021, 0)) matched_unknown_likelyfish_low,
  sum(if( matched_category = 'matched_unknown',
               (1-fishing_score_low)/overpasses_2017_2021, 0)) matched_unknown_likelynonfish_low,
  sum(if( matched_category = 'unmatched', 
               fishing_score_low/overpasses_2017_2021, 0)) unmatched_fishing_low,
  sum(if( matched_category = 'unmatched', 
                (1-fishing_score_low)/overpasses_2017_2021, 0)) unmatched_nonfishing_low,

  -- use higher calibrated score
  sum(if( matched_category = 'matched_unknown',
               fishing_score_high/overpasses_2017_2021, 0)) matched_unknown_likelyfish_high,
  sum(if( matched_category = 'matched_unknown',
               (1-fishing_score_high)/overpasses_2017_2021, 0)) matched_unknown_likelynonfish_high,
  sum(if( matched_category = 'unmatched', fishing_score_high/overpasses_2017_2021, 0)) unmatched_fishing_high,
  sum(if( matched_category = 'unmatched', (1-fishing_score_high)/overpasses_2017_2021, 0)) unmatched_nonfishing_high,
 
  -- all detections
  sum(1/overpasses_2017_2021) detections 
 '''


