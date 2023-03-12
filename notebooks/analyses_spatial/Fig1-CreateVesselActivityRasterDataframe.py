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

# # Create Vessel Activity Raster Dataframe
#
# This creates a dataframe that can be used to make activity rasters at a given resolution.
# This is used to create figure 1.
#




# +
# use the standard for eliminating ice locations.
from prj_global_sar_analysis.eliminate_ice_string import eliminate_ice_string
import pyperclip

eliminated_locations = eliminate_ice_string()
pyperclip.copy(eliminated_locations)


import pandas as pd

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

def save_raster(scale):
    # choose a scale, (so, 20 is a 20th of a degree) to save the dataframe as

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
      detect_lat,
      detect_lon,
      detect_id,
      eez_iso3,
      ssvid_mult_recall_length as ssvid,
      score_mult_recall_length as score,
      overpasses_2017_2021,
      7.4e-6 as dd_perkm2
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
      and overpasses_2017_2021 >= 30
      -- our cutoff for noise -- this could be adjusted down, but makes
      -- very little difference between .5 and .7
      and presence > .7
      {eliminated_locations}
      and not in_road_doppler
      ),

    final_table as (
    select
      detect_lat,
      detect_lon,
      overpasses_2017_2021,
      fishing_score,
      case when score > dd_perkm2 and on_fishing_list then "matched_fishing"
       when score > dd_perkm2 and not on_fishing_list then "matched_nonfishing"
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
      floor(detect_lat*{scale}) lat_index,
      floor(detect_lon*{scale}) lon_index,
      sum(if( matched_category = 'matched_fishing', 1/overpasses_2017_2021, 0)) matched_fishing,
      sum(if( matched_category = 'matched_nonfishing', 1/overpasses_2017_2021, 0)) matched_nonfishing,
      
      sum(if( matched_category = 'matched_unknown',
                   fishing_score/overpasses_2017_2021, 0)) matched_unknown_likelyfish,
      sum(if( matched_category = 'matched_unknown',
                   (1-fishing_score)/overpasses_2017_2021, 0)) matched_unknown_likelynonfish,
                   
      sum(if( matched_category = 'unmatched', fishing_score/overpasses_2017_2021, 0)) dark_fishing,
      sum(if( matched_category = 'unmatched', (1-fishing_score)/overpasses_2017_2021, 0)) dark_nonfishing,
      avg(overpasses_2017_2021) overpasses_2017_2021,
      sum(1/overpasses_2017_2021) detections
    from
      final_table
    group by
     lat_index, lon_index'''

    # for debugging... to copy to clipboard
    pyperclip.copy(q)

    dfr = pd.read_gbq(q)

    dfr['tot_fishing'] = dfr.dark_fishing + dfr.matched_fishing + dfr.matched_unknown_likelyfish
    dfr['tot_nonfishing'] = dfr.dark_nonfishing + dfr.matched_nonfishing + dfr.matched_unknown_likelynonfish
    
    dfr['AIS_fishing'] = dfr.matched_fishing + dfr.matched_unknown_likelyfish
    dfr['AIS_nonfishing'] = dfr.matched_nonfishing + dfr.matched_unknown_likelynonfish
  
    dfr.to_feather(f"../../data/raster_{scale}th_degree.feather")


save_raster(20)

save_raster(10) # 10th of a degree is what we use for the main figures

save_raster(5) # 10th of a degree is what we use for the main figures


