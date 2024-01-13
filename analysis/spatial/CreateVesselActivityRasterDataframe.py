# ---
# jupyter:
#   jupytext:
#     custom_cell_magics: kql
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

# # Create Vessel Activity Raster Dataframe
#
# This creates a dataframe that can be used to make activity rasters at a given resolution.
# This is used to create figure 1.
#

import pandas as pd


import sys
sys.path.append('../utils')
from vessel_queries import *

# +
scale = 200
q = f'''

{final_query_static} -- from the file vessel_queries

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

import pyperclip

pyperclip.copy(q)


# -

def save_raster(scale):
    # choose a scale, (so, 20 is a 20th of a degree) to save the dataframe as

    q = f'''
    
    {final_query_static} -- from the file vessel_queries

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
#     pyperclip.copy(q)

    dfr = pd.read_gbq(q)

    dfr['tot_fishing'] = dfr.dark_fishing + dfr.matched_fishing + dfr.matched_unknown_likelyfish
    dfr['tot_nonfishing'] = dfr.dark_nonfishing + dfr.matched_nonfishing + dfr.matched_unknown_likelynonfish
    
    dfr['AIS_fishing'] = dfr.matched_fishing + dfr.matched_unknown_likelyfish
    dfr['AIS_nonfishing'] = dfr.matched_nonfishing + dfr.matched_unknown_likelynonfish
  
    dfr.to_feather(f"../data/raster_{scale}th_degree.feather")


save_raster(10) # 10th of a degree is what we use for the main figures

save_raster(5) # 10th of a degree is what we use for the main figures

save_raster(20)

df = pd.read_feather("../data/raster_5th_degree.feather")

df.columns

for c in df.columns:
    print(c)


