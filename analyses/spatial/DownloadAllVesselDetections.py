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


# Get the query templates
import sys
sys.path.append('../utils') 
from vessel_queries import *
from bigquery_helper_functions import query_to_table, update_table_description

# # Download with more details on each detection

# +
# query to get every single vessel detection...

q = f'''
{final_query_static} -- from the file vessel_queries


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
  overpasses_2017_2021
from
  final_table'''

# +
## Uncomment to run more

table_id = 'project-id.proj_global_sar.detections_classified_v20230326'
query_to_table(q, table_id)



# +
description = '''
This classifies each likely vessel detection into categories (matched fishing, matched non-fihsing, etc.
''' + q

update_table_description(table_id, 
                        description)
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
 (select *, rand()  as random from  proj_global_sar.detections_classified_v20230326)
''')

# save this *very large* data frame to a csv 
df.to_feather("../data/all_detections_matched_rand.feather")

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
 (select *, rand()  as random from  proj_global_sar.detections_classified_v20230326)
  where year = 2021
''')

df.to_feather("../data/all_detections_matched_rand_2021.feather")
# -


