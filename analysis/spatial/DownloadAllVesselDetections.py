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

# # Download All Vessel Detections
# This creates a table and downloads a dataframe to a compressed CSV that contains all likely vessel detections. These detections are in turn used to produce the figures that show individual vessels.

import pandas as pd
import pyperclip


# Get the query templates
import sys
sys.path.append('../utils')
from vessel_queries import *
from proj_id import project_id
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
pyperclip.copy(q)

# +
## Uncomment to run more

table_id = f'{project_id}.proj_global_sar.detections_classified_v20230803'
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
lat, 
lon,
year,
fishing_score,
matched_category,
overpasses_2017_2021
from 
proj_global_sar.detections_classified_v20230803

''')

df.to_feather("../data/all_detections_v20230803.feather")

len(df)


