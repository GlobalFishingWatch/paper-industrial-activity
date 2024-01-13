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

# # Concentration
#
#

import numpy as np
import pandas as pd
import proplot as pplt
import pyseas.cm
import pyseas.contrib as psc
import matplotlib.pyplot as plt

# +
scale = 10

import sys
sys.path.append('../utils')
from vessel_queries import *
from eliminate_ice_string import *
eliminated_locations = eliminate_ice_string()


# +
q = f'''

{final_query_static} -- from the file vessel_queries

,

gridded as (
select
  floor(detect_lat*{scale}) lat_index,
  floor(detect_lon*{scale}) lon_index,
  {fields_static} -- from the file vessel_queries
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
# +
# import pyperclip
# pyperclip.copy(q)
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
  `proj_global_sar.overpasses_200_by_year_filtered_v20220508`
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
d = df.sort_values('detections', ascending=False)
d['detects_cumsum'] = d.detections.cumsum()
d['detects_area_cumsum'] = d.area_km.cumsum()
tot = d.detections.sum()


comparison_threshold=.25
for index, row in d.iterrows():
    if row.detects_cumsum/tot > comparison_threshold:
        print(f"{comparison_threshold*100:.1f}% of all activity is in {row.detects_area_cumsum/1e6:.2f}M km2, \
 {row.detects_area_cumsum/area_km2_imaged*100:.1f}% of area imaged")
        break
        
comparison_threshold = .5
for index, row in d.iterrows():
    if row.detects_cumsum/tot> comparison_threshold:
        print(f"{comparison_threshold*100:.1f}% of all activity is in {row.detects_area_cumsum/1e6:.2f}M km2, \
 {row.detects_area_cumsum/area_km2_imaged*100:.1f}% of area imaged")
        break

        
print(f"100% of fishing is in {d.detects_area_cumsum.max()/1e6:.2f}M km2, \
 {d.detects_area_cumsum.max()/area_km2_imaged*100:.1f}% of area imaged")


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
plt.plot(np.array(df.area_km.cumsum()/1e6), np.array(df.fishing.cumsum()/df.fishing.sum()),label = 'fishing')

d = df.sort_values('nonfishing', ascending=False)
plt.plot(np.array(d.area_km.cumsum()/1e6), np.array(d.nonfishing.cumsum()/d.nonfishing.sum()), label = 'nonfishing')
# plt.xlim(0,2e6)
plt.legend(frameon=False)
plt.ylabel("fraction of activity")
plt.xlabel("area, million km2")

# -


