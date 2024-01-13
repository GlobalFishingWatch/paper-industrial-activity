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

# # AIS Fishing Activity by Continent
#
# This produces the table in the supplement that identifies the amount of AIS fishing by continent from GFW fishing effort data

import pandas as pd

q = '''with eez_info as 
(
SELECT 
  cast(eez_id as string) eez_id, 
  territory1_iso3 
FROM 
  `project-id.gfw_research.eez_info` 
),

fishing_vessels as (
select 
  ssvid, 
  year
from 
  `gfw_research.fishing_vessels_ssvid_v20221201`
),

good_segs as (
select 
  seg_id 
from 
  `gfw_research.pipe_v20201001_segs` 
where 
  good_seg and not overlapping_and_short
),

activity as (
SELECT 
  sum(hours) fishing_hours,
  regions.eez[offset(0)] as eez_id
FROM 
 `project-id.gfw_research.pipe_v20201001_fishing`  a
join
 good_segs b
using(seg_id)
join
 fishing_vessels c
on
  extract(year from _partitiontime) = c.year
  and a.ssvid = c.ssvid
where 
 array_length(regions.eez)>0
 and date(_partitiontime) between "2017-01-01" and "2021-12-31"
 and nnet_score > .5
group 
  by eez_id 
)

select 
  territory1_iso3,
  sum(fishing_hours) fishing_hours 
from 
  activity
join 
  eez_info
using(eez_id)
group 
  by territory1_iso3
'''
df = pd.read_gbq(q)

# +
import pycountry
from pycountry_convert import (
    country_alpha2_to_continent_code,
    country_alpha2_to_country_name,
    country_name_to_country_alpha2,
)

continents = {
    'NA': 'North America',
    'SA': 'South America',
    'AS': 'Asia',
    'OC': 'Australia',
    'AF': 'Africa',
    'EU': 'Europe'
}

def get_continent(x):
    try:
        return continents[country_alpha2_to_continent_code(pycountry.countries.get(alpha_3=x).alpha_2)]
    except:
        "None"

def get_country(x):
    try:
        return country_alpha2_to_country_name(pycountry.countries.get(alpha_3=x).alpha_2)
    except:
        "None"


# -

df['continent'] = df.territory1_iso3.apply(get_continent)

d = df.groupby("continent").sum()
d

d['fraction'] = d.fishing_hours/d.fishing_hours.sum()

d

from tabulate import tabulate
rows = []
for index, row in d.iterrows():
    rows.append([index,f"{row.fraction:.2f}"])
print(tabulate((rows)))

from tabulate import tabulate
rows = []
for index, row in d.iterrows():
    rows.append([index,f"{row.fraction:.2f}"])
print(tabulate((rows)))

print(f"{d.fishing_hours.sum()/1e6:.1f} million fishing hours")

# # How many AIS positions in GFW database?

# +
## all vessels
q = '''
select 
  count(*)/1e9 
from 
  `project-id.gfw_research.pipe_v20201001`
where 
  date(_partitiontime) between "2017-01-01" and "2021-12-31"
'''

pd.read_gbq(q)

# +
## just fishing vessels
q = '''

select 
  count(*)/1e9 
from 
  `project-id.gfw_research.pipe_v20201001_fishing` a
join
   `gfw_research.fishing_vessels_ssvid_v20221201` b
on 
  extract(year from a._partitiontime) = b.year
  and a.ssvid = b.ssvid
where date(_partitiontime) between "2017-01-01" and "2021-12-31"
'''

pd.read_gbq(q)
# -


