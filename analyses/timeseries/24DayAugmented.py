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

import pandas as pd
import matplotlib.pyplot as plt
import proplot as pplt
import pyperclip
from datetime import datetime, timedelta, date
import numpy as np

# +
# use the standard for eliminating ice locations.
import sys
sys.path.append("../utils")
from vessel_queries import *

# ice string elimination
from eliminate_ice_string import eliminate_ice_string

eliminated_locations = eliminate_ice_string()
# -

# # Unsmoothed Time Series

seen_table = "proj_global_sar.detections_24_w_zeroes_v20230815"
interpolated_table = "scratch_fernando.detections_24_w_interp_v20230815"
vessel_info_table = "gfw_research.vi_ssvid_v20230701"


# +


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

# +
q = f'''with interpolated as (select 
date_24,
sum(ais_fishing + ais_nonfishing + dark_fishing + dark_nonfishing) as detections_interp
 from {interpolated_table} 
 group by date_24
 order by date_24),

 seen_table as 
(
 select date_24, sum(detections) detections_seen
  from {seen_table} 
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

df.head()

plt.figure(figsize=(10,4))
plt.plot(df.date_24.values, df.detections_seen.values, label = "detected")
plt.plot(df.date_24.values, (df.detections_seen+df.detections_interp).values, label = "detected + interpolated")
plt.legend()
plt.ylabel("total vessels per day")

# ## What fraction of detections are interpolated?

df.detections_interp.sum()/(df.detections_seen.sum()+df.detections_interp.sum())

(df.detections_seen+df.detections_interp).mean()

# what fraction of the activity in our study area is in these areas that
# we are using for thie time series? 64500 is the number of vessels we see on average
# everywhere that has at least 30 overpasses over 2017-2021
60322/64500

# +



q = f"""


with

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
  date_24,
  periods24_with_overpass,
  overpasses_24day,
  detect_lat,
  detect_lon,
  detect_id,
  score_mult_recall_length as score,
  ssvid_mult_recall_length as ssvid,
  eez_iso3,
  length_m,
  overpasses_2017_2021,  
  7.4e-6 as dd_perkm2
  from
  proj_global_sar.detections_w_overpasses_v20230803
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


final_table as (
select
  detect_id,
  overpasses_24day,
  date_24,
  eez_iso3,
  fishing_score,
  length_m,
  case when score > dd_perkm2 and on_fishing_list then "matched_fishing"
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
where
  -- important! only places with 70 or more detections... 
  periods24_with_overpass >= 70

)

select 
  date(rolling_date) rolling_date,
  extract(year from rolling_date) year,
  eez_iso3,
  sum(if( matched_category = 'matched_fishing', 1/overpasses, 0)) matched_fishing,
  sum(if( matched_category = 'matched_nonfishing', 1/overpasses, 0)) matched_nonfishing,
  sum(if( matched_category = 'matched_unknown', 1/overpasses, 0)) matched_unknown,
  sum(if( matched_category = 'matched_unknown',
               fishing_score/overpasses, 0)) matched_unknown_likelyfish,
  sum(if( matched_category = 'matched_unknown',
               (1-fishing_score)/overpasses, 0)) matched_unknown_likelynonfish,               
  sum(if( matched_category = 'unmatched', fishing_score/overpasses, 0)) unmatched_fishing,
  sum(if( matched_category = 'unmatched', (1-fishing_score)/overpasses, 0)) unmatched_nonfishing,
  sum(1/overpasses) detections,
  
  sum(if( matched_category = 'unmatched' and length_m > 100,
          (1-fishing_score)/overpasses_24day, 0)) unmatched_nonfishing_100,
          
  sum(if( matched_category = 'matched_nonfishing' and length_m > 100,
             1/overpasses_24day, 0)) +
  sum(if( matched_category = 'matched_unknown' and length_m > 100,
               (1-fishing_score)/overpasses_24day, 0)) matched_nonfish_100
from 
  final_table
join
  proj_global_sar.detections_w_overpasses_rolling24_v20220805 
using(detect_id)
group by 
  eez_iso3, rolling_date, year
order by rolling_date



"""
import pyperclip

# uncomment to copy to clipboard
pyperclip.copy(q)
# -

df_detect = pd.read_gbq(q)

# +
df = df_detect
df["ais_fishing"] = df.matched_fishing + df.matched_unknown_likelyfish
df["dark_fishing"] = df.unmatched_fishing
df["ais_nonfishing"] = df.matched_nonfishing + df.matched_unknown_likelynonfish
df["dark_nonfishing"] = df.unmatched_nonfishing
df["ais_nonfishing100"] = df.matched_nonfish_100
df["dark_nonfishing100"] = df.unmatched_nonfishing_100
df['eez_iso3'] = df['eez_iso3'].fillna("None")

df = df[
    [
        "rolling_date",
        "eez_iso3",
        "ais_fishing",
        "dark_fishing",
        "ais_nonfishing",
        "dark_nonfishing",
        "ais_nonfishing100",
        "dark_nonfishing100",
    ]
]
df = df.groupby(["rolling_date", "eez_iso3"]).sum().reset_index()
# -
df_detect.head()




# # Turn the interoplated positions into 24 day interpolations
#
#

# +
q = f'''

with 
interp_table as 
(select * from {interpolated_table}),

date_table as 
-- generate array of tables
(
select *,
timestamp_add( timestamp("2017-01-01"),
       interval cast( floor(timestamp_diff(timestamp(date),  
                                    timestamp("2017-01-01"),
                                     day)/24)*24 as int64) day) as date_24,
 from unnest(GENERATE_DATE_ARRAY("2017-01-01", "2021-12-31")) as date
),


-- Get the EEZ
regions as (
SELECT 
  regions.eez as eez_array, 
  -- If there are multiple EEZ values, just get the first one
  if(array_length(regions.eez)>0, regions.eez[ordinal(1)], null) MRGID,
  gridcode 
FROM 
  `pipe_static.spatial_measures_20201105` 
),


eez_table as (
select 
  lat_index, lon_index,
  ISO_TER1 as eez_iso3
from 
  (select 
    distinct lat_index, lon_index, 
    format("lon:%+07.2f_lat:%+07.2f", 
      lon_index/200 +1/400, lat_index/200 + 1/400 
      ) as gridcode, -- for matching
   from 
     interp_table
  )
join
  regions
using(gridcode)
left join
  proj_global_sar.eez_info_v11
using(MRGID)
),


smoothed as 
(select 
  date,
  date_24,
  eez_iso3,
  lat_index,
  lon_index,
  ais_fishing/24 as ais_fishing,
  ais_nonfishing/24 as ais_nonfishing,
  dark_fishing/24 as dark_fishing,
  dark_nonfishing/24 as dark_nonfishing,
  ais_nonfishing_100/24 as ais_nonfishing100,
  dark_nonfishing_100/24 as dark_nonfishing100
from 
  interp_table 
join
  date_table
using(date_24)
  join
eez_table 
  using (lat_index, lon_index)
),


rolling_date_table as 
(select 
  a.date as rolling_date,
  b.date as date
from 
  date_table a
cross join
  date_table b
where 
  b.date between 
              timestamp_sub(a.date, interval 12 day) 
              and timestamp_add(a.date, interval 11 day)
)

select 
  rolling_date,
  eez_iso3,
  sum(ais_fishing) as ais_fishing_i,
  sum(ais_nonfishing) as ais_nonfishing_i,
  sum(dark_fishing) as dark_fishing_i,
  sum(dark_nonfishing) as dark_nonfishing_i,
  sum(ais_nonfishing100) as ais_nonfishing100_i,
  sum(dark_nonfishing100) as dark_nonfishing100_i
from 
  smoothed
join
  rolling_date_table
using(date)
group by 
  rolling_date, eez_iso3
order by 
  eez_iso3, rolling_date

'''

## uncomment to copy to clipboard
pyperclip.copy(q)

# -

df_int = pd.read_gbq(q)

df_int['eez_iso3'].head()

df_int['eez_iso3']=df_int['eez_iso3'].fillna("None")



# +

# pandas.merge() by Column
df = pd.merge(df,df_int, how='outer',on=['eez_iso3','rolling_date']).reset_index()

# replace nulls with 0s




def get_multiples(x):
    '''we have multiples of 24 days to work with, so we can't include the 
    last few days of the year. January 13th of each year is a window that 
    starts on January 1 and ends on January 24 (24 day period), and this 
    gets every 24 day period of that year. '''
    year = x.year
    multiple = (date(year,1,13) - x).days % 24
    if multiple == 0 and x >= date(year,1,13) and x < date(year+1,1,1):
        return 1
    else: 
        return 0

df['include'] = df.rolling_date.apply(get_multiples)

df['year'] = df.rolling_date.apply(lambda x: x.year)


# -
df.head()

df[['ais_fishing',
 'dark_fishing',
 'ais_nonfishing',
 'dark_nonfishing',
 'ais_nonfishing100',
 'dark_nonfishing100',
 'ais_fishing_i',
 'ais_nonfishing_i',
 'dark_fishing_i',
 'dark_nonfishing_i',
 'ais_nonfishing100_i',
 'dark_nonfishing100_i']] = df[['ais_fishing',
 'dark_fishing',
 'ais_nonfishing',""
 'dark_nonfishing',
 'ais_nonfishing100',
 'dark_nonfishing100',
 'ais_fishing_i',
 'ais_nonfishing_i',
 'dark_fishing_i',
 'dark_nonfishing_i',
 'ais_nonfishing100_i',
 'dark_nonfishing100_i']].fillna(0)

df.head()

df["detections"] = (
    df["ais_fishing"]
    + df["dark_fishing"]
    + df["ais_nonfishing"]
    + df["dark_nonfishing"]
    + df["ais_fishing_i"]
    + df["ais_nonfishing_i"]
    + df["dark_fishing_i"]
    + df["dark_nonfishing_i"]
)

df['fishing'] = df.ais_fishing + df.dark_fishing + df.ais_fishing_i + df.dark_fishing_i
df['nonfishing'] = df.ais_nonfishing + df.dark_nonfishing + df.ais_nonfishing_i + df.dark_nonfishing_i
df['nonfishing100'] = df.ais_nonfishing100 + df.ais_nonfishing100_i + df.dark_nonfishing100 + df.dark_nonfishing100_i


df.head()

df[df.include==1].rolling_date.max()

# +
# df.to_feather("../data/24day_rolling_augmented_v20230816.feather")
# -

df.to_csv("../data/24day_rolling_augmented_v20230816.csv", index=False)

import pandas as pd
df = pd.read_csv("../data/24day_rolling_augmented_v20230816.csv")
df.head()

# +
plt.figure(figsize=(10, 4))
d = df[df.rolling_date >= date(2017, 1, 13)]
d = d[d.rolling_date < date(2021, 12, 20)]
d = d.groupby("rolling_date").sum().reset_index()
plt.plot(
    d.rolling_date.values,
    (d.fishing)
    .rolling(3)
    .median().values,
    label = 'all fishing'
)

plt.plot(
    d.rolling_date.values,
    (d.dark_fishing + d.dark_fishing_i)
    .rolling(3)
    .median().values,
    label = 'dark fishing'
)

plt.plot(
    d.rolling_date.values,
    (d.ais_fishing + d.ais_fishing_i)
    .rolling(3)
    .median().values,
    label = 'ais fishing'
)

plt.legend()
# plt.plot(d.rolling_date, (df2.ais_fishing + df2.dark_fishing + di2.ais_fishing + di2.dark_fishing).rolling(3).median() )
# plt.plot(d.rolling_date, di2.ais_fishing + di2.dark_fishing)

# +
plt.figure(figsize=(10, 4))
d = df[df.rolling_date >= date(2017, 1, 13)]
d = d[d.rolling_date < date(2021, 12, 20)]
d = d.groupby("rolling_date").sum().reset_index()
plt.plot(
    d.rolling_date.values,
    (d.ais_fishing + d.ais_fishing_i)
    .rolling(3)
    .median().values / 
    (d.dark_fishing + d.dark_fishing_i +d.ais_fishing + d.ais_fishing_i )
    .rolling(3)
    .median().values
)

plt.ylim(0,.5)
# plt.plot(
#     d.rolling_date.values,
#     (d.dark_fishing + d.dark_fishing_i)
#     .rolling(3)
#     .median().values,
#     label = 'dark fishing'
# )

# plt.plot(
#     d.rolling_date.values,
#     (d.ais_fishing + d.ais_fishing_i)
#     .rolling(3)
#     .median().values,
#     label = 'ais fishing'
# )
plt.title("Fraction of fishing vessel activity with AIS")
# plt.legend()
# plt.plot(d.rolling_date, (df2.ais_fishing + df2.dark_fishing + di2.ais_fishing + di2.dark_fishing).rolling(3).median() )
# plt.plot(d.rolling_date, di2.ais_fishing + di2.dark_fishing)
# -



# +
plt.figure(figsize=(10, 4))
d = df[df.rolling_date >= date(2017, 1, 13)]
d = d[d.rolling_date < date(2021, 12, 20)]
d = d.groupby("rolling_date").sum().reset_index()
plt.plot(
    d.rolling_date.values,
    (d.dark_nonfishing + d.dark_nonfishing_i + d.ais_nonfishing + d.ais_nonfishing_i)
    .rolling(3)
    .median().values,
    label = 'all nonfishing'
)

plt.plot(
    d.rolling_date.values,
    (d.dark_nonfishing + d.dark_nonfishing_i)
    .rolling(3)
    .median().values,
    label = 'dark nonfishing'
)

plt.plot(
    d.rolling_date.values,
    (d.ais_nonfishing + d.ais_nonfishing_i)
    .rolling(3)
    .median().values,
    label = 'ais nonfishing'
)

plt.legend()
# plt.plot(d.rolling_date, (df2.ais_fishing + df2.dark_fishing + di2.ais_fishing + di2.dark_fishing).rolling(3).median() )
# plt.plot(d.rolling_date, di2.ais_fishing + di2.dark_fishing)

# +
plt.figure(figsize=(10, 4))
d = df[df.rolling_date >= date(2017, 1, 13)]
d = d[d.rolling_date < date(2021, 12, 20)]
d = d.groupby("rolling_date").sum().reset_index()
plt.plot(
    d.rolling_date.values,
    (d.dark_nonfishing100 + d.dark_nonfishing100_i + d.ais_nonfishing100 + d.ais_nonfishing100_i )
    .rolling(3)
    .median().values,
    label = 'all nonfishing 100'
)

plt.plot(
    d.rolling_date.values,
    (d.dark_nonfishing100 + d.dark_nonfishing100_i)
    .rolling(3)
    .median().values,
    label = 'dark nonfishing 100'
)

plt.plot(
    d.rolling_date.values,
    (d.ais_nonfishing100 + d.ais_nonfishing100_i)
    .rolling(3)
    .median().values,
    label = 'ais nonfishing 100'
)

plt.legend()
# plt.plot(d.rolling_date, (df2.ais_fishing + df2.dark_fishing + di2.ais_fishing + di2.dark_fishing).rolling(3).median() )
# plt.plot(d.rolling_date, di2.ais_fishing + di2.dark_fishing)
# -

d.head()


d = df[df.include == 1]
d.rolling_date.max()

d.columns

# +
plt.figure(figsize=(10, 4))
d = df[df.include == 1]
d = d.drop(columns=['rolling_date', 'index', 'eez_iso3'])
d = d.groupby(["year"]).sum().reset_index()
d = d.groupby("year").mean().reset_index()
d["date"] = d.year.apply(lambda x: date(x, 7, 1))
plt.plot(
    d.date.values,
    d.fishing.values,
    label="all fishing",
)
plt.plot(
    d.date.values,
    (d.dark_fishing + d.dark_fishing_i).values,
    label="dark fishing",
)

plt.plot(d.date.values, (d.ais_fishing + d.ais_fishing_i).values, label="ais fishing")

plt.legend()

# +
fishing_prepandemic = d[d.year.isin([2018,2019])].fishing.mean()
fishing_pandemic = d[d.year.isin([2020,2021])].fishing.mean()

fishing_pandemic/fishing_prepandemic
# -

nonfishing_prepandemic = d[d.year.isin([2018,2019])].nonfishing.mean()
nonfishing_pandemic = d[d.year.isin([2020,2021])].nonfishing.mean()
nonfishing_pandemic/nonfishing_prepandemic

# +
plt.figure(figsize=(10, 4))
d = df[(df.include == 1)&(df.eez_iso3=="CHN")]
d = d.drop(columns=['rolling_date', 'index', 'eez_iso3'])
d = d.groupby(["year"]).sum().reset_index()
d["date"] = d.year.apply(lambda x: date(x, 7, 1))
plt.plot(
    d.date.values,
    (d.ais_fishing + d.dark_fishing + d.ais_fishing_i + d.dark_fishing_i).values,
    label="all fishing",
)
plt.plot(
    d.date.values,
    (d.dark_fishing + d.dark_fishing_i).values,
    label="dark fishing",
)

plt.plot(d.date.values, (d.ais_fishing + d.ais_fishing_i).values, label="ais fishing")

plt.legend()
d
# -

fishing_prepandemic = d[d.year.isin([2018,2019])].fishing.mean()
fishing_pandemic = d[d.year.isin([2020,2021])].fishing.mean()
fishing_pandemic/fishing_prepandemic

nonfishing_prepandemic = d[d.year.isin([2018,2019])].nonfishing.mean()
nonfishing_pandemic = d[d.year.isin([2020,2021])].nonfishing.mean()
nonfishing_pandemic/nonfishing_prepandemic

nonfishing_prepandemic = d[d.year.isin([2017])].nonfishing.mean()
nonfishing_pandemic = d[d.year.isin([2021])].nonfishing.mean()
nonfishing_pandemic/nonfishing_prepandemic

nonfishing_prepandemic = d[d.year.isin([2017,2018,2019])].nonfishing.mean()
nonfishing_pandemic = d[d.year.isin([2020,2021])].nonfishing.mean()
nonfishing_pandemic/nonfishing_prepandemic

# +
plt.figure(figsize=(10, 4))
d = df[(df.include == 1)&(df.eez_iso3!="CHN")]
d = d.drop(columns=['rolling_date', 'index', 'eez_iso3'])
d = d.groupby("year").mean().reset_index()
d["date"] = d.year.apply(lambda x: date(x, 7, 1))
plt.plot(
    d.date.values,
    (d.ais_fishing + d.dark_fishing + d.ais_fishing_i + d.dark_fishing_i).values,
    label="all fishing",
)
plt.plot(
    d.date.values,
    (d.dark_fishing + d.dark_fishing_i).values,
    label="dark fishing",
)

plt.plot(d.date.values, (d.ais_fishing + d.ais_fishing_i).values, label="ais fishing")

plt.legend()
# -

fishing_prepandemic = d[d.year.isin([2018,2019])].fishing.mean()
fishing_pandemic = d[d.year.isin([2020,2021])].fishing.mean()
fishing_pandemic/fishing_prepandemic

nonfishing_prepandemic = d[d.year.isin([2018,2019])].nonfishing.mean()
nonfishing_pandemic = d[d.year.isin([2020,2021])].nonfishing.mean()
nonfishing_pandemic/nonfishing_prepandemic


