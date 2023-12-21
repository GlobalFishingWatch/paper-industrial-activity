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

# # Make the Rolling Chart for North Korea's Western EEZ

import pandas as pd
import matplotlib.pyplot as plt
import proplot as pplt
import pyperclip
from datetime import datetime, timedelta, date
import numpy as np
import matplotlib.dates as mdates
import matplotlib as mpl
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# Get the query templates
import sys
sys.path.append('../utils')
from vessel_queries import *

seen_table = "proj_global_sar.detections_24_w_zeroes_v20230219"
interpolated_table = "proj_global_sar.detections_24_w_interp_v4"

# +
# # select detections with a strict filter removing noise and repeats.
# # Assign matched and unmatched fishing and non-fishing labels
# # based on vessel info table and differing thresholds of rf model results probabilities
# # sum the number of detects for each class by day and eez for an entire year


# q = f"""


# {detections_table},


# final_table as (
# select
#   detect_id,
#   overpasses_24day,
#   date_24,
#   eez_iso3,
#   fishing_score,
#   length_m,
#   case when score > dd_perkm2 and on_fishing_list then "matched_fishing"
#    when score > dd_perkm2 and not on_fishing_list then "matched_nonfishing"
#    when score > dd_perkm2 and on_fishing_list is null  then "matched_unknown"
#    when score < dd_perkm2 then "unmatched" end as matched_category   
# from
#   detections_table a
# left join
#   vessel_info
# using(ssvid)
# left join
#   predictions_table
# using(detect_id)
# where
#   -- important! only places with 70 or more detections... 
#   periods24_with_overpass >= 70
#   and 
#   --------- IMPORTANT ---- ONLY WESTERN EEZ OF NORTH KOREA
#   eez_iso3 = "PRK" 
#   and detect_lon < 127

# )

# select 
#   date(rolling_date) rolling_date,
#   extract(year from rolling_date) year,
#   eez_iso3,
#   sum(if( matched_category = 'matched_fishing', 1/overpasses, 0)) matched_fishing,
#   sum(if( matched_category = 'matched_nonfishing', 1/overpasses, 0)) matched_nonfishing,
#   sum(if( matched_category = 'matched_unknown', 1/overpasses, 0)) matched_unknown,
#   sum(if( matched_category = 'matched_unknown',
#                fishing_score/overpasses, 0)) matched_unknown_likelyfish,
#   sum(if( matched_category = 'matched_unknown',
#                (1-fishing_score)/overpasses, 0)) matched_unknown_likelynonfish,               
#   sum(if( matched_category = 'unmatched', fishing_score/overpasses, 0)) unmatched_fishing,
#   sum(if( matched_category = 'unmatched', (1-fishing_score)/overpasses, 0)) unmatched_nonfishing,
#   sum(1/overpasses) detections,
  
#   sum(if( matched_category = 'unmatched' and length_m > 100,
#           (1-fishing_score)/overpasses_24day, 0)) unmatched_nonfishing_100,
          
#   sum(if( matched_category = 'matched_nonfishing' and length_m > 100,
#              1/overpasses_24day, 0)) +
#   sum(if( matched_category = 'matched_unknown' and length_m > 100,
#                (1-fishing_score)/overpasses_24day, 0)) matched_nonfish_100
# from 
#   final_table
# join
#   proj_global_sar.detections_w_overpasses_rolling24_v20220805 
# using(detect_id)
# group by 
#   eez_iso3, rolling_date, year
# order by rolling_date




# """
# import pyperclip

# pyperclip.copy(q)

# +
# df_detect_nk = pd.read_gbq(q)
# -

df_detect_nk = pd.read_csv('../data/detect_nk.csv.zip',  parse_dates=['rolling_date'])
df_detect_nk['rolling_date'] = pd.to_datetime(df_detect_nk['rolling_date']).dt.date


df_detect_nk.head()

# +
df = df_detect_nk
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



interpolated_table



# +
# q = f'''

# with 

# interp_table as 
# (select * from {interpolated_table}
# where lon_index/200 < 127
# ),


# date_table as 
# -- generate array of tables
# (
# select *,
# timestamp_add( timestamp("2017-01-01"),
#        interval cast( floor(timestamp_diff(timestamp(date),  
#                                     timestamp("2017-01-01"),
#                                      day)/24)*24 as int64) day) as date_24,
#  from unnest(GENERATE_DATE_ARRAY("2017-01-01", "2021-12-31")) as date
# ),


# -- Get the EEZ

# regions as (
# SELECT 
#   regions.eez as eez_array, 
#   -- If there are multiple EEZ values, just get the first one
#   -- Note that this will muddy a bit values in disputed areas
#   if(array_length(regions.eez)>0, regions.eez[ordinal(1)], null) MRGID,
#   gridcode 
# FROM 
#   `project-id.pipe_static.spatial_measures_20201105` 
# ),


# eez_table as (
# select 
#   lat_index, lon_index,
#   ISO_TER1 as eez_iso3
# from 
#   (select 
#     distinct lat_index, lon_index, 
#     format("lon:%+07.2f_lat:%+07.2f", 
#       lon_index/200 +1/400, lat_index/200 + 1/400 
#       ) as gridcode, -- for matching
#    from 
#      interp_table
#   )
# join
#   regions
# using(gridcode)
# join
#   proj_global_sar.eez_info_v11
# using(MRGID)
# where ISO_TER1 = "PRK"
# ),




# smoothed as 

# (select 
#   date,
#   date_24,
#   eez_iso3,
#   lat_index,
#   lon_index,
#   ais_fishing/24 as ais_fishing,
#   ais_nonfishing/24 as ais_nonfishing,
#   dark_fishing/24 as dark_fishing,
#   dark_nonfishing/24 as dark_nonfishing,
#   ais_nonfishing_100/24 as ais_nonfishing100,
#   dark_nonfishing_100/24 as dark_nonfishing100
# from 
#   interp_table 
# join
#   date_table
# using(date_24)
#   join
# eez_table 
#   using (lat_index, lon_index)
# ),


# rolling_date_table as 
# (select 
#   a.date as rolling_date,
#   b.date as date
# from 
#   date_table a
# cross join
#   date_table b
# where 
#   b.date between 
#               timestamp_sub(a.date, interval 12 day) 
#               and timestamp_add(a.date, interval 11 day)
# )

# select 
# rolling_date,
# eez_iso3,
# sum(ais_fishing) as ais_fishing_i,
# sum(ais_nonfishing) as ais_nonfishing_i,
# sum(dark_fishing) as dark_fishing_i,
# sum(dark_nonfishing) as dark_nonfishing_i,
# sum(ais_nonfishing100) as ais_nonfishing100_i,
# sum(dark_nonfishing100) as dark_nonfishing100_i
# from 
# smoothed
# join
# rolling_date_table
# using(date)
# group by rolling_date, eez_iso3
# order by eez_iso3, rolling_date

# '''

# pyperclip.copy(q)


# +
# df_int_nk = pd.read_gbq(q)
# -


df_int_nk = pd.read_csv('../data/int_nk.csv.zip',  parse_dates=['rolling_date'])
df_int_nk['rolling_date'] = pd.to_datetime(df_int_nk['rolling_date']).dt.date


df_int_nk.head()

# +
df_int_nk['eez_iso3']=df_int_nk['eez_iso3'].fillna("None")

# pandas.merge() by Column
df = pd.merge(df,df_int_nk, how='outer',on=['eez_iso3','rolling_date']).reset_index()

# replace nulls with 0s
df = df.fillna(0)

def get_multiples(x):
    '''we have multiples of 24 days to work with, so we can't include the 
    last few days of the year. January 13th of each year is a window that 
    starts on January 1 and ends on January 24 (24 day period), and this 
    gets every 24 day period of that year. '''
    year = x.year
    multiple = (date(year=year,month=1,day=13) - x).days % 24
    if multiple == 0 and x >= date(year=year,month=1,day=13) and x < date(year+1,1,1):
        return 1
    else: 
        return 0

df['include'] = df.rolling_date.apply(get_multiples)

df['year'] = df.rolling_date.apply(lambda x: x.year)

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
# -



# +
fig, ax = plt.subplots(figsize=(10, 4),facecolor="white")


# for year in range(2017,2022):
#     plt.plot([datetime(year,5,1),datetime(year,5,1)],[0,2500],"--", color = "#CCCCCC")
#     plt.plot([datetime(year,8,16),datetime(year,8,16)],[0,2500],"--", color = "#CCCCCC")
    

d = df[df.rolling_date >= date(2017, 1, 13)]
d = d[d.rolling_date < date(2021, 12, 20)]
d = d.groupby("rolling_date").sum().reset_index()

ax.plot(
    d.rolling_date,
    (d.dark_fishing + d.dark_fishing_i+d.ais_fishing + d.ais_fishing_i)
    .rolling(3)
    .mean(),
    label = 'dark fishing'
)



ax.set_ylabel("Vessels")
ax.set_ylim(0,2300)

ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%b'))
for label in ax.get_xticklabels(which='major'):
    label.set(rotation=30, horizontalalignment='right')
plt.savefig("./WNorthKoreaFishing.png",bbox_inches='tight',dpi=300)
# plt.legend()
# plt.plot(d.rolling_date, (df2.ais_fishing + df2.dark_fishing + di2.ais_fishing + di2.dark_fishing).rolling(3).median() )
# plt.plot(d.rolling_date, di2.ais_fishing + di2.dark_fishing)
# -


