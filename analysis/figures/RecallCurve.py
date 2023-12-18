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

# # RecallCurve
#
# In this notebook, we estimate the recall of our Sentinel-1 detection algorithm as a function of a vessel's length and its distance to the neareast other vessel with AIS. 
#
# To estimate this, we analyze all vessels that definitely appeared within SAR scenes (that is, at hte moment of the image, the vessel was definitely within the scene footprint) and which had a position within two minutes of the scene, such that we know with high accuracy where the vessel is and, if we detected it, it should match to a SAR detection.
#
#

import pandas as pd
import matplotlib.pyplot as plt
import proplot
import numpy as np
import matplotlib as mpl
import proplot
import matplotlib.dates as mdates
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# +
# q = '''with 
# vessel_info as (
# select ssvid,
# best.best_length_m,
# from `world-fishing-827.gfw_research.vi_ssvid_v20220401`
# where activity.overlap_hours < 24 
# -- and registry_info.best_known_length_m is not null
# ),


# definitely_in_scene as 
# (
# select scene_id, ssvid from proj_global_sar.likelihood_inside 
# where prob_inside > .99
# ),


# extrap as (
# select * from 
# (SELECT 
# ssvid,
# least(abs(delta_minutes1),abs(delta_minutes2)) delta_minutes,
# extract(year from _partitiontime) year,
# st_geogpoint(likely_lon, likely_lat) pos,
# within_footprint_5km,
# within_footprint_1km,
# row_number() over(partition by ssvid, scene_id order by rand()) as row,
#  scene_id 
#  FROM `world-fishing-827.proj_sentinel1_v20210924.detect_foot_ext_ais`
#  join definitely_in_scene
#  using(ssvid,scene_id)
#   WHERE DATE(_PARTITIONTIME) 
# between 
# "2017-01-01"
# and
# "2021-12-31" 
# and source = "AIS"
# and within_footprint_5km)
# where row = 1

# ),

# distance_to_closest as 
# (select a.ssvid, a.delta_minutes, a.year, a.within_footprint_5km,
# a.within_footprint_1km,
#  scene_id ,
#  min(st_distance(a.pos,b.pos)) min_distance_m
#  from extrap a
#  join
#  extrap b
#  using(scene_id)
#  where a.ssvid != b.ssvid
#  group by ssvid, delta_minutes, year, within_footprint_5km, 
#  within_footprint_1km, scene_id),

#  vessels_in_scenes as (
# SELECT 
# ssvid,
# min_distance_m,
# delta_minutes,
# best_length_m,
# within_footprint_5km,
# within_footprint_1km,
#  scene_id 
#  FROM distance_to_closest
# join
# vessel_info
# using(ssvid)
# where delta_minutes < 2
# ),

# match_table as (
# select ssvid, scene_id, presence, score, source
#  from `world-fishing-827.proj_sentinel1_v20210924.detect_scene_match`
# join
# `world-fishing-827.proj_sentinel1_v20210924.detect_scene_pred_*`
# using(detect_id)
# where presence > .7
# and DATE(_PARTITIONTIME) 
# between "2017-01-01" and "2021-12-31"
# and _table_suffix between "20170101" and "20211231"
# )



# select 
# case when min_distance_m < 1000 then 
# floor(min_distance_m/100)*100 
# else 1000 end min_distance_m,
# count(*) vessels,
# sum(if(b.ssvid is not null and b.score > 1e-1, 1,0)) matched1,
# sum(if(b.ssvid is not null and b.score > 1e-2, 1,0)) matched2,
# sum(if(b.ssvid is not null and b.score > 1e-3, 1,0)) matched3,
# sum(if(b.ssvid is not null and b.score > 1e0, 1,0)) matched0,

# case when best_length_m < 30 
#  then floor(best_length_m/2)*2 + 1 
# when best_length_m between 30 and 59.9999999 
#   then floor(best_length_m/4)*4 + 2 
# when best_length_m > 60  and best_length_m < 200
#   then floor(best_length_m/20)*20 + 10 
# when best_length_m >= 200 then 200
# else -20 
#    end length_m

# from 
# vessels_in_scenes a
# left join
# match_table b
# using(ssvid, scene_id)
# group by length_m, min_distance_m
# order by length_m'''

# df = pd.read_gbq(q)
# -


df = pd.read_csv('../data/recal_curve.csv.zip')


# +

df['frac_matched0'] =  df.matched0/df.vessels
df['frac_matched1'] =  df.matched1/df.vessels
df['frac_matched2'] =  df.matched2/df.vessels
df['frac_matched3'] =  df.matched3/df.vessels


# +
mindists = df.min_distance_m.unique()
mindists.sort()

plt.figure(figsize=(8,4))
for min_distance_m in mindists:
    d = df[df.min_distance_m == min_distance_m]
    plt.scatter(d.length_m, d.frac_matched3, label = min_distance_m)
plt.legend()
plt.xlim(0,210)
plt.xlabel("length, m")
plt.ylabel("fraction detected")
plt.title("recall as a function of length and vessel spacing")

# -

plt.figure(figsize=(8,4),facecolor='white')
d = df.groupby('length_m').sum()
d['frac_matched3'] =  d.matched3/d.vessels
plt.scatter(np.array(d.index.values),np.array(d.frac_matched3.values), label = "all vessels")
plt.plot(np.array(d.index.values[1:]),np.array(d.frac_matched3.values[1:]))
d = df[df.min_distance_m == 1000]
plt.scatter(d.length_m.values, d.frac_matched3.values, label = "well spaced vessels")
plt.plot(np.array(d.length_m.values[1:]), np.array(d.frac_matched3.values[1:]))
plt.xlim(0,200)
plt.legend(frameon=False)
plt.xlabel("Length, m")
plt.ylabel("Fraction detected")
plt.savefig("./recall.png")


df.matched3.sum()

# +
## upload to bigquery to use in our recall estimates
# df.to_gbq("proj_global_sar.s1recall", if_exists='replace')
# -
# # What fraction of vessels in the scenes are more than 1km spaced?

# +
# q = '''with 
# vessel_info as (
# select ssvid,
# best.best_length_m,
# from `world-fishing-827.gfw_research.vi_ssvid_v20220401`
# where activity.overlap_hours < 24 
# -- and registry_info.best_known_length_m is not null
# ),


# likelihood_in_scene as 
# (
# select prob_inside,scene_id, ssvid from proj_global_sar.likelihood_inside 
# ),


# extrap as (
# select * from 
# (SELECT 
# ssvid,
# prob_inside,
# least(abs(delta_minutes1),abs(delta_minutes2)) delta_minutes,
# extract(year from _partitiontime) year,
# st_geogpoint(likely_lon, likely_lat) pos,
# within_footprint_5km,
# within_footprint_1km,
# row_number() over(partition by ssvid, scene_id order by rand()) as row,
#  scene_id 
#  FROM `world-fishing-827.proj_sentinel1_v20210924.detect_foot_ext_ais`
#  join likelihood_in_scene
#  using(ssvid,scene_id)
#   WHERE DATE(_PARTITIONTIME) 
# between 
# "2017-01-01"
# and
# "2021-12-31" 
# and within_footprint
# and source = "AIS")
# where row = 1

# ),

# distance_to_closest as 
# (select a.ssvid, a.delta_minutes, a.year, a.within_footprint_5km,
# a.within_footprint_1km,
#  scene_id ,
#  min(st_distance(a.pos,b.pos)) min_distance_m
#  from extrap a
#  join
#  extrap b
#  using(scene_id)
#  where a.ssvid != b.ssvid
#  group by ssvid, delta_minutes, year, within_footprint_5km, 
#  within_footprint_1km, scene_id),

#  vessels_in_scenes as (
# SELECT 
# ssvid,
# min_distance_m,
# delta_minutes,
# best_length_m,
# within_footprint_5km,
# within_footprint_1km,
#  scene_id 
#  FROM distance_to_closest
# join
# vessel_info
# using(ssvid)
# where delta_minutes < 2
# )


# select 
# sum(if(min_distance_m >= 1000,1,0))/count(*) 
# from 
# vessels_in_scenes '''


# -


vessels_in_scenes = pd.read_csv('../data/vessels_in_scenes.csv.zip')

sum(vessels_in_scenes.min_distance_m >= 1000) / len(vessels_in_scenes)


