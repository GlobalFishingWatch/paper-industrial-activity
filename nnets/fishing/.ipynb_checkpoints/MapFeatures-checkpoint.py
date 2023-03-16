# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Make Maps of Features
#
#  - vessel length, 
#  - followed by hours of cargo or tanker per km2, 
#  - density of detections averaged over a 20th of a degree, 
#  - distance from port, 
#  - average vessel length over a 20th of a degree, 
#  - ocean depth, 
#  - distance from shore, 
#  - slope of the sea floor, 
#  - the standard deviations of vessel length over a 20th of a degree, over a 5-km radius, and over a 10-km radius.
#

# 20th of a degree, SAR detection raster
q = '''select lat_index, lon_index, 
detections_km2_5km, 
length_m_ave_5km,
length_m_stddev_5km  
from proj_global_sar.vessel_density_statistics_2017_2021 '''

# +
# distance from port and shore

q = '''distance_from_shore_table  as (
select 
  distance_from_shore_m,
  floor(lat*100) as lat_index,
  floor(lon*100) as lon_index,
from 
  `pipe_static.distance_from_shore`
),

distance_from_port_table as (
select 
  distance_from_port_m,
  floor(lat*100) as lat_index,
  floor(lon*100) as lon_index
from
  `pipe_static.distance_from_port_20201105`
)'''

# +
# depth at 100th of a degree

q = '''select round(lon*100) lon_index,
round(lat*100) lat_index,
elevation_m

 from `world-fishing-827.pipe_static.bathymetry` '''
# -

# rasters to make:
#
# at 20th of a degree:
#  - detections_km2_5km
#  - length_m_ave_5km
#  - length_m_stddev_5km
#
# at 100th of a degree:
#  - distance_from_shore_m
#  - distance_from_port_m
#  - elevation_m

 


