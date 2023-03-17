# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: dark_targets
#     language: python
#     name: python3
# ---

# %%
import os
from datetime import datetime

import cartopy
import matplotlib as mpl
import matplotlib.cbook as cbook
import matplotlib.colors as mpcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")

import pycountry
from pycountry_convert import (
    country_alpha2_to_continent_code,
    country_name_to_country_alpha2)
    
import pyseas.cm
import pyseas.contrib as psc
import pyseas.maps as psm


import prj_global_sar_infra.infra_modules as im

# %load_ext autoreload
# %autoreload 2

# %%
oil_bplot = pd.read_gbq(
'''
with 

all_oil_polygons as (
  select 
    ST_GEOGFROMTEXT(geometry, make_valid => TRUE) as geometry  
  from 
  proj_global_sar.oil_areas),

reclassified_infra as (
  select 
  id, date, lon, lat,
  if( label = 'unknown', 'other', label) as label
  from
  `world-fishing-827.proj_global_sar.infrastructure_reclassified_v20221206`
),

detections_labeled as (
SELECT 
  detect_id,
  detect_lon as lon,
  detect_lat as lat,
  extract( date from midpoint) detect_date,
  if(b.label is null, c.label, b.label) label,
  if( 
     array_length(ISO_TER1)>0, 
     if(ISO_TER1[ordinal(1)]="", "None",ISO_TER1[ordinal(1)] ),   
    'None') iso3
FROM 
  `world-fishing-827.proj_global_sar.infrastructure_repeat_cat_6m_v20220805` a
left join
  reclassified_infra b
on (detect_lon = lon and detect_lat = lat)
left join (select detect_id, label 
from `proj_global_sar.composite_ids_labeled_v20220708`) c
using (detect_id)
where detect_id is not null
and not (detect_lat between -57.8 and -35.05 and detect_lon between -87.98 and -52.54)  -- argentina_chile 
and not (detect_lat between -58.03 and -50.4 and detect_lon between -45.38 and -27.77)  -- south_atlantic 
and not (detect_lat between -46.1 and -33.7 and detect_lon between 136.9 and 151.4)  -- se_australia 
and not (detect_lat between 60.6 and 71.5 and detect_lon between -8.3 and 16.0)  -- norway_s 
and not (detect_lat between 65.96 and 90 and detect_lon between -4.96 and 30.0)  -- norway_n 
and not (detect_lat between 50.8 and 90 and detect_lon between -118.2 and -48.6)  -- canada_ne 
and not (detect_lat between 65.2 and 90 and detect_lon between -178.0 and -110.9)  -- alaska_n 
and not (detect_lat between 62.4 and 90 and detect_lon between 33.3 and 179.1)  -- russia_n 
and not (detect_lat between 8.8958 and 11.0153 and detect_lon between -72.2486 and -70.8539)  -- lake_maracaibo 
),

final_detections as (
  select * from detections_labeled
  where label in ('oil', 'probable_oil')
),

oil_region as (
select 
  detect_id,
  detect_date,
  label,
  iso3,
  "inside_all_oil_polygons" as region,
from 
  final_detections
cross join
  all_oil_polygons
where  
  st_contains(geometry,
             ST_GEOGPOINT(lon,lat))
),

detections_outside_regions as (
select 
  detect_id,
  detect_date,
  label,
  iso3,
  "outside_all_oil_polygons" as region,
from 
  final_detections
where 
 detect_id not in (select detect_id from oil_region)
),

unioned
 as (
select * from oil_region
union all
select * from detections_outside_regions),

oil_detects as (
select * except(iso3, region),iso3, region from
( select
detect_date,
sum(if(label='oil',1,0)) oil,
sum(if(label='probable_oil',1,0)) probable_oil,
iso3,
region
from unioned
group by detect_date, region, iso3
order by detect_date)),

detect_eez as (
select * from oil_detects
where extract (year from detect_date) = 2021
and region = 'inside_all_oil_polygons')

select 
distinct
iso3,
PERCENTILE_CONT(oil, 0.5) OVER(partition by iso3) AS oil_count_median,
PERCENTILE_CONT(probable_oil, 0.5) OVER(partition by iso3) AS probable_oil_count_median
from 
detect_eez
order by oil_count_median desc
''')

# %%
wind_bplot = pd.read_gbq(
'''
with 

all_wind_polygons as (
  select 
    ST_GEOGFROMTEXT(geometry, make_valid => TRUE) as geometry  
  from 
  `world-fishing-827.proj_global_sar.infra_wind_regions`),

reclassified_infra as (
  select 
  id, date, lon, lat,
  if( label = 'unknown', 'other', label) as label
  from
  `world-fishing-827.proj_global_sar.infrastructure_reclassified_v20221206`
),

detections_labeled as (
SELECT 
  detect_id,
  detect_lon as lon,
  detect_lat as lat,
  extract( date from midpoint) detect_date,
  if(b.label is null, c.label, b.label) label,
  if( 
     array_length(ISO_TER1)>0, 
     if(ISO_TER1[ordinal(1)]="", "None",ISO_TER1[ordinal(1)] ),   
    'None') iso3
FROM 
  `world-fishing-827.proj_global_sar.infrastructure_repeat_cat_6m_v20220805` a
left join
  reclassified_infra b
on (detect_lon = lon and detect_lat = lat)
left join (select detect_id, label 
from `proj_global_sar.composite_ids_labeled_v20220708`) c
using (detect_id)
where detect_id is not null
and not (detect_lat between -57.8 and -35.05 and detect_lon between -87.98 and -52.54)  -- argentina_chile 
and not (detect_lat between -58.03 and -50.4 and detect_lon between -45.38 and -27.77)  -- south_atlantic 
and not (detect_lat between -46.1 and -33.7 and detect_lon between 136.9 and 151.4)  -- se_australia 
and not (detect_lat between 60.6 and 71.5 and detect_lon between -8.3 and 16.0)  -- norway_s 
and not (detect_lat between 65.96 and 90 and detect_lon between -4.96 and 30.0)  -- norway_n 
and not (detect_lat between 50.8 and 90 and detect_lon between -118.2 and -48.6)  -- canada_ne 
and not (detect_lat between 65.2 and 90 and detect_lon between -178.0 and -110.9)  -- alaska_n 
and not (detect_lat between 62.4 and 90 and detect_lon between 33.3 and 179.1)  -- russia_n 
and not (detect_lat between 8.8958 and 11.0153 and detect_lon between -72.2486 and -70.8539)  -- lake_maracaibo 
),

final_detections as (
  select * from detections_labeled
  where label in ('wind', 'probable_wind')
),

wind_region as (
select 
  detect_id,
  detect_date,
  label,
  iso3,
  "inside_all_wind_polygons" as region,
from 
  final_detections
cross join
  all_wind_polygons
where  
  st_contains(geometry,
             ST_GEOGPOINT(lon,lat))
),

detections_outside_regions as (
select 
  detect_id,
  detect_date,
  label,
  iso3,
  "outside_all_wind_polygons" as region,
from 
  final_detections
where 
 detect_id not in (select detect_id from wind_region)
),

unioned
 as (
select * from wind_region
union all
select * from detections_outside_regions),

wind_detects as (
select * except(iso3, region),iso3, region from
( select
detect_date,
sum(if(label='wind',1,0)) oil,
sum(if(label='probable_wind',1,0)) probable_oil,
iso3,
region
from unioned
group by detect_date, region, iso3
order by detect_date)),

detect_eez as (
select * from wind_detects
where extract (year from detect_date) = 2021
and region = 'inside_all_wind_polygons')

select 
distinct
iso3,
PERCENTILE_CONT(oil, 0.5) OVER(partition by iso3) AS wind_count_median,
PERCENTILE_CONT(probable_oil, 0.5) OVER(partition by iso3) AS probable_wind_count_median
from 
detect_eez
order by wind_count_median desc
''')

# %%
oil_bplot = oil_bplot[:15]
wind_bplot = wind_bplot[:6]

# %%
infra_plot = pd.merge(oil_bplot, wind_bplot, how='outer')

# %%
infra_plot

# %%
wind_bplot

# %%
fig1, axs = plt.subplots(ncols=1, nrows=2, gridspec_kw={'height_ratios': [2, 1]},figsize=(8,11))
sns.set(style = 'whitegrid')

b1 = oil_bplot.set_index('iso3').plot(
    kind="barh", stacked=True, width=0.8, 
    color = ['#003f5c', '#ffa600'], ax = axs[0],).invert_yaxis()


for x, y in enumerate(oil_bplot.sum(axis=1).astype(int)):
    axs[0].annotate(y, (y, x), ha='left', va='center', size=14, xytext=(3, 0),
    color = '#003f5c', textcoords='offset points')


b2 = wind_bplot.set_index('iso3').plot(ax=axs[1],
    kind="barh", stacked=True, width=0.65, 
    color = ['#003f5c', '#ffa600']).invert_yaxis()

for x, y in enumerate(wind_bplot.sum(axis=1).astype(int)):
    axs[1].annotate(y, (y, x), ha='left', va='center', size=14, xytext=(3, 0),
    color = '#003f5c', textcoords='offset points')

axs[0].set_title("Oil Infrastructure 2021", fontsize = 16)
axs[1].set_title("Wind Infrastructure 2021", fontsize = 16)
for i in axs:
    i.spines['top'].set_visible(False)
    i.spines['right'].set_visible(False)
    i.spines['bottom'].set_visible(False)
    i.spines['left'].set_visible(False)
    i.set_ylabel('')
    i.grid(visible=False, axis = 'y')
    i.grid(visible=True, axis = 'x', color = 'gainsboro')
    # i.set_ylim(15-0.5, -0.5)

axs[0].get_legend().remove()
axs[1].get_legend().remove()
axs[0].legend(labels = ['oil ', 'probable oil'], frameon=False, fontsize = 15, loc = 'lower right')
axs[1].legend(labels = ['wind ', 'probable wind'], frameon=False, fontsize = 15, loc = 'lower right')


axs[0].tick_params(axis='both', which='major', labelsize=14)
axs[1].tick_params(axis='both', which='major', labelsize=14)
plt.tight_layout()
# import matplotlib.pyplot as plt

plt.savefig('barchart_oil_wind_eez.png', bbox_inches="tight", dpi = 300)
plt.show()



# %%
