# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## This code is used to inspect the wind infrastructure detectections by creating regional and global time series plots

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

# %load_ext autoreload
# %autoreload 2

# %%
import sys
sys.path.append('../analyses_functions') 
from infra_modules import *
elimination_string = messy_areas()

# %% [markdown]
# ### Global timeseries of wind detections

# %%
q = f'''

with 

all_wind_polygons as (
  select 
    ST_GEOGFROMTEXT(geometry, make_valid => TRUE) as geometry  
  from 
  `proj_global_sar.infra_wind_regions`),

reclassified_infra as (
  select 
  id, date, lon, lat,
  if( label = 'unknown', 'other', label) as label
  from
  `project-id.proj_global_sar.infrastructure_reclassified_v20230222`
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
  `project-id.proj_global_sar.infrastructure_repeat_cat_6m_v20220805` a
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
  where label in ('oil', 'probable_oil', 'wind', 'probable_wind', 'other')),

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
and label in ('wind', 'probable_wind')
),

detections_outside_wind_regions as (
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

unioned_wind
 as (
select * from wind_region
union all
select * from detections_outside_wind_regions),

wind_detects as (
select detect_date, wind, probable_wind, region from
( select
detect_date,
sum(if(label='wind',1,0)) wind,
sum(if(label='probable_wind',1,0)) probable_wind,
region
from unioned_wind
group by detect_date, region
order by detect_date)),


wind_detect_eez as (
select * except (region) from wind_detects
-- where extract (year from detect_date) = 2021
where region = 'inside_all_wind_polygons')

select detect_date, wind from 
wind_detect_eez
'''
wind_time = pd.read_gbq(q)

# %%
wind_time.min()

# %%
wind_change = (((wind_time['wind'].iloc[-1] - wind_time['wind'].iloc[0]) / wind_time['wind'].iloc[0]))* 100

# %%
print(f'Wind percent change = {round(wind_change,2)}')

# %%
mpl.rc('xtick', labelsize=16) 
mpl.rc('ytick', labelsize=16) 
fig, ax = plt.subplots(figsize=(20, 6))
# ax.plot(eez["date"].tolist(), eez["count"].tolist())
wind_time.plot(x = 'detect_date', y = 'wind', kind="line", ax=ax, label='wind', linewidth = 4)

plt.legend(fontsize=20)
ax.set_ylim([0, wind_time["wind"].max()+1000])

plt.title(f"Infrastructure Detected", fontsize = 18)
plt.savefig('wind_timeseries', dpi = 300)
plt.show()

# %% [markdown]
# ## Global timeseries using median monthly detections

# %%
median_month = pd.read_gbq(
'''
with 

all_wind_polygons as (
  select 
    ST_GEOGFROMTEXT(geometry, make_valid => TRUE) as geometry  
  from 
  `proj_global_sar.infra_wind_regions`),

reclassified_infra as (
  select 
  id, date, lon, lat,
  if( label = 'unknown', 'other', label) as label
  from
  `project-id.proj_global_sar.infrastructure_reclassified_v20230222`
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
  `project-id.proj_global_sar.infrastructure_repeat_cat_6m_v20220805` a
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
  where label in ('oil', 'probable_oil', 'wind', 'probable_wind', 'other')),

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
and label in ('wind', 'probable_wind')
),

detections_outside_wind_regions as (
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

unioned_wind
 as (
select * from wind_region
union all
select * from detections_outside_wind_regions),

wind_detects as (
select detect_date, wind, probable_wind, region from
( select
detect_date,
sum(if(label='wind',1,0)) wind,
sum(if(label='probable_wind',1,0)) probable_wind,
region
from unioned_wind
where iso3 = 'GBR'
group by detect_date, region
order by detect_date)),


wind_detect_eez as (
select * except (region) from wind_detects
where region = 'inside_all_wind_polygons')


select 
distinct
DATE(extract( year from detect_date), extract( month from detect_date),01) year_month,
PERCENTILE_CONT(wind, 0.5) OVER(partition by extract( year from detect_date), extract( month from detect_date)) +
PERCENTILE_CONT(probable_wind, 0.5) OVER(partition by extract( year from detect_date), extract( month from detect_date)) AS wind_count_median
from 
wind_detect_eez
order by year_month desc

'''


)

# %%
median_month

# %%
mpl.rc('xtick', labelsize=16) 
mpl.rc('ytick', labelsize=16) 
fig, ax = plt.subplots(figsize=(20, 6))
# ax.plot(eez["date"].tolist(), eez["count"].tolist())
median_month.plot(x = 'year_month', y = 'wind_count_median', kind="line", ax=ax, label='wind', linewidth = 4)

plt.legend(fontsize=20)
ax.set_ylim([0, 3000])

plt.title(f"monthly Infrastructure Detected", fontsize = 18)
plt.savefig('wind_timeseries_monthly_median_gbr', dpi = 300)
plt.show()

# %% [markdown]
# ## Fit a line to the data

# %%
# create an evenly spaced line that represents the dates
wind_x_values = list(range(1,62))
wind_y_values = wind_time.loc[:,'wind'].to_numpy().astype(float)

# %%
t = pd.date_range(start='2017-01-01',
                  end='2031-12-31',
                  periods=180)

# %%
new_x = list(range(1,181))

# %%
polynomial = np.polyfit(wind_x_values, wind_y_values, 2)
wind_poly_eqn = np.poly1d(polynomial)
wind_y_hat = wind_poly_eqn(wind_x_values)
new_y_hat = wind_poly_eqn(new_x)

# %%
import datetime
plt.figure(figsize=(20,6))

plt.plot(t, new_y_hat)

plt.plot(list(wind_time.loc[:, "detect_date"]), list(wind_time.loc[:,"wind"]), "ro")
plt.plot(list(wind_time.loc[:, "detect_date"]), wind_y_hat, label = 'CHN Wind')



# plt.legend(fontsize=14)
# plt.title('CHN and GBR wind fitted line')
plt.ylabel('Count', size = 18)
plt.xlabel('Date', size = 18)
plt.xlim([datetime.date(2017, 1, 1), datetime.date(2031, 12, 31)])
# plt.ylim(0,15000)
plt.savefig("quadratic_wind.png", dpi = 300)

# %% [markdown]
# ## Percent of wind in EEZ

# %%
q = '''

with 

all_wind_polygons as (
  select 
    ST_GEOGFROMTEXT(geometry, make_valid => TRUE) as geometry  
  from 
  `proj_global_sar.infra_wind_regions`),

reclassified_infra as (
  select 
  id, date, lon, lat,
  if( label = 'unknown', 'other', label) as label
  from
  `project-id.proj_global_sar.infrastructure_reclassified_v20230222`
),

not_infra as (
SELECT
distinct
  detect_id,
FROM
  (select detect_id,
detect_lon,
detect_lat,
FROM 
`project-id.proj_global_sar.infrastructure_repeat_cat_6m_v20220805` 
where midpoint = '2021-10-01')
CROSS JOIN
  (SELECT
  ST_GEOGFROMTEXT(geometry, make_valid => TRUE) AS geometry
FROM
  `proj_global_sar.skytruth_review_error_not_infra`)
WHERE
  ST_CONTAINS(geometry, ST_GEOGPOINT(detect_lon, detect_lat))),

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
  `project-id.proj_global_sar.infrastructure_repeat_cat_6m_v20220805` a
left join
  reclassified_infra b
on (detect_lon = lon and detect_lat = lat)
left join (select detect_id, label 
from `proj_global_sar.composite_ids_labeled_v20220708`) c
using (detect_id)
where detect_id is not null
and detect_id not in (select detect_id from not_infra where detect_id is not null)
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
and label in ('wind', 'probable_wind')
),

detections_outside_wind_regions as (
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

unioned_wind
 as (
select * from wind_region
union all
select * from detections_outside_wind_regions),

wind_detects as (
select * except(iso3, region),iso3, region from
( select
detect_date,
sum(if(label='wind',1,0)) wind,
sum(if(label='probable_wind',1,0)) probable_wind,
iso3,
region
from unioned_wind
group by detect_date, region, iso3
order by detect_date)),

wind_detect_eez as (
select * except (region) from wind_detects
where extract (year from detect_date) = 2021
and region = 'inside_all_wind_polygons'),

final2 as (
select 
distinct
iso3,
PERCENTILE_CONT(wind, 0.5) OVER(partition by iso3) AS wind_count_median,
PERCENTILE_CONT(probable_wind, 0.5) OVER(partition by iso3) AS probable_wind_count_median
from 
wind_detect_eez
order by wind_count_median desc)


select iso3, 
wind_count_median + probable_wind_count_median as total_wind,
round((wind_count_median + probable_wind_count_median) * 100 / sum((wind_count_median + probable_wind_count_median)) over ()) as perc 
from final2
order by perc desc

'''

eez_perc = pd.read_gbq(q)

# %%
eez_perc

# %% [markdown]
# ### Timeseries of wind in different eezs

# %%
q = f'''
with 

all_wind_polygons as (
  select 
    ST_GEOGFROMTEXT(geometry, make_valid => TRUE) as geometry  
  from 
  `proj_global_sar.infra_wind_regions`),

reclassified_infra as (
  select 
  id, date, lon, lat,
  if( label = 'unknown', 'other', label) as label
  from
  `project-id.proj_global_sar.infrastructure_reclassified_v20230222`
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
    'None') iso3,
FROM 
  `project-id.proj_global_sar.infrastructure_repeat_cat_6m_v20220805` a
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
  where label in ('oil', 'probable_oil', 'wind', 'probable_wind', 'other')),


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
and label in ('wind', 'probable_wind')
),

detections_outside_wind_regions as (
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

unioned_wind
 as (
select * from wind_region
union all
select * from detections_outside_wind_regions),

wind_detects as (
  select
detect_date,
sum(if(label='wind',1,0)) wind,
sum(if(label='probable_wind',1,0)) probable_wind,
iso3,
region
from unioned_wind
group by detect_date, region, iso3
order by detect_date),

wind_detect_eez as (
select * except (region) 
from wind_detects
where region = 'inside_all_wind_polygons')


select 
distinct
iso3,
DATE(extract( year from detect_date), extract( month from detect_date),01) year_month,
PERCENTILE_CONT(wind, 0.5) OVER(partition by iso3, extract( year from detect_date), extract( month from detect_date)) +
PERCENTILE_CONT(probable_wind, 0.5) OVER(partition by iso3, extract( year from detect_date), extract( month from detect_date)) AS wind_count_median
from 
wind_detect_eez
order by iso3, year_month desc


'''
eez = pd.read_gbq(q)

# %%
eez

# %%
eez_grouped = eez[['iso3', 'wind_count_median']].groupby('iso3').sum().sort_values('wind_count_median', ascending=False)[:10].reset_index()
top_eez = eez_grouped['iso3']

# %%
top_eez

# %%
for i in top_eez:
    df = eez.loc[eez['iso3'] == i]

    wind_change = ((df['wind_count_median'].iloc[-0] - df['wind_count_median'].iloc[-1]) / df['wind_count_median'].iloc[-1]) * 100

    print (i)
    
    print(f'Wind percent change = {round(wind_change,2)}')

# %%
for i in top_eez:
    df = eez.loc[eez['iso3'] == i]
    fig, ax = plt.subplots(figsize=(20, 6))
    df.plot(x = 'year_month', y = 'wind_count_median', kind="line", ax=ax, label='wind', linewidth = 4)

    plt.legend(fontsize=20)
    ax.set_ylim([0, df["wind_count_median"].max()+1000])

    plt.title(f"{i} wind time series", fontsize = 18)
    plt.savefig(f'{i}_infra_wind_timeseries', dpi = 300)
    plt.show()

# %%
top_eez_chn_gbr = ['CHN', 'GBR']

# %%
fig, ax = plt.subplots(figsize=(20, 6))

for i in top_eez_chn_gbr:
    df = eez.loc[eez['iso3'] == i]
    df.plot(x = 'year_month', y = 'wind_count_median', kind="line", ax=ax, label=i, linewidth = 4)

    plt.legend(fontsize=20)
    # ax.set_ylim([0, df["count"].max()+1000])

plt.title(f"CHN GBR Wind Infra", fontsize = 18)
plt.savefig(f'chn_gbr_infra_wind_timeseries', dpi = 300)
plt.show()

# %% [markdown]
# ### Smooth the line

# %%
gbr = eez.loc[eez['iso3'] == 'GBR']
chn = eez.loc[eez['iso3'] == 'CHN']

# %%
gbr = gbr[['year_month', 'wind_count_median']].set_index('year_month')
chn = chn[['year_month', 'wind_count_median']].set_index('year_month')

# %%
gbr_wind_rolmean = gbr.rolling(window=3).median() 
chn_wind_rolmean = chn.rolling(window=3).median() 

# %%
fig, ax = plt.subplots(figsize=(16, 4))
ax.plot(gbr_wind_rolmean, label='GBR Wind')
ax.plot(chn_wind_rolmean, label='CHN Wind')
ax.legend()

# %% [markdown]
# ### Summarize by wind region

# %%
q = f'''


with 

all_wind_polygons as (
  select 
  region,
    ST_GEOGFROMTEXT(geometry, make_valid => TRUE) as geometry  
  from 
  `proj_global_sar.infra_wind_regions`),

reclassified_infra as (
  select 
  distinct
  id, lon, lat,
  label
  from
  `project-id.proj_global_sar.infrastructure_reclassified_v20230222`
  where label in ('wind', 'possible_wind')
)


select 
count(*)/ 10828 num ,
  region,
from 
  reclassified_infra
cross join
  all_wind_polygons
where  
  st_contains(geometry,
             ST_GEOGPOINT(lon,lat))
  group by region

'''
wind_reg = pd.read_gbq(q)

# %%
wind_reg

# %%
