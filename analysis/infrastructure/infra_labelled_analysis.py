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
# # This code inspects the labelled infrastructure detectections by creating regional and global time series plots

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
sys.path.append('../utils')
from infra_modules import *

elimination_string = messy_areas()

# %%
#reclassified infrastructure single date composite midpoint - '2021-10-01'
q = '''
SELECT
  lon,
  lat,
  label,
  from
  `project-id.proj_global_sar.infrastructure_reclassified_v20230222` 
where 
  label in('oil', 'wind', 'unknown')
'''
df = pd.read_gbq(q)

# %%
df

# %% [markdown]
# ## Global map of detections by class

# %%
unknown = df.loc[df['label'] == 'unknown']

wind = df.loc[df['label'] == 'wind']

oil = df.loc[df['label'] == 'oil']

# %%
with psm.context(psm.styles.light):
    fig = plt.figure(figsize=(25, 15))
    psm.create_map()
    psm.add_land()
    psm.add_eezs()
    
    plt.scatter(oil.lon, oil.lat, s = 5, color = 'g', transform = psm.identity, label = 'Oil')

    plt.scatter(wind.lon, wind.lat, s = 5, color = 'r', 
                transform = psm.identity, label = 'Wind')

    plt.scatter(unknown.lon, unknown.lat, s = 5, color = 'b', 
                transform = psm.identity, label = 'unknown')

    plt.legend(fontsize = 24, markerscale=6)

    plt.show()
    
    # plt.savefig('infra_map', dpi = 300)

# %% [markdown]
# ## Global timeseries - all infrastructure 

# %%
q = f'''

with 

all_oil_polygons as (
  select 
    ST_GEOGFROMTEXT(geometry, make_valid => TRUE) as geometry  
  from 
  proj_global_sar.oil_areas),

all_wind_polygons as (
  select 
    ST_GEOGFROMTEXT(geometry, make_valid => TRUE) as geometry  
  from 
  `project-id.proj_global_sar.infra_wind_regions`),

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
  where label in ('oil', 'wind', 'other')
),

wind_region as (
select 
  detect_id,
  detect_date,
  lat,
  lon,
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
and label = 'wind'),

oil_region as (
select 
  detect_id,
  detect_date,
  lat,
  lon,
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
and label = 'oil')

select detect_date, count(*) as count from (
select * except(region) from oil_region
union all
select * except(region) from wind_region
union all (select detect_id, detect_date, lat, lon, label, iso3 from final_detections where label = 'other'))
group by 1
order by 1


'''
global_infra = pd.read_gbq(q)

# %%
global_infra

# %%
mpl.rc('xtick', labelsize=16) 
mpl.rc('ytick', labelsize=16) 
fig, ax = plt.subplots(figsize=(20, 6))
# ax.plot(eez["date"].tolist(), eez["count"].tolist())
global_infra.plot(x = 'detect_date', y = 'count', kind="line", ax=ax, linewidth = 4)

plt.legend(fontsize=20)
# ax.set_ylim([0, class_time["count"].max()+1000])

plt.title(f"Global Infrastructure Detections", fontsize = 18)
plt.savefig('global_infra', dpi = 300)
plt.show()

# %% [markdown]
# ## Global timeseries of infra by class

# %%
q = f'''

with 

all_oil_polygons as (
  select 
    ST_GEOGFROMTEXT(geometry, make_valid => TRUE) as geometry  
  from 
  proj_global_sar.oil_areas),

all_wind_polygons as (
  select 
    ST_GEOGFROMTEXT(geometry, make_valid => TRUE) as geometry  
  from 
  `project-id.proj_global_sar.infra_wind_regions`),

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
  where label in ('oil', 'wind', 'other')
),

wind_region as (
select 
  detect_id,
  detect_date,
  lat,
  lon,
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
and label = 'wind'),

oil_region as (
select 
  detect_id,
  detect_date,
  lat,
  lon,
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
and label = 'oil')

select detect_date, label, count(*) as count from (
select * except(region) from oil_region
union all
select * except(region) from wind_region
union all (select detect_id, detect_date, lat, lon, label, iso3 from final_detections where label = 'other'))
group by 1, 2
order by 1


'''
class_time = pd.read_gbq(q)

# %%
class_time

# %%
mpl.rc('xtick', labelsize=16) 
mpl.rc('ytick', labelsize=16) 
fig, ax = plt.subplots(figsize=(20, 6))
# ax.plot(eez["date"].tolist(), eez["count"].tolist())
for label, table in class_time.groupby('label'):
    table.plot(x = 'detect_date', y = 'count', kind="line", ax=ax, label=label, linewidth = 4)

plt.legend(fontsize=20)
ax.set_ylim([0, class_time["count"].max()+1000])

plt.title(f"Infrastructure Detected", fontsize = 18)
plt.savefig('infra_label_timeseries', dpi = 300)
plt.show()

# %% [markdown]
# ## Continent/ EEZ bar chart

# %%
q = f'''
with 

all_oil_polygons as (
  select 
    ST_GEOGFROMTEXT(geometry, make_valid => TRUE) as geometry  
  from 
  proj_global_sar.oil_areas),

all_wind_polygons as (
  select 
    ST_GEOGFROMTEXT(geometry, make_valid => TRUE) as geometry  
  from 
  `project-id.proj_global_sar.infra_wind_regions`),

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
  where label in ('oil', 'wind', 'other')
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
and label = 'wind'),

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
and label = 'oil'),

detect_eez as (
select * from (
select * except(region) from oil_region
union all
select * except(region) from wind_region
union all (select detect_id, detect_date, label, iso3 from final_detections where label = 'other'))
where extract (year from detect_date) = 2021),


final as (
select
detect_date,
iso3,
sum(if(label='oil',1,0)) oil,
sum(if(label='wind',1,0)) wind,
sum(if(label='other',1,0)) other
from detect_eez 
group by 1,2
order by 1)

select 
distinct
iso3,
PERCENTILE_CONT(oil, 0.5) OVER(partition by iso3) AS oil,
PERCENTILE_CONT(wind, 0.5) OVER(partition by iso3) AS wind,
PERCENTILE_CONT(other, 0.5) OVER(partition by iso3) AS other
from 
final
-- order by oil_count_median desc
'''

cont_counts = pd.read_gbq(q)

# %%
cont_counts

# %%
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

# %%
cont_counts = cont_counts.fillna(0)

# %%
cont_counts['continent'] = cont_counts.iso3.apply(get_continent)

# %%
df_cont_grouped = cont_counts.groupby(['continent']).sum().reset_index()

# %%
df_cont_grouped

# %%
fig1, axs = plt.subplots(ncols=2, nrows=3,figsize=(15, 8))
plt.subplots_adjust(hspace = .3, wspace=0.9)



plt.rcParams['figure.facecolor'] = 'white'

for e, (c, i) in enumerate(zip(['Africa', 'South America', 'Europe', 'North America', 'Asia',
       'Australia'],axs.ravel())):
    df_plot = cont_counts.loc[cont_counts['continent'] == c]
    df_plot['total'] = df_plot['oil'] + df_plot['wind'] + df_plot['other']


    df_plot = df_plot.sort_values("total", ascending=False)
    df_plot = df_plot.head(10)
    df_plot = df_plot.drop(['total', 'continent'], axis = 1).set_index('iso3')

    df_plot.plot(kind='barh', stacked=True, ax = i, figsize=(12, 11), width = .8).invert_yaxis()

    i.legend(frameon=False)
    if e != 0:
        i.get_legend().remove()
    i.set(ylabel=None)
    i.set_title(f"{c}", size = 16)
    i.grid(b=None)

# plt.tight_layout()
plt.subplots_adjust(wspace=.2)
plt.suptitle('Oil, Wind, and Other Infrastructure Detections', size = 18, y = .93)

plt.savefig('all_infra_eez.png', bbox_inches="tight", dpi = 300)
plt.show()

# %% [markdown]
# ### Summarize by continent

# %%
cont_sum = cont_counts[['continent', 'oil', 'other', 'wind']]
cont_sum_grouped = cont_sum.groupby(['continent']).sum().reset_index()

# %%
cont_sum_grouped

# %%
sns.set_style("whitegrid")
sns.set(rc={"figure.figsize":(9, 6)})
cont_sum_grouped.set_index('continent').plot(kind='bar', stacked=True)
plt.xticks(rotation= 40)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title('infra count midpoint 2021-10-01', size = 18 )
plt.savefig('infra_label_count_single_date_', dpi=300)
plt.show()

# %% [markdown]
# ### Plot timeseries by continent

# %%
q = f'''

with 

all_oil_polygons as (
  select 
    ST_GEOGFROMTEXT(geometry, make_valid => TRUE) as geometry  
  from 
  proj_global_sar.oil_areas),

all_wind_polygons as (
  select 
    ST_GEOGFROMTEXT(geometry, make_valid => TRUE) as geometry  
  from 
  `project-id.proj_global_sar.infra_wind_regions`),

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
  where label in ('oil', 'wind', 'other')
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
and label = 'wind'),

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
and label = 'oil'),

detect_eez as (
select * from (
select * except(region) from oil_region
union all
select * except(region) from wind_region
union all (select detect_id, detect_date, label, iso3 from final_detections where label = 'other')))


select detect_date as date, label, iso3, count(*) as count
from detect_eez
group by 1,2,3
order by 1
'''
eez = pd.read_gbq(q)

# %%
eez

# %%
eez['continent']  = eez.iso3.apply(get_continent)

# %%
eez

# %%
cont_timeseries = eez.drop(['iso3'], axis=1)

# %%
cont_timeseries = cont_timeseries.groupby(['date', 'label', 'continent']).sum().reset_index()

# %%
cont = eez.continent.dropna().unique()

# %%
cont

# %%
for i in cont:
    df = cont_timeseries.loc[cont_timeseries['continent'] == i]
    fig, ax = plt.subplots(figsize=(20, 6))
    for label, table in df.groupby('label'):
        table.plot(x = 'date', y = 'count', kind="line", ax=ax, label=label, linewidth = 4)

    plt.legend(fontsize=20)
    ax.set_ylim([0, df["count"].max()+1000])

    plt.title(f"{i} Infrastructure", fontsize = 18)
    plt.savefig(f'{i}_infra_label_timeseries', dpi = 300)
    plt.show()

# %% [markdown]
# ## Timeseries of infra by class in different eezs

# %%
eez_grouped = eez[['iso3', 'count']].groupby('iso3').sum().sort_values('count', ascending=False)[:10].reset_index()
top_eez = eez_grouped['iso3']

# %% [markdown]
# ### Percent Change for top eez

# %%
for i in top_eez:
    df = eez.loc[eez['iso3'] == i]

    oil = df.loc[df['label'] == 'oil']
    wind = df.loc[df['label'] == 'wind']
    other = df.loc[df['label'] == 'other']

    if not oil.empty:
        oil_change = (((oil['count'].iloc[-1] - oil['count'].iloc[0]) / oil['count'].iloc[0]) * 100)
    if not wind.empty:
        wind_change = (((wind['count'].iloc[-1] - wind['count'].iloc[0]) / wind['count'].iloc[0]) * 100)
    if not other.empty:
        other_change = (((other['count'].iloc[-1] - other['count'].iloc[0]) / other['count'].iloc[0]) * 100)


    print (i)
    print(f'Oil Percent Change = {round(oil_change,2)}')
    print(f'Wind Percent Change = {round(wind_change,2)}')
    print(f'Other Percent Change = {round(other_change,2)}\n')

# %%
for i in top_eez:
    df = eez.loc[eez['iso3'] == i]
    fig, ax = plt.subplots(figsize=(20, 6))
    for label, table in df.groupby('label'):
        table.plot(x = 'date', y = 'count', kind="line", ax=ax, label=label, linewidth = 4)

    plt.legend(fontsize=20)
    ax.set_ylim([0, df["count"].max()+1000])

    plt.title(f"{i} infrastructure", fontsize = 18)
    # plt.savefig(f'{i}_infra_label_timeseries', dpi = 300)
    plt.show()

# %%

# %%

# %%
