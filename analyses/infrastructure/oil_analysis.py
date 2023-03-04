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
# ## This code is used to summarize how much infrastructure is in oil producing regions, and plot infrastructre by continent, oil producing regions, and eez.
#

# %%
import os

import cartopy
import geopandas as gpd
import matplotlib as mpl
import matplotlib.cbook as cbook
import matplotlib.colors as mpcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

import geopandas as gpd
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")
import pyseas.cm
import pyseas.contrib as psc
import pyseas.maps as psm
# import squarify
import matplotlib.pyplot as plt



# import pycountry
# from pycountry_convert import (
#     country_alpha2_to_continent_code,
#     country_name_to_country_alpha2,
# )

# # %load_ext autoreload
# # %autoreload 2

# # %reload_ext google.cloud.bigquery

# # %matplotlib inline

# import warnings
# warnings.filterwarnings('ignore')

# %% [markdown]
# ### Count and map infrastructure inside and outside of oil fields

# %%
q = f'''

with 

all_oil_polygons as (
  select 
    ST_GEOGFROMTEXT(geometry, make_valid => TRUE) as geometry  
  from 
  `proj_global_sar.oil_areas`),

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
and midpoint = '2020-10-01'
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
  -- where label in ('oil', 'probable_oil', 'wind', 'probable_wind', 'other')
  where label in ('oil', 'probable_oil', 'other')
),

oil_region as (
select 
  detect_id,
  detect_date,
  lon, 
  lat,
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
and label in ('oil', 'probable_oil')
),

detections_outside_oil_regions as (
select 
  detect_id,
  detect_date,
  lon, 
  lat,
  label,
  iso3,
  "outside_all_oil_polygons" as region,
from 
  final_detections
where 
 detect_id not in (select detect_id from oil_region)
),

unioned_oil
 as (
select * from oil_region
union all
select * from detections_outside_oil_regions),

oil_detects as (
select
detect_id, 
lon, 
lat,
-- sum(if(label='oil',1,0)) oil,
region, 
iso3
from unioned_oil
group by detect_id, region, lon, lat, iso3)
-- order by detect_date)

select * from oil_detects

'''


all_oil_areas = pd.read_gbq(q)

# %%
all_oil_areas

# %%
counts = all_oil_areas[['region', 'detect_id']].groupby('region').count().reset_index()

# %%
counts

# %%
counts['detect_id'].iloc[1]/counts['detect_id'].sum()

# %%
inside = all_oil_areas.loc[all_oil_areas['region'] == 'inside_all_oil_polygons']

outside = all_oil_areas.loc[all_oil_areas['region'] == 'outside_all_oil_polygons']

# %%
with psm.context(psm.styles.light):
    fig = plt.figure(figsize=(25, 15))
    psm.create_map()
    psm.add_land()
    psm.add_eezs()
    
    plt.scatter(inside.lon,inside.lat, s = 5, color = 'g', transform = psm.identity, label = 'inside_region')

    plt.scatter(outside.lon,outside.lat, s = 5, color = 'r', 
                transform = psm.identity, label = 'outside_all_oil_fields')

    plt.legend(fontsize = 24, markerscale=6)
    
    # plt.savefig('oil_in_out_regions', dpi = 600)

# %% [markdown]
# ### Oil infrastructure grouped by continent

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
q = f'''


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


'''


cont_counts = pd.read_gbq(q)

# %%
cont_counts = cont_counts.fillna(0)

# %%
cont_counts['continent'] = cont_counts.iso3.apply(get_continent)

# %%
cont_counts

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


    df_plot = df_plot.sort_values("oil_count_median", ascending=False)
    df_plot = df_plot.head(10)
    df_plot = df_plot.drop(['continent'], axis = 1).set_index('iso3')

    df_plot.plot(kind='barh', stacked=True, ax = i, figsize=(12, 11), width = .8).invert_yaxis()

    # i.legend(frameon=False)
    if e != 5:
        i.get_legend().remove()
    else:
        i.legend(labels = ['oil ', 'probable oil'], frameon=False, fontsize = 15, loc = 'lower right')
    i.set(ylabel=None)
    i.set_title(f"{c}", size = 16)
    i.grid(b=None)

# plt.tight_layout()
plt.subplots_adjust(wspace=.2)
plt.suptitle('Infrastructure Oil Detections', size = 18, y = .93)

plt.savefig('oil_cont_eez.png', bbox_inches="tight", dpi = 300)
plt.show()

# %% [markdown]
# ### By manual oil regions

# %%
q = '''
with 

all_oil_polygons as (
  select 
    ST_GEOGFROMTEXT(geometry, make_valid => TRUE) as geometry  
  from 
  `proj_global_sar.oil_areas`),

oil_regions as (
  select 
  region as manual_region,
    ST_GEOGFROMTEXT(geometry, make_valid => TRUE) as manual_geometry  
  from 
  `proj_global_sar.infra_oil_regions`),

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
  where label in ('oil', 'probable_oil', 'wind', 'probable_wind', 'other')
),

oil_region as (
select 
  detect_id,
  detect_date,
  label,
  iso3,
  lat,
  lon,
  "inside_all_oil_polygons" as region,
from 
  final_detections
cross join
  all_oil_polygons
where  
  st_contains(geometry,
             ST_GEOGPOINT(lon,lat))
and label in ('oil', 'probable_oil')
),

detections_outside_oil_regions as (
select 
  detect_id,
  detect_date,
  label,
  iso3,
  lat,
  lon,
  "outside_all_oil_polygons" as region,
from 
  final_detections
where 
 detect_id not in (select detect_id from oil_region)
),

unioned_oil
 as (
select * from oil_region
union all
select * from detections_outside_oil_regions),

oil_manual_region as (
select * from unioned_oil
cross join
  oil_regions
where  
  st_contains(manual_geometry,
             ST_GEOGPOINT(lon,lat))
and region = 'inside_all_oil_polygons'
and extract (year from detect_date) = 2021),

final as (
select 
manual_region as region, 
detect_date,
sum(if(label='oil',1,0)) oil,
sum(if(label='probable_oil',1,0)) probable_oil,
 from oil_manual_region
 group by region, detect_date
 order by region, detect_date)

select 
distinct
region,
PERCENTILE_CONT(oil, 0.5) OVER(partition by region) AS oil_count_median,
PERCENTILE_CONT(probable_oil, 0.5) OVER(partition by region) AS probable_oil_count_median
from 
final
order by oil_count_median desc

'''

regions_cnt = pd.read_gbq(q)

# %%
regions_cnt

# %%
count_single_date_grouped = cont_counts[['iso3', 'oil_count_median', 'probable_oil_count_median']].sort_values('oil_count_median', ascending = False)

# %%
count_single_date_grouped

# %%
regions_cnt = regions_cnt.set_index('region')

# %%
sns.set_style("whitegrid")
sns.set(rc={"figure.figsize":(9, 6)})
regions_cnt.plot(kind='barh', stacked=True).invert_yaxis()
plt.xticks(rotation= 40)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title('Detections 2021', size = 18 )
plt.legend(labels = ['oil ', 'probable oil'], frameon=False, fontsize = 15, loc = 'lower right')
plt.savefig('regional_oil_median_2021', dpi=300, bbox_inches = 'tight')
plt.show()

# %% [markdown]
# ### Median monthly timeseries

# %%
median_monthly = pd.read_gbq(
'''with 

all_oil_polygons as (
  select 
    ST_GEOGFROMTEXT(geometry, make_valid => TRUE) as geometry  
  from 
  `proj_global_sar.oil_areas`),

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
  -- where label in ('oil', 'probable_oil', 'wind', 'probable_wind', 'other')
  where label in ('oil', 'probable_oil')
),

oil_region as (
select 
  detect_id,
  detect_date,
  lon, 
  lat,
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
and label in ('oil', 'probable_oil')
),

detections_outside_oil_regions as (
select 
  detect_id,
  detect_date,
  lon, 
  lat,
  label,
  iso3,
  "outside_all_oil_polygons" as region,
from 
  final_detections
where 
 detect_id not in (select detect_id from oil_region)
),

unioned_oil
 as (
select * from oil_region
union all
select * from detections_outside_oil_regions),

oil_detects as (
select detect_date, oil, probable_oil, region from
( select
detect_date,
sum(if(label='oil',1,0)) oil,
sum(if(label='probable_oil',1,0)) probable_oil,
region
from unioned_oil
-- where iso3 = 'CHN'
group by detect_date, region
order by detect_date)),


oil_detect_eez as (
select * except (region) from oil_detects
where region = 'inside_all_oil_polygons')


select 
distinct
DATE(extract( year from detect_date), extract( month from detect_date),01) year_month,
PERCENTILE_CONT(oil, 0.5) OVER(partition by extract( year from detect_date), extract( month from detect_date)) +
PERCENTILE_CONT(probable_oil, 0.5) OVER(partition by extract( year from detect_date), extract( month from detect_date)) AS oil_count_median
from 
oil_detect_eez
order by year_month desc'''


)

# %%
median_monthly

# %%
mpl.rc('xtick', labelsize=16) 
mpl.rc('ytick', labelsize=16) 
fig, ax = plt.subplots(figsize=(20, 6))
# ax.plot(eez["date"].tolist(), eez["count"].tolist())
median_monthly.plot(x = 'year_month', y = 'oil_count_median', kind="line", ax=ax, label='oil', linewidth = 4)

plt.legend(fontsize=20)
# ax.set_ylim([0, 3000])

plt.title(f"monthly Infrastructure Detected", fontsize = 18)
plt.savefig('oil_timeseries_monthly_median_gbr', dpi = 300)
plt.show()

# %% [markdown]
# ### Continent timeseries

# %%
q = f'''

with 

all_oil_polygons as (
  select 
    ST_GEOGFROMTEXT(geometry, make_valid => TRUE) as geometry  
  from 
  `proj_global_sar.oil_areas`),


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
  where label in ('oil', 'probable_oil', 'wind', 'probable_wind', 'other')
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
and label in ('oil', 'probable_oil')
),

detections_outside_oil_regions as (
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

unioned_oil
 as (
select * from oil_region
union all
select * from detections_outside_oil_regions),

oil_detects as (
  select
detect_date,
sum(if(label='oil',1,0)) oil,
sum(if(label='probable_oil',1,0)) probable_oil,
iso3,
region
from unioned_oil
group by detect_date, region, iso3
order by detect_date),

oil_detect_eez as (
select * except (region) 
from oil_detects
where region = 'inside_all_oil_polygons')


select 
distinct
iso3,
DATE(extract( year from detect_date), extract( month from detect_date),01) year_month,
PERCENTILE_CONT(oil, 0.5) OVER(partition by iso3, extract( year from detect_date), extract( month from detect_date)) +
PERCENTILE_CONT(probable_oil, 0.5) OVER(partition by iso3, extract( year from detect_date), extract( month from detect_date)) AS oil_count_median
from 
oil_detect_eez
order by iso3, year_month desc


'''

eez = pd.read_gbq(q)

# %%
eez

# %%
eez['continent'] = eez.iso3.apply(get_continent)

# %%
df_cont = eez[['iso3','oil_count_median', 'continent', 'year_month']]

# %%
df_cont_grouped = df_cont.groupby(['continent', 'year_month']).sum().reset_index()

# %%
df_cont_grouped

# %%
for i in df_cont_grouped['continent'].unique():
    print(i)

# %%
for i in df_cont_grouped['continent'].unique():

    fig = plt.figure(figsize =(30,15))
    fig.tight_layout()
    df = df_cont_grouped.loc[df_cont_grouped['continent'] == i]
    ax = sns.lineplot(x = df['year_month'], y = df['oil_count_median'], label = 'oil_count_median', linewidth=3.0)
    plt.legend(bbox_to_anchor=[1, 1], prop={'size': 16})
    plt.xlabel('year_month', fontsize=24)
    plt.ylabel('Count', fontsize=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.xticks(rotation=45)
    plt.margins(x=0)
    plt.title(f"oil_count_median in {i}", fontsize = 24)
    plt.savefig(f'oil_count_median in {i}', dpi = 300)
    plt.show()

# %%
eez_grouped = eez[['iso3', 'oil_count_median']].groupby('iso3').sum().sort_values('oil_count_median', ascending=False)[:10].reset_index()
top_eez = eez_grouped['iso3']

# %%
for i in top_eez:
    df = eez.loc[eez['iso3'] == i]

    oil_change = (((df['oil_count_median'].iloc[-1] - df['oil_count_median'].iloc[0]) / df['oil_count_median'].iloc[0]) * 100)

    print (i)
    print(f'oil percent change = {round(oil_change,2)}')

# %%
for i in top_eez:
    df = eez.loc[eez['iso3'] == i]
    fig, ax = plt.subplots(figsize=(20, 6))
    df.plot(x = 'year_month', y = 'oil_count_median', kind="line", ax=ax, label='wind', linewidth = 4)

    plt.legend(fontsize=20)
    ax.set_ylim([0, df["oil_count_median"].max()+1000])

    plt.title(f"{i} oil time series", fontsize = 18)
    # plt.savefig(f'{i}_infra_oil_timeseries', dpi = 300)
    plt.show()



# %%
