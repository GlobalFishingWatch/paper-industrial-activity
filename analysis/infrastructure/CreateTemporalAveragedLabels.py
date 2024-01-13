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

# # Create Temorally Averaged Labels
#
# Our algorithm to detect infrastructure identifies objects in a six month window of time and then assigns a class to those objects (noise, wind, oil, or other). The queries in this notebook average those classifications across time. That is, if a detection appears in two different time windows, this averages the values to get the best value across time. The result is more stable and accurate predictions.

# +
from datetime import datetime, timedelta, date
from google.cloud import bigquery
import pandas as pd
# Construct a BigQuery client object.
client = bigquery.Client()
import sys
sys.path.append('../utils') 

from bigquery_helper_functions import query_to_table, update_table_description
# -

q = f'''
with labeled as

(
select 
  detect_id,
  max_value,
  case 
    when wind = max_value then "wind"
    when oil = max_value then "oil"
    when other = max_value then "other"
    when noise = max_value then "noise"
    else null end as label,
  wind,
  oil,
  other,
  noise
from
(
select 
  greatest(wind	,oil,	other,	noise) max_value,
  *
from 
  `proj_sentinel1_v20210924.detect_comp_pred_v2_*` 
)


),

by_h3cell as (
select 
  start_time,
  label,
  detect_lon,
  detect_lat,
  detect_id,
  jslibs.h3.ST_H3(st_geogpoint(detect_lon, detect_lat),10) h3,
  max_value,
  wind,
  oil,
  other,
  noise
from 
  labeled
join
  `proj_sentinel1_v20210924.detect_comp_raw_*`
using(detect_id)
),

max_label_grouped as (
select 
  h3,
  case 
    when wind = max_value then "wind"
    when oil = max_value then "oil"
    when other = max_value then "other"
    when noise = max_value then "noise"
    else null end as avg_label,
  max_value as avg_max_value,
  wind, oil, other, noise
from
  (

  select 
    h3,
    avg(wind) wind,
    avg(oil) oil,
    avg(other) other,
    avg(noise) noise,
    greatest(avg(wind), avg(oil),avg(other), avg(noise)) max_value
  from
    by_h3cell
  group by 
  h3
  )
),


grouped as (

select 
  h3,
  array_agg(struct(label, detections,avg_score)) as label_array ,
  sum(detections) months_detected
from
  (
  select 
    h3,
    label,
    count(*) detections,
    avg(max_value) avg_score
  from 
    by_h3cell
  group by 
    h3, label
  )
group by 
  h3
)


select 
  detect_id,
  avg_label as label,
  avg_max_value as max_value,
  label as monthly_label,
  max_value monthly_max_value,
  h3
 from 
   by_h3cell
join
  max_label_grouped
using(h3)
'''

table_name = f'{project_id}.proj_global_sar.composite_ids_labeled_v20230616'

query_to_table(q, table_name)

#add description to GBQ table
table_description = '''this table averages the score labels across time to get the most accurate labels for objects'

the query is: ''' + q

#update table with description
update_table_description(table_name, table_description )


