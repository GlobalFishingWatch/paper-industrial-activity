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

# # Rasetrize
#
# This notebook runs the rasterization code, turning each Sentinel-1 footprint into a raster with a resolution of 1/200th of a degree (about 550 meters). 

import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
def gbq(q):
    return pd.read_gbq(q, project_id='project-id')


def execute_commands_in_parallel(commands):
    '''This takes a list of commands and runs them in parallel
    Requires having gnu parallel installed on your machine.
    '''
    with open('commands.txt', 'w') as f:
        f.write("\n".join(commands))    
    os.system("parallel -j 16 < commands.txt")
    os.system("rm -f commands.txt")


# +
# footprint_raster_table = 'proj_sentinel1_v20200820.sentinel_1_footprints_500m_1km_v2'
# footrpint_vector_table = 'proj_sentinel1_v20200820.detect_footprints_buff1km'

footprint_raster_table = 'proj_sentinel1_v20210924.detect_foot_raster_200'
# footprint_raster_table = 'proj_sentinel1_v20210924.detect_foot_raster_new_buff250m_200'
# footrpint_vector_table = 'proj_sentinel1_v20210924.detect_foot_raw_new_'
footrpint_vector_table = 'proj_sentinel1_v20210924.detect_foot_raw_'

# +
# # !bq rm -f proj_sentinel1_v20210924.detect_foot_raster_new_200

# +
# os.system("bq mk --time_partitioning_type=DAY " + footprint_raster_table)

# +
# # !bq rm -f proj_sentinel1_v20210924.detect_foot_raster_new_200
# -

q = f'''

with raw as (
    SELECT 
      distinct _table_suffix t 
    FROM 
      `project-id.{footrpint_vector_table}*`
    order by t),


processed as (
  select 
    format_timestamp( "%Y%m%d",_partitiontime) t, 
  FROM 
    {footprint_raster_table}
)


select 
  * 
from 
 raw
where 
 t not in (select * from processed)


'''
df = gbq(q)


df = df.sort_values('t')
df

# +
commands = []

one_over_cellsize = 200

for t in df.t.values:
    try:
        d = datetime.strptime(t,"%Y%m%d")
    except:
        continue
    d = d.strftime("%Y-%m-%d")
    footrpint_vector_table_day = footrpint_vector_table+t
    footprint_raster_table_day = footprint_raster_table + "\$" +t

    command = (  f"jinja2 raster.sql.j2 "
        f"-D one_over_cellsize='{one_over_cellsize}' "
        f"-D YYYY_MM_DD='{d}' "       
        f"-D footrpint_vector_table='{footrpint_vector_table}' "
        "| "
        "bq query --replace "
        f"--destination_table={footprint_raster_table_day} "
        f"--allow_large_results --use_legacy_sql=false ")
    commands.append(command)
# -

commands[0]

len(commands)

# +
# # !jinja2 raster.sql.j2 -D one_over_cellsize='200' -D YYYY_MM_DD='2015-01-01' -D footrpint_vector_table='proj_sentinel1_v20210924.detect_foot_raw_new_'
# -



print(command)

# +
# # !jinja2 raster.sql.j2 -D one_over_cellsize='200' -D YYYY_MM_DD='2017-01-01' -D footrpint_vector_table='proj_sentinel1_v20210924.detect_foot_raw_new_' 

# +
# os.system(commands[0])
# -

execute_commands_in_parallel(commands)



# +
# # !bq rm -f proj_sentinel1_v20210924.detect_foot_raster_10

# +
# # !bq cp proj_sentinel1_v20210924.detect_foot_raster_new_buff250m_200 proj_sentinel1_v20210924.detect_foot_raster_200
# -



# # Now create 10th of a degree raster

footprint_raster_table10 = 'proj_sentinel1_v20210924.detect_foot_raster_10'

os.system("bq mk --time_partitioning_type=DAY "+footprint_raster_table10)


# +
q_template = '''select scene_id, 
floor(lat_index/200*10) lat_index,
floor(lon_index/200*10) lon_index,
avg(look_angle) look_angle,
sum(1/200*1/200*10*10) overpasses
from proj_sentinel1_v20210924.detect_foot_raster_200
where _partitiontime = "{{ YYYY_MM_DD }}"
group by scene_id, lat_index, lon_index
'''

with open("downsample200_10.sql.j2",'w') as f:
    f.write(q_template)
# -

q = f'''select t from (SELECT distinct _partitiontime t 
FROM proj_sentinel1_v20210924.detect_foot_raster_200)
where t not in
  (select distinct _partitiontime t from {footprint_raster_table10} )

order by t

'''
df = gbq(q)



df.t.values[0], df.t.values[-1], len(df)

commands = []
for t in df.t.values:
    ts = pd.to_datetime(str(t))  
    d = ts.strftime('%Y-%m-%d')  
    t = ts.strftime('%Y%m%d')  
    footprint_raster_table10_day = footprint_raster_table10 + "\$" +t

    command = (  f"jinja2 downsample200_10.sql.j2 "
        f"-D YYYY_MM_DD='{d}' "       
        "| "
        "bq query --replace "
        f"--destination_table={footprint_raster_table10_day} "
        f"--allow_large_results --use_legacy_sql=false ")
    commands.append(command)

commands[0]

execute_commands_in_parallel(commands)




