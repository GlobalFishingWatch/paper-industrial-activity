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

import matplotlib.pyplot as plt
import proplot

# # Likelihood in Scene
#
# Does some matrix multiplication to get the likelihood each vessel in the SAR scenes
#
# Tables created:
#  - proj_global_sar.extrap_scene_edge
#  - proj_global_sar.mult_inside_0
#  - proj_global_sar.mult_inside_1
#  - proj_global_sar.mult_inside_2
#  - proj_global_sar.mult_inside_3
#  - proj_global_sar.expected_recall

import matplotlib.pyplot as plt
import proplot

import sys
sys.path.append('../utils')
from bigquery_helper_functions import query_to_table
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pyperclip
import pandas as pd
from proj_id import project_id

tables = ''' - proj_global_sar.extrap_scene_edge
 - proj_global_sar.mult_inside_0
 - proj_global_sar.mult_inside_1
 - proj_global_sar.mult_inside_2
 - proj_global_sar.mult_inside_3
 - proj_global_sar.expected_recall'''.replace(' - ','').split("\n")
tables

# check to see if all the tables were created
for table in tables:
    q = f'''
    select distinct _partitiontime t from 
    `gfw_research.pipe_v20201001_fishing`
    where date(_partitiontime) between "2017-01-01" and "2017-12-31" -- date range to check
    and _partitiontime not in
    (select distinct _partitiontime t from 
    {table}
    order by t
    )'''
    df = pd.read_gbq(q)
    print(table, df.t.values)





### Range of dates to process
the_dates = np.arange(
    datetime(2022, 1, 1),
    datetime(2022, 1, 1) + timedelta(days=1),
    timedelta(days=1),
).astype(datetime)




def create_extrap_interesting(the_date):
    
    '''This query gets all vessels who raster-lookup table (the area within the two 
    raster loookup tables) is not entirely within the scene. If the lookup raster
    is entirely within the scene, than we don't have to calculate the likelihood
    it is in the scene because we know it had to appear in the scene. ''' 

    q = f'''

    CREATE TEMP FUNCTION map_label(label string)
    as (
    case when label ="drifting_longlines" then "drifting_longlines"
    when label ="purse_seines" then "purse_seines"
    when label ="other_purse_seines" then "purse_seines"
    when label ="tuna_purse_seines" then "purse_seines"
    when label ="cargo_or_tanker" then "cargo_or_tanker"
    when label ="cargo" then "cargo_or_tanker"
    when label ="tanker" then "cargo_or_tanker"
    when label ="tug" then "tug"
    when label = "trawlers" then "trawlers"
    else "other" end
    );



    with 
    
    vessel_info as 
    (  select best.best_vessel_class as label, ssvid from gfw_research.vi_ssvid_v20221001),
    
    extrapolate_table as (
    -- get rid of duplicates... not sure why they are here
    select * except(row, footprint_wkt, footprint_wkt_1km) from
        (select *, row_number() over (partition by ssvid, scene_id, source order by rand()) row 
        from proj_sentinel1_v20210924.detect_foot_ext_doppler
        where date(_partitiontime) = "{the_date:%Y-%m-%d}"
        and source = "AIS")
    join
    (select distinct scene_id, footprint_wkt, footprint_wkt_1km 
      from proj_sentinel1_v20210924.detect_foot_raw_{the_date:%Y%m%d}
     where (safe.st_geogfromtext(footprint_wkt) is not null or
           safe.st_geogfromtext(footprint_wkt_1km) is not null)
    )
    using(scene_id)
    where not st_contains(
        ifnull(safe.st_geogfromtext(footprint_wkt), st_geogfromtext(footprint_wkt_1km))
        ,lookup_raster) 
    and row = 1
    )


    select 
    map_label(label) as label,
    rand() random_number,
     * except(label) from 
    extrapolate_table
    left join
    vessel_info
    using(ssvid)

    '''
    query_to_table(q, f"{project_id}.proj_global_sar.extrap_scene_edge${the_date:%Y%m%d}")



# +
## run the first time to create the table
# # !bq mk --time_partitioning_type=DAY proj_global_sar.extrap_scene_edge
# -

with ThreadPoolExecutor(max_workers=16) as e:
    for d in the_dates:
        e.submit(create_extrap_interesting, d)





# +

def prob_inside(the_date, value):

    extrap_table = f"proj_global_sar.extrap_scene_edge" 

    
    q = f'''#StandardSql



        create temp function radians(x float64) as (
          3.14159265359 * x / 180
        );

        create temp function deglat2km() as (
          111
        );

        create temp function meters2deg() as (
           1/(deglat2km()*1000)
        );

        create temp function kilometers_per_nautical_mile() as (
          1.852
        );


        create temp function map_label(label string)
        as (
          case when label ="drifting_longlines" then "drifting_longlines"
          when label ="purse_seines" then "purse_seines"
          when label ="other_purse_seines" then "purse_seines"
          when label ="tuna_purse_seines" then "purse_seines"
          when label ="cargo_or_tanker" then "cargo_or_tanker"
          when label ="cargo" then "cargo_or_tanker"
          when label ="tanker" then "cargo_or_tanker"
          when label ="tug" then "tug"
          when label = "trawlers" then "trawlers"
          else "other" end
        );

        create temp function map_speed(x float64) as (
          case
          when x < 2 then 0
          when x < 4 then 2
          when x < 6 then 4
          when x < 8 then 6
          when x < 10 then 8
          else 10 end
        );

        create temp function map_minutes(x float64) as (
          case when x < -384 then  -512
          when x < -256 then -384
          when x < -192 then -256
          when x < -128 then -192
          when x < -96 then -128
          when x < -64 then -96
          when x < -48 then -64
          when x < -32 then -48
          when x < -24 then -32
          when x < -16 then -24
          when x < -12 then -16
          when x < -8 then -12
          when x < -6 then -8
          when x < -4 then -6
          when x < -3 then -4
          when x < -2 then -3
          when x < -1 then -2
          when x < 0 then -1
          when x < 1 then 0
          when x < 2 then 1
          when x < 3 then 2
          when x < 4 then 3
          when x < 6 then 4
          when x < 8 then 6
          when x < 12 then 8
          when x < 16 then 12
          when x < 24 then 16
          when x < 32 then 24
          when x < 48 then 32
          when x < 64 then 48
          when x < 96 then 64
          when x < 128 then 96
          when x < 192 then 128
          when x < 256 then 192
          when x < 384 then 256
          else 384 end);





        with
        ######################################
        -- Data sources
        ######################################
        --

        prob_raster as (select * from
         `global-fishing-watch.paper_longline_ais_sar_matching.point_cloud_mirror_nozeroes_contour_v20190502`
         ),
        --
        --
        ##################################
        # Probability raster adjustments
        ##################################
        --
        -- get the average for general fishing. This is creating a composite category
        -- of other categories.
        probability_table as (select
           probability,labels,minutes_lower,speed_lower,i,j
        from
          prob_raster
        union all
        select
           avg(probability) probability, "fishing" labels, minutes_lower, speed_lower,i,j
        from
          prob_raster
        where labels in ("trawlers","purse_seines","drifting_longlines")
        group by
           labels,minutes_lower,speed_lower,i,j),
        --
        --


        --
        --
        -- the raster has only positive i values because it is symmetrical
        -- around the i axis (the boat is as equally likely to turn left as right).
        -- Make a full raster with negative values for the later join.
        probabilities_and_weights_neg as (
        select
        labels, minutes_lower, speed_lower, probability, i, j --, weight
        from
        probability_table
        union all
        select
        -- same except take negative i!
        labels, minutes_lower, speed_lower, probability, -i as i, j -- , weight
        from
        probability_table
        where i >0
        ),
        #########################
        ##
        ## SAR subqueries
        ##
        #########################
        --
        --

       interesting_vessels as (
         select * from 
         {extrap_table}
         where 
           date(_partitiontime) = "{the_date:%Y-%m-%d}"
           and random_number between {value/4} and {value/4} + .25 
         ),




        just_bounds_label as (

       select
         ssvid,
        label,
        scene_id,
        least(ifnull(scale1,1e10), ifnull(scale2,1e10)) as max_scale,
        speed1 as speed,
        course1 as course,
        scale1 as scale,
        round( least(ifnull(scale1,1e10), ifnull(scale2,1e10))*111) as pixels_per_degree,
        round( least(ifnull(scale1,1e10), ifnull(scale2,1e10))*111*cos(radians(ST_BOUNDINGBOX(lookup_raster).ymin/2+ST_BOUNDINGBOX(lookup_raster).ymin/2))) as pixels_per_degree_lon,
        least(abs(delta_minutes1), abs(delta_minutes2)) min_delta_minutes,
        is_single,
        delta_minutes1 as delta_minutes,
        ST_BOUNDINGBOX(lookup_raster).xmin as min_lon,
        ST_BOUNDINGBOX(lookup_raster).ymin as min_lat,
        ST_BOUNDINGBOX(lookup_raster).xmax as max_lon,
        ST_BOUNDINGBOX(lookup_raster).ymax as max_lat,
        lat_extrapolate1 + lat_doppler_offset as lat_interpolate_adjusted,
        lon_extrapolate1 + lon_doppler_offset as lon_interpolate_adjusted,
        from interesting_vessels
        where delta_minutes1 is not null
        union all
        select
        ssvid,
        label,
        scene_id,
        least(ifnull(scale1,1e10), ifnull(scale2,1e10)) as max_scale,
        speed2 as speed,
        course2 as course,
        scale2 as scale,
        round( least(ifnull(scale1,1e10), ifnull(scale2,1e10))*111) as pixels_per_degree,
        round( least(ifnull(scale1,1e10), ifnull(scale2,1e10))*111*cos(radians(ST_BOUNDINGBOX(lookup_raster).ymin/2+ST_BOUNDINGBOX(lookup_raster).ymin/2))) as pixels_per_degree_lon,
        least(abs(delta_minutes1), abs(delta_minutes2)) min_delta_minutes,
        is_single,
        delta_minutes2 as delta_minutes,
        ST_BOUNDINGBOX(lookup_raster).xmin as min_lon,
        ST_BOUNDINGBOX(lookup_raster).ymin as min_lat,
        ST_BOUNDINGBOX(lookup_raster).xmax as max_lon,
        ST_BOUNDINGBOX(lookup_raster).ymax as max_lat,
        lat_extrapolate2 + lat_doppler_offset as lat_interpolate_adjusted,
        lon_extrapolate2 + lon_doppler_offset as lon_interpolate_adjusted,
        from interesting_vessels
        where delta_minutes2 is not null

        ),




         lat_array AS(
          SELECT
            * ,
            lat_bin/pixels_per_degree as detect_lat  -- to get the middle of the cell
          FROM
            just_bounds_label
          CROSS JOIN
            UNNEST(GENERATE_ARRAY(
                   cast(FLOOR(min_lat*pixels_per_degree) as int64) - 2, -- just to pad it a bit
                   cast(FLOOR(max_lat*pixels_per_degree) as int64) + 2,
                   1))
             AS lat_bin

             ),


          lon_array AS (
          SELECT
            *,
            lon_bin/pixels_per_degree_lon as detect_lon -- to get the middle of the cell
          FROM
            just_bounds_label
          CROSS JOIN
                UNNEST(GENERATE_ARRAY(
                   cast(FLOOR(min_lon*pixels_per_degree_lon) as int64) - 2, -- just to pad it a bit
                   cast(FLOOR(max_lon*pixels_per_degree_lon) as int64) + 2,
                   1))  AS lon_bin

           --      where scene_id = "RS2_20190924_150552_0074_DVWF_HH_SCS_759668_9852_29818072"
    ),



          id_lat_lon_array AS (
          select
          a.ssvid,
          a.label,
          a.scene_id,
          a.speed,
          a.course,
          a.scale,
          a.delta_minutes,
          a.lat_interpolate_adjusted,
          a.lon_interpolate_adjusted,
          lon_bin,
          lat_bin,
          detect_lon,
          detect_lat,
          a.is_single,
          FROM
            lon_array a
          CROSS JOIN
            lat_array b
          WHERE
            a.scene_id=b.scene_id
            and a.ssvid=b.ssvid
            and a.delta_minutes = b.delta_minutes
            ),

        --
        ####################
        # joins to the probability raster
        ###################
        --


        # --
        # -- adjust by scale -- that is the probability raster will be scalled
        # -- based on how long before or after the ping the image was taken.
        # -- Also, move the raster so that 0,0 is where the vessel would be
        # -- if it traveled in a straight line.
        key_query_1 as (
        select *,
          deglat2km() * (detect_lon - lon_interpolate_adjusted) * cos(radians(lat_interpolate_adjusted)) as u,
          deglat2km() * (detect_lat - lat_interpolate_adjusted) as v,
          radians(course) as course_rads
        from
          id_lat_lon_array
        ),

        # --
        # -- rotate the coordinates
        key_query_2 as (
        select
          *,
          cos(course_rads) * u - sin(course_rads) * v as x,
          cos(course_rads) * v + sin(course_rads) * u as y,
          -- rotation of coordinates, described here: https://en.wikipedia.org/wiki/Rotation_of_axes
          -- Note that our u and v / x and y are switched from the standard way to measure
          -- this, largely because vessels measure course from due north, moving clockwise,
          -- while mosth math measures angle from the x axis counterclockwise. Annoying!
          --
          # 1000 / colmax(1.0, ABS(delta_minutes)) as scale
        #     This is the python function we are copying here:
        #      def scale(dt):
        #         return 1000.0 / max(1, abs(dt))
        from
          key_query_1
        ),
        # --
        # -- adjust by scale -- that is the probability raster will be scalled
        # -- based on how long before or after the ping the image was taken.
        # -- Also, move the raster so that 0,0 is where the vessel would be
        # -- if it traveled in a straight line.
        key_query_3 as
        (
        select * from (
          select *,
            x * scale as x_key,
            # (y - speed*kilometers_per_nautical_mile()*delta_minutes/60 ) * scale  as y_key,
            y  * scale  as y_key, # because using interpolated position, already centered at this location
            # Map these values to the values in the probability rasters
            map_speed(speed) as speed_key,
            map_minutes(delta_minutes) as minute_key,
            map_label(label) as label_key
          from
            key_query_2
          )
        where abs(x_key) <=500 and abs(y_key) <=500
        ),

        # # --
        # # --
        -- Match to probability, and interpolate between
        -- the four closest values. This bilinear interpoloation
        -- in theory allows us to reduce the size of the raster we are joining on
        messages_with_probabilities as
         (
         select
        --   -- this would get the value exact, the weight_scaled
        --   -- / pow((1000/(colmax( 1, probs.minutes_lower/2 + probs.minutes_upper /2))),2) * scale*scale
           * except(i,j,probability),
           probs_11.probability as probability
        --   -- to get at least one value.
        --   -- weight *should* be the same for each, but we need to make sure it isn't null
         from
           key_query_3
         left join
        -- joining on four locaitons to do bilinear interpolation
         probabilities_and_weights_neg  as probs_11
           on  probs_11.i = cast(x_key - .5 as int64) and probs_11.j = cast(y_key - .5 as int64)
           and probs_11.speed_lower = cast(speed_key as int64)
           and probs_11.minutes_lower = cast(minute_key as int64)
           and probs_11.labels = label_key
         ),



        prob_multiplied_table as
        (
          select
          ssvid,
          scene_id,
          a.probability*b.probability   probability,
          a.detect_lon,
          a.detect_lat
          from messages_with_probabilities a
          join
          messages_with_probabilities b
          using(ssvid, lat_bin, lon_bin, scene_id)
          where b.delta_minutes < a.delta_minutes
          and not a.is_single
          
          union all
          select 
          ssvid,
          scene_id,
          probability,
          detect_lon,
          detect_lat
          from messages_with_probabilities
          where is_single
        )


        select ssvid, a.scene_id, 
        sum(if(lat_index is not null, probability,0))/sum(probability) as prob_inside
        from prob_multiplied_table a 
        left join proj_sentinel1_v20210924.detect_foot_raster_200 b
        on floor(detect_lat*200) = lat_index and
        floor(detect_lon*200) = lon_index 
        and a.scene_id = b.scene_id
        and date(_partitiontime) = "{the_date:%Y-%m-%d}"
        group by ssvid, scene_id

    '''
#     pyperclip.copy(q)
    query_to_table(
        q, f"{project_id}.proj_global_sar.mult_inside_{value}${the_date:%Y%m%d}"
    )

# +
## Run the first time this script is run to create the tables

# for value in range(4):
#     command = f'bq mk --time_partitioning_type=DAY proj_global_sar.mult_inside_{value}'
#     subprocess.call(command.split())
# -

with ThreadPoolExecutor(max_workers=8) as e:
    for d in the_dates:
        for value in range(4):
            e.submit(prob_inside, d, value)



# # Combine to get total likely vessels in each scene

def combine_likelihoods(the_date):


    q = f'''CREATE TEMP FUNCTION YYYYMMDD(d DATE) AS (
      # Format a date as YYYYMMDD
      # e.g. DATE('2018-01-01') => '20180101'
      FORMAT_DATE('%Y%m%d',
        d) );

    Create temp function the_date() as 
    (  date("{the_date:%Y-%m-%d}"));

    with   extrapolate_table as (
        -- get rid of duplicates... not sure why they are here
        select * except(row) from
            (select *, row_number() over (partition by ssvid, scene_id, source order by rand()) row 
            from proj_sentinel1_v20210924.detect_foot_ext_doppler
            where date(_partitiontime) = the_date()
            and source = "AIS"
            )
         
        where row = 1
        ),

    missing as (
    select ssvid, scene_id, 1 as prob_inside  from 
    extrapolate_table
    where concat(ssvid, scene_id) not in (select concat(ssvid,scene_id) from
    `proj_global_sar.extrap_scene_edge` b
    where date(_partitiontime) = the_date() )
    ),


    vessels_inside as (
    select ssvid, scene_id, prob_inside
    from proj_global_sar.mult_inside_0 
    where date(_partitiontime) = the_date()
    union all
    select ssvid, scene_id, prob_inside
    from proj_global_sar.mult_inside_1 
    where date(_partitiontime) = the_date()
    union all
    select ssvid, scene_id, prob_inside
    from proj_global_sar.mult_inside_2 
    where date(_partitiontime) = the_date()
    union all
    select ssvid, scene_id, prob_inside
    from proj_global_sar.mult_inside_3 
    where date(_partitiontime) = the_date()),

    all_expectations as 
    (select * from 
    vessels_inside
    union all
    select * from missing
    )  

    select ssvid, scene_id, avg(prob_inside) prob_inside
     from all_expectations 
     group by ssvid, scene_id'''
#     pyperclip.copy(q)
    
    query_to_table(
        q, f"{project_id}.proj_global_sar.likelihood_inside${the_date:%Y%m%d}"
    )

# +
## Run the first time this table is notebook is run
# command = f'bq mk --time_partitioning_type=DAY proj_global_sar.likelihood_inside'
# subprocess.call(command.split())
# -

with ThreadPoolExecutor(max_workers=32) as e:
    for d in the_dates:
        e.submit(combine_likelihoods, d)


# # Calculate the likelihood inside and recall for each vessel

# +

def calculate_likely_recall(the_date):
    q = f'''
    
    Create temp function the_date() as 
    (  date("{the_date:%Y-%m-%d}"));

CREATE TEMP FUNCTION YYYYMMDD(d DATE) AS (
      # Format a date as YYYYMMDD
      # e.g. DATE('2018-01-01') => '20180101'
      FORMAT_DATE('%Y%m%d',
        d) );

with 
vessel_info as (
  select 
    ssvid,
    best.best_length_m,
  from `{project_id}.gfw_research.vi_ssvid_v20220401`
),

extrap_raw as (
  select * from 
     (select *, row_number() over (partition by ssvid, scene_id order by rand()) row
       from 
      `{project_id}.proj_sentinel1_v20210924.detect_foot_ext_ais`
      WHERE DATE(_PARTITIONTIME) = the_date()
      )
   where row = 1
),

likelihood_table as (
select 
  ssvid, scene_id, avg(prob_inside) prob_inside
from 
  proj_global_sar.likelihood_inside
  where date(_partitiontime) = the_date()
  group by ssvid, scene_id
),

recall_table as 
(
select 
  length_m,
  min_distance_m,
  frac_matched3 as recall
from 
  proj_global_sar.s1recall 
),

extrap as (
SELECT 
  ssvid,
  scene_id, 
  likely_lon,
  likely_lat,
  least(abs(ifnull(delta_minutes1,1e10)), abs(ifnull(delta_minutes2,1e10))) min_delta_minutes,
  within_footprint,
  st_geogpoint(likely_lon, likely_lat) pos,
  within_footprint_5km,
  within_footprint_1km,
  best_length_m,
  case when best_length_m < 30 
    then floor(best_length_m/2)*2 + 1 
    when best_length_m between 30 and 59.9999999 
    then floor(best_length_m/4)*4 + 2 
    when best_length_m > 60  and best_length_m < 200
    then floor(best_length_m/20)*20 + 10 
    when best_length_m >= 200 then 200
    else -20 end length_m,
FROM 
    extrap_raw
left join 
  vessel_info
using(ssvid)
),

distance_to_closest as (
select 
  a.ssvid, 
  scene_id,
  min(st_distance(a.pos,b.pos)) min_distance_m
from 
  extrap a
join
  extrap b
using(scene_id)
where 
  a.ssvid != b.ssvid  
group by 
  ssvid, scene_id
),

extrap_distance_length as
(
select 
  ssvid,
  scene_id,
  min_delta_minutes,
  ifnull(min_distance_m, 1e10) actual_dist_to_nearest_m,
  case when min_distance_m < 1000 then 
    floor(min_distance_m/100)*100 
    else 1000 end min_distance_m,
  length_m,
  best_length_m
from 
  extrap a
left join
  distance_to_closest b
using(ssvid, scene_id)
),

extrap_recall as (
select 
  ssvid,
  scene_id,
  min_delta_minutes,
  recall,
  min_distance_m,
  actual_dist_to_nearest_m,
  length_m,
  best_length_m
from 
  extrap_distance_length
join
  recall_table
using(min_distance_m, length_m)
)

select * from extrap_recall
left join
likelihood_table
using(scene_id, ssvid)'''
    
    query_to_table(
        q, f"{project_id}.proj_global_sar.expected_recall${the_date:%Y%m%d}"
    )


# +
## run the first time to create the table

# command = f'bq mk --time_partitioning_type=DAY proj_global_sar.expected_recall'
# subprocess.call(command.split())
# -

with ThreadPoolExecutor(max_workers=32) as e:
    for d in the_dates:
        e.submit(calculate_likely_recall, d)


# # calculate actually inside scenes for AIS

def is_inside(the_date):
    q = f'''

with locations as 
(
select * from 
(select ssvid, scene_id, st_geogpoint(likely_lon, likely_lat) pos, row_number() over (partition by ssvid, scene_id order by rand()) row
from `proj_sentinel1_v20210924.detect_foot_ext_ais` 
where date(_partitiontime) = "{the_date:%Y-%m-%d}")
where row = 1),

footprints_table as
(select scene_id, ifnull(safe.st_geogfromtext(footprint_wkt),st_geogfromtext(footprint_wkt_1km)) scene
from 
  (select 
    distinct 
    scene_id,footprint_wkt,footprint_wkt_1km  
    from `proj_sentinel1_v20210924.detect_foot_raw_{the_date:%Y%m%d}` )
    where safe.st_geogfromtext(footprint_wkt) is not null or safe.st_geogfromtext(footprint_wkt_1km) is not null
 )


select ssvid,
 scene_id, 
st_contains(scene, pos) inside_scene from  
locations
left join
footprints_table
using(scene_id)'''
    query_to_table(
    q, f"{project_id}.proj_global_sar.likely_loc_in_scene${the_date:%Y%m%d}"
)


# +
## Run the first time
# command = f'bq mk --time_partitioning_type=DAY proj_global_sar.likely_loc_in_scene'
# subprocess.call(command.split())
# -

with ThreadPoolExecutor(max_workers=25) as e:
    for d in the_dates:
        e.submit(is_inside, d)


