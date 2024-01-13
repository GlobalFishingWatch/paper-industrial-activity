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

# # Match SAR and AIS 
#
# This notebook matches SAR and AIS by multiplying rasters. It makes a lot of tables and takes about two weeks, using almost all of GFW's slots, to run arcoss five years of data.
#
# Tables created:
#  - proj_global_sar.extrap_mult
#  - proj_global_sar.extrap_mult_single
#  - proj_global_sar.score_mult_0
#  - proj_global_sar.score_mult_1
#  - proj_global_sar.score_mult_2
#  - proj_global_sar.score_mult_3
#  - proj_global_sar.score_single
#  - proj_global_sar.matched_ave
#  - proj_global_sar.matched_mult
#  - proj_global_sar.matched_mult_recall
#  - proj_global_sar.matched_mult_recall_length
#  - proj_global_sar.matched_mult_recall_length
import sys
sys.path.append('../utils')
from bigquery_helper_functions import query_to_table
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import subprocess

import sys
sys.path.append('../utils') 
from proj_id import project_id

# ## check that all tables where made

tables = ''' - proj_global_sar.extrap_mult
 - proj_global_sar.extrap_mult_single
 - proj_global_sar.score_mult_0
 - proj_global_sar.score_mult_1
 - proj_global_sar.score_mult_2
 - proj_global_sar.score_mult_3
 - proj_global_sar.score_single
 - proj_global_sar.matched_ave
 - proj_global_sar.matched_mult
 - proj_global_sar.matched_mult_recall
 - proj_global_sar.matched_mult_recall_length
 - proj_global_sar.matched_mult_recall_length'''.replace(' - ','').split("\n")
tables

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

# +
## create date range to run

the_dates = np.arange(
    datetime(2017, 1, 1),
    datetime(2022, 1, 12) + timedelta(days=1),
    timedelta(days=1),
).astype(datetime)

# -

def create_extrap(the_date):
    
    
    

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



    with scored as 
    (select distinct ssvid, scene_id 

    from `{project_id}.proj_sentinel1_v20210924.detect_scene_score`
    where date(_partitiontime) = "{the_date:%Y-%m-%d}"
    and score > 1e-9
    ),
    
    
    vessel_info as 
    (  select best.best_vessel_class as label, ssvid from gfw_research.vi_ssvid_v20221001),
    
    extrapolate_table as (
    select * except(row) from
        (select *, row_number() over (partition by ssvid, scene_id order by rand()) row 
        from proj_sentinel1_v20210924.detect_foot_ext_doppler
        where date(_partitiontime) = "{the_date:%Y-%m-%d}")
    where row = 1
    )


    select 
    map_label(label) as label,
    rand() random_number,
     * except(label) from 
    extrapolate_table
    join 
    scored
    using(ssvid,scene_id) 
    left join
    vessel_info
    using(ssvid)
    where delta_minutes1 is not null and delta_minutes2 is not null
    and least(abs(delta_minutes1), abs(delta_minutes2)) >= 8 -- only where there is more than 8 minutes 
    -- the neareast ping


    '''

    query_to_table(q, f"{project_id}.proj_global_sar.extrap_mult${the_date:%Y%m%d}")


with ThreadPoolExecutor(max_workers=16) as e:
    for d in the_dates:
        e.submit(create_extrap, d)


def create_extrap_single(the_date):
    '''same as extrap, but creates for only the ones that are is_single'''
    
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



    with scored as 
    (select distinct ssvid, scene_id 

    from `{project_id}.proj_sentinel1_v20210924.detect_scene_score`
    where date(_partitiontime) = "{the_date:%Y-%m-%d}"
    and score > 1e-9
    ),
    
    
    vessel_info as 
    (  select best.best_vessel_class as label, ssvid from gfw_research.vi_ssvid_v20221001),
    
    extrapolate_table as (
    select * except(row) from
        (select *, row_number() over (partition by ssvid, scene_id order by rand()) row 
        from proj_sentinel1_v20210924.detect_foot_ext_doppler
        where date(_partitiontime) = "{the_date:%Y-%m-%d}")
    where row = 1
    )


    select 
    map_label(label) as label,
    rand() random_number,
     * except(label) from 
    extrapolate_table
    join 
    scored
    using(ssvid,scene_id) 
    left join
    vessel_info
    using(ssvid)
    where delta_minutes1 is null or delta_minutes2 is null



    '''

    query_to_table(q, f"{project_id}.proj_global_sar.extrap_mult_single${the_date:%Y%m%d}")



# +
## run this the first time to create the table 
# # !bq mk --time_partitioning_type=DAY proj_global_sar.extrap_mult_single
# -

with ThreadPoolExecutor(max_workers=32) as e:
    for d in the_dates:
        e.submit(create_extrap_single, d)


# # Multiply Rasters to Score

# +
# import subprocess
# for value in range(4):
#     command = f'bq mk --time_partitioning_type=DAY scratch_david.score_mult2_{value}'
#     subprocess.call(command.split())

# +
# import subprocess
# for value in range(4):
#     command = f'bq cp scratch_david.score_mult2_{value} proj_global_sar.score_mult_{value}'
#     subprocess.call(command.split())
# -

def score_day(the_date, value):

    extrap_table = f"proj_global_sar.extrap_mult"
    
    q = f'''#StandardSql

        CREATE TEMP FUNCTION bilinear_interpolation(Q11 float64,
        Q12 float64, Q22 float64, Q21 float64, x1 float64, x2 float64,
        y1 float64, y2 float64, x float64, y float64) AS (
            # see https://en.wikipedia.org/wiki/Bilinear_interpolation
            # Q11 is the value at x1, y1, Q12 is the value at x1, y2, etc.
            # x and y are the coordinates we want the value for
            1/( (x2 - x1) * (y2 - y1)) *
              (
                 Q11 * (x2 - x) * (y2 - y)
               + Q21 * (x - x1) * (y2 - y)
               + Q12 * (x2 - x) * (y - y1)
               + Q22 * (x - x1) * (y - y1)
              )
            );


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

        create temp function one_over_cellsize() as (
          200  -- 250 meter resolution roughly
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


        just_bounds_label as (
    --     select *,
    --     pixels_per_degree*cos(radians(min_lat/2+max_lat/2)) as pixels_per_degree_lon
    --     from scratch_david.temp_justbounds
       select
         ssvid,
        label,
        scene_id,
        greatest(scale1, scale2) as max_scale,
        speed1 as speed,
        course1 as course,
        scale1 as scale,
        round(max_scale*111) as pixels_per_degree,
        round(max_scale*111*cos(radians(ST_BOUNDINGBOX(lookup_raster).ymin/2+ST_BOUNDINGBOX(lookup_raster).ymin/2))) as pixels_per_degree_lon,
        least(abs(delta_minutes1), abs(delta_minutes2)) min_delta_minutes,
        is_single,
        delta_minutes1 as delta_minutes,
        ST_BOUNDINGBOX(lookup_raster).xmin as min_lon,
        ST_BOUNDINGBOX(lookup_raster).ymin as min_lat,
        ST_BOUNDINGBOX(lookup_raster).xmax as max_lon,
        ST_BOUNDINGBOX(lookup_raster).ymax as max_lat,
        lat_extrapolate1 + lat_doppler_offset as lat_interpolate_adjusted,
        lon_extrapolate1 + lon_doppler_offset as lon_interpolate_adjusted,
        from {extrap_table}
        where random_number >={value/4} and random_number < {value/4 + .25}
        and date(_partitiontime) = "{the_date:%Y-%m-%d}"
        union all
        select
        ssvid,
        label,
        scene_id,
        greatest(scale1, scale2) as max_scale,
        speed2 as speed,
        course2 as course,
        scale2 as scale,
        round(max_scale*111) as pixels_per_degree,
        round(max_scale*111*cos(radians(ST_BOUNDINGBOX(lookup_raster).ymin/2+ST_BOUNDINGBOX(lookup_raster).ymin/2))) as pixels_per_degree_lon,
        least(abs(delta_minutes1), abs(delta_minutes2)) min_delta_minutes,
        is_single,
        delta_minutes2 as delta_minutes,
        ST_BOUNDINGBOX(lookup_raster).xmin as min_lon,
        ST_BOUNDINGBOX(lookup_raster).ymin as min_lat,
        ST_BOUNDINGBOX(lookup_raster).xmax as max_lon,
        ST_BOUNDINGBOX(lookup_raster).ymax as max_lat,
        lat_extrapolate2 + lat_doppler_offset as lat_interpolate_adjusted,
        lon_extrapolate2 + lon_doppler_offset as lon_interpolate_adjusted,
        from {extrap_table}
        where random_number >={value/4} and random_number < {value/4 + .25}
        and date(_partitiontime) = "{the_date:%Y-%m-%d}"


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
         --  bilinear_interpolation(
         --  ifnull(probs_11.probability,0),
        --   ifnull(probs_12.probability,0),
        --   ifnull(probs_22.probability,0),
        --   ifnull(probs_21.probability,0),
        --   cast(x_key - .5 as int64), cast(x_key + .5 as int64),
        --   cast(y_key - .5 as int64), cast(y_key + .5 as int64) ,
        --   x_key, y_key) 
           probability
        --   -- to get at least one value.
        --   -- weight *should* be the same for each, but we need to make sure it isn't null
         from
           key_query_3
         left join
        -- joining on four locaitons to do bilinear interpolation
         probabilities_and_weights_neg  as probs_11
           on  probs_11.i = cast(x_key as int64) and probs_11.j = cast(y_key  as int64)
         --  on  probs_11.i = cast(x_key - .5 as int64) and probs_11.j = cast(y_key - .5 as int64)
           and probs_11.speed_lower = cast(speed_key as int64)
           and probs_11.minutes_lower = cast(minute_key as int64)
           and probs_11.labels = label_key
           where probability > 0
        -- left join
        -- probabilities_and_weights_neg  as probs_12
        --   on  probs_12.i = cast(x_key -.5 as int64) and probs_12.j = cast(y_key + .5 as int64)
        --   and probs_12.speed_lower = cast(speed_key as int64)
        --   and probs_12.minutes_lower = cast(minute_key as int64)
         --  and probs_12.labels = label_key
       --  left join
       --  probabilities_and_weights_neg  as probs_22
        --   on  probs_22.i = cast(x_key +.5 as int64) and probs_22.j = cast(y_key + .5 as int64)
         --  and probs_22.speed_lower = cast(speed_key as int64)
         --  and probs_22.minutes_lower = cast(minute_key as int64)
         --  and probs_22.labels = label_key
        -- left join
        -- probabilities_and_weights_neg  as probs_21
        --   on  probs_21.i = cast(x_key +.5 as int64) and probs_21.j = cast(y_key - .5 as int64)
        --   and probs_21.speed_lower = cast(speed_key as int64)
        --   and probs_21.minutes_lower = cast(minute_key as int64)
         --  and probs_21.labels = label_key
         ),



        prob_multiplied_table as
        (
          select
          ssvid,
          scene_id,
          a.probability*b.probability   probability,
          a.detect_lat as y_raw,
          a.detect_lon as x_raw
          from messages_with_probabilities a
          join
          messages_with_probabilities b
          using(ssvid, lat_bin, lon_bin, scene_id)
          where b.delta_minutes < a.delta_minutes
          and a.probability > 0 and b.probability > 0
        ),
        
            detections_ssvids as (
       select distinct a.detect_id, scene_id, b.ssvid,
       max_scale as scale,
       min_lat, min_lon, max_lat, max_lon,
         pixels_per_degree,
         pixels_per_degree_lon,
        cast(round((detect_lat-min_lat)*pixels_per_degree) as int64) as y_key,
        cast(round((detect_lon-min_lon)*pixels_per_degree*cos(radians(min_lat/2+max_lat/2))) as int64) as x_key
        from
         (select detect_lat, detect_lon, ssvid, scene_id, detect_id from proj_sentinel1_v20210924.detect_scene_score 
         where date(_partitiontime) = "{the_date:%Y-%m-%d}") a
         join just_bounds_label b
         using(ssvid, scene_id)
       ),
        
        
        

      --  normalized_prob as (
      --  select * except (probability),
      --  probability/prob_sum probability_unscalled,
      --   detect_lat as y_raw,
      --   detect_lon as x_raw
      --  from
      --  prob_multiplied_table
      --  join
      --  (select ssvid, scene_id, sum(probability) prob_sum from prob_multiplied_table
      --   group by ssvid, scene_id)
      --   using(ssvid,scene_id) ),







        scored_detects as (
    select
    a.scene_id scene_id,
    a.ssvid,
    "AIS" as source,
    detect_id,
    scale,
    --        bilinear_interpolation(
    --        ifnull(probs_11.probability,0),
    --        ifnull(probs_12.probability,0),
    --        ifnull(probs_22.probability,0),
    --        ifnull(probs_21.probability,0),
    --        cast(round(x_key - .5) as int64), cast(round(x_key + .5) as int64),
    --        cast(round(y_key - .5) as int64), cast(round(y_key + .5) as int64) ,
    --        x_key, y_key) as
           --probability_unscalled*(scale*scale) as score
    probability
    from
    detections_ssvids a
    left join
    prob_multiplied_table as probs_11
    on cast(round(y_key) as int64) =   cast(round((y_raw-min_lat)*pixels_per_degree) as int64)
    and cast(round(x_key) as int64) = cast(round((x_raw-min_lon)*pixels_per_degree_lon) as int64)
    and a.scene_id = probs_11.scene_id and a.ssvid = probs_11.ssvid

    )

    select 
    scene_id,
    ssvid,
    detect_id,
    max(b.probability)/sum(b.probability) * scale * scale as max_score,
    a.probability/sum(b.probability) * scale * scale as score,
    sum(if(b.probability <= a.probability, b.probability, 0))/sum(b.probability) as likelihood
    from 
    scored_detects a 
    join
    prob_multiplied_table  b
    using(ssvid, scene_id)
    where a.probability > 0
    group by scene_id, ssvid, detect_id, a.probability, scale
    
    
    '''
#     pyperclip.copy(q)
    query_to_table(
        q, f"{project_id}.proj_global_sar.score_mult_{value}${the_date:%Y%m%d}"
    )


with ThreadPoolExecutor(max_workers=8) as e:
    for d in the_dates:
        for value in range(4):
            e.submit(score_day, d, value)






# # Caluclate for single vessels

def score_day_single(the_date):
    
    extrap_table = f"proj_global_sar.extrap_mult_single"
    
    q = f'''#StandardSql

    create temp function radians(x float64) as (
      3.14159265359 * x / 180
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

    probability_table as (select * from
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
  --
  --
  -- weight is a measure of how concetrated the raster is, and is used to assess score
  weights as (
  select
    # this is because the value at 0 is not mirrored
    sum(if(i>0, 2*probability *probability, probability *probability )) as weight,
    pow(sum(if(i>0, 2*probability, probability  )),.5) original_raster_scale,
    labels,
    minutes_lower,
    speed_lower
  from
    probability_table
  group by
    labels,
    minutes_lower,
    speed_lower
  ),
  --
  -- combine probabilities and weights into one table
  probabilities_and_weights as (
  select * from
    probability_table
  join
    weights
  using(labels, minutes_lower, speed_lower)
  ),
  --
  --
  -- the raster has only positive i values because it is symmetrical
  -- around the i axis (the boat is as equally likely to turn left as right).
  -- Make a full raster with negative values for the later join.
  probabilities_and_weights_neg as (
    select
      labels, minutes_lower, speed_lower, probability, i, j, weight, original_raster_scale
    from
      probabilities_and_weights
    union all
    select
    -- same except take negative i!
      labels, minutes_lower, speed_lower, probability, -i as i, j, weight, original_raster_scale
    from
      probabilities_and_weights
    where i > 0
  ),

    #########################
    ##
    ## SAR subqueries
    ##
    #########################
    --
    --


    just_bounds_label as (
  select
    ssvid,
    label,
    scene_id,
    greatest(ifnull(scale1,0), ifnull(scale2,0)) as max_scale,
    speed1 as speed,
    course1 as course,
    scale1 as scale,
    round(max_scale*111) as pixels_per_degree,
    round(max_scale*111*cos(radians(ST_BOUNDINGBOX(lookup_raster).ymin/2+ST_BOUNDINGBOX(lookup_raster).ymin/2))) as pixels_per_degree_lon,
    least(ifnull(abs(delta_minutes1),1e10), ifnull(abs(delta_minutes2),1e10)) min_delta_minutes,
    is_single,
    delta_minutes1 as delta_minutes,
    ST_BOUNDINGBOX(lookup_raster).xmin as min_lon,
    ST_BOUNDINGBOX(lookup_raster).ymin as min_lat,
    ST_BOUNDINGBOX(lookup_raster).xmax as max_lon,
    ST_BOUNDINGBOX(lookup_raster).ymax as max_lat,
    lat_extrapolate1 + lat_doppler_offset as lat_interpolate_adjusted,
    lon_extrapolate1 + lon_doppler_offset as lon_interpolate_adjusted,
    from {extrap_table}
    where delta_minutes1 is not null
    and date(_partitiontime) = "{the_date:%Y-%m-%d}"
    union all
    select
    ssvid,
    label,
    scene_id,
    greatest(ifnull(scale1,0), ifnull(scale2,0)) as max_scale,
    speed2 as speed,
    course2 as course,
    scale2 as scale,
    round(max_scale*111) as pixels_per_degree,
    round(max_scale*111*cos(radians(ST_BOUNDINGBOX(lookup_raster).ymin/2+ST_BOUNDINGBOX(lookup_raster).ymin/2))) as pixels_per_degree_lon,
    least(ifnull(abs(delta_minutes1),1e10), ifnull(abs(delta_minutes2),1e10)) min_delta_minutes,
    is_single,
    delta_minutes2 as delta_minutes,
    ST_BOUNDINGBOX(lookup_raster).xmin as min_lon,
    ST_BOUNDINGBOX(lookup_raster).ymin as min_lat,
    ST_BOUNDINGBOX(lookup_raster).xmax as max_lon,
    ST_BOUNDINGBOX(lookup_raster).ymax as max_lat,
    lat_extrapolate2 + lat_doppler_offset as lat_interpolate_adjusted,
    lon_extrapolate2 + lon_doppler_offset as lon_interpolate_adjusted,
    from {extrap_table}
    where delta_minutes2 is not null
    and date(_partitiontime) = "{the_date:%Y-%m-%d}"
    ),



    
  detections_ssvids as (
  select distinct a.detect_id, scene_id, b.ssvid,
  max_scale as scale,
  min_lat, min_lon, max_lat, max_lon, score,
  map_speed(speed) as speed_key,
  map_minutes(delta_minutes) as minute_key,
  map_label(label) as label_key,
    pixels_per_degree,
    pixels_per_degree_lon,
    cast(round((detect_lat-min_lat)*pixels_per_degree) as int64) as y_key,
    cast(round((detect_lon-min_lon)*pixels_per_degree*cos(radians(min_lat/2+max_lat/2))) as int64) as x_key
    from
    (select detect_lat, detect_lon, ssvid, scene_id, detect_id, score from proj_sentinel1_v20210924.detect_scene_score 
    where date(_partitiontime) = "{the_date:%Y-%m-%d}") a
    join just_bounds_label b
    using(ssvid, scene_id)
  )


  
    


select 
scene_id,
ssvid,
detect_id,
max(probability)/sum(probability) * scale * scale as max_score,
score as score,
sum(if(probability * ( scale * scale ) / ( original_raster_scale*original_raster_scale) <= score, 
      probability, 0))/sum(probability) as likelihood
from 
detections_ssvids a 
join
probabilities_and_weights_neg  
on labels = label_key
and speed_lower = speed_key
and minutes_lower = minute_key
where probability > 0
group by scene_id, ssvid, detect_id, scale, score, original_raster_scale
    
    
    
    
    
      '''
#     pyperclip.copy(q)
    query_to_table(
        q, f"{project_id}.proj_global_sar.score_single${the_date:%Y%m%d}"
    )


with ThreadPoolExecutor(max_workers=10) as e:
    for d in the_dates:
        e.submit(score_day_single, d)

# +
## Run the first time
# command = f'bq mk --time_partitioning_type=DAY proj_global_sar.score_combined'
# subprocess.call(command.split())
# -

# # Combine Scores, Include Factors for Recall and Length Mismatch

# +
q = '''with score_mult as 

(select ssvid , score , detect_id, 
b.length_m as length_m_sar,
best.best_length_m length_m_ais
from  
proj_sentinel1_v20210924.detect_scene_match a
join
`proj_sentinel1_v20210924.detect_scene_pred_*` b
using(detect_id)
join
`gfw_research.vi_ssvid_v20221001`
using(ssvid)
where
_table_suffix between '20170101' and '20210131'
and date(_partitiontime) between "2017-01-01" and "2021-12-31" 
and score > 10
and confidence > .99
),

counts as 
(select
floor((length_m_ais - length_m_sar)/length_m_ais*10)/10 as frac_diff ,
count(*) number
 from score_mult
 group by frac_diff
 ),
 
total_table as
(select sum(number) as total 
from counts
)
 

select 
frac_diff,
number/total probability,
from counts 
cross join
total_table '''


### UNCOMMENT TO REGENERATE -- this only needs to be generated once
# query_to_table(
#     q, f"{project_id}.proj_global_sar.frac_diff_lookup"
# )
# -

def make_combined_scored_tables_length(the_date):
    q = f'''Create temp function the_date() as 
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

sar_length_table as (
 select 
   detect_id, avg(length_m) as sar_length_m 
 from 
  `proj_sentinel1_v20210924.detect_scene_pred_*` 
 where 
  _table_suffix = YYYYMMDD(the_date())
  and presence > .7 -- only match with score over .7
 group by detect_id
),

old_score as (
  SELECT 
  distinct
  scene_id,
  ssvid,
  source,
  detect_id,
  detect_timestamp,
  detect_lat,
  detect_lon,
  timestamp1
  timestamp2,
  seg_id,
  score as score_ave
FROM 
  `{project_id}.proj_sentinel1_v20210924.detect_scene_score` 
WHERE DATE(_PARTITIONTIME) = the_date()
and source = "AIS"
),

extrap_recall  as (
select 
  ssvid,
  scene_id,
  min_delta_minutes,
  recall,
  min_distance_m,
  length_m,
  best_length_m,
  prob_inside,
  from proj_global_sar.expected_recall
WHERE DATE(_PARTITIONTIME) = the_date()
),


new_score_raw as 
(
select 
  detect_id, scene_id, ssvid, score as score_mult, likelihood, max_score
from proj_global_sar.score_mult_0 
where date(_partitiontime) = the_date()
union all
select
  detect_id, scene_id, ssvid, score as score_mult, likelihood, max_score
from proj_global_sar.score_mult_1 
where date(_partitiontime) = the_date()
union all
select       
  detect_id, scene_id, ssvid, score as score_mult, likelihood, max_score
from proj_global_sar.score_mult_2 
where date(_partitiontime) = the_date()
union all
select       
  detect_id, scene_id, ssvid, score as score_mult, likelihood, max_score
from proj_global_sar.score_mult_3 
where date(_partitiontime) = the_date()
union all
select       
  detect_id, scene_id, ssvid, score as score_mult, likelihood, max_score
from proj_global_sar.score_single 
where date(_partitiontime) = the_date()
),


new_score as(
-- just in case there are duplicates...
select 
  detect_id, ssvid, scene_id, 
  avg(score_mult) as score_mult, avg(likelihood) as likelihood, avg(max_score) as max_score
from 
  new_score_raw
group by 
  detect_id, ssvid, scene_id
),


without_row_numbers as 
(select *,
  if(min_delta_minutes<10 or score_mult is null, score_ave, score_mult) as score,
 -- when the vessel doesn't have a length in the vessel info table, 
  -- it is assinged a length_m of -20... the average value for matches
  -- is 0.09 --- but then make it half this value because why not
-- here's where that number came from:
-- with matched_table as 
-- (select ssvid, detect_id  from proj_global_sar.matched_mult_recall_length
-- where score > 1e-5)
-- select avg(prob_length_match) prob_length_match,
-- avg(recall) recall,
-- avg(prob_inside) prob_inside,
-- avg(prob_length_match*recall*prob_inside) factor_reduced
-- from proj_global_sar.score_combined
-- join matched_table
-- using(ssvid, detect_id)
  case when probability is null and length_m = -20 then .09*.5
  when probability is null then 1e-6
  else probability end as  prob_length_match
from 
  old_score a
left join
  new_score c
using(detect_id, ssvid, scene_id)
left join
  extrap_recall
using(scene_id, ssvid)
join
  sar_length_table
using(detect_id) 
left join
proj_global_sar.frac_diff_lookup
on frac_diff = floor((best_length_m - sar_length_m)/best_length_m*10)/10
)


select 
  *,
  ROW_NUMBER() OVER (PARTITION BY detect_id ORDER BY score DESC) as row_number_detect_id_mult,
  ROW_NUMBER() OVER (PARTITION BY concat (ssvid, source), scene_id ORDER BY score DESC) row_number_ssvid_mult,

  ROW_NUMBER() OVER (PARTITION BY detect_id ORDER BY score_ave DESC) as row_number_detect_id_ave,
  ROW_NUMBER() OVER (PARTITION BY concat (ssvid, source), scene_id ORDER BY score_ave DESC) row_number_ssvid_ave, 

  ROW_NUMBER() OVER (PARTITION BY detect_id ORDER BY score*recall*prob_inside DESC) as row_number_detect_id_mult_recall,
  ROW_NUMBER() OVER (PARTITION BY concat (ssvid, source), scene_id ORDER BY score*recall*prob_inside DESC) row_number_ssvid_mult_recall,

  ROW_NUMBER() OVER (PARTITION BY detect_id ORDER BY score_ave*recall*prob_inside DESC) as row_number_detect_id_ave_recall,
  ROW_NUMBER() OVER (PARTITION BY concat (ssvid, source), scene_id ORDER BY score_ave*recall*prob_inside DESC) row_number_ssvid_ave_recall, 

  ROW_NUMBER() OVER (PARTITION BY detect_id ORDER BY score*recall*prob_inside*prob_length_match DESC) as row_number_detect_id_mult_recall_len,
  ROW_NUMBER() OVER (PARTITION BY concat (ssvid, source), scene_id ORDER BY score*recall*prob_inside*prob_length_match DESC) row_number_ssvid_mult_recall_len,  

  ROW_NUMBER() OVER (PARTITION BY detect_id ORDER BY score*prob_length_match DESC) as row_number_detect_id_mult_len,
  ROW_NUMBER() OVER (PARTITION BY concat (ssvid, source), scene_id ORDER BY score*prob_length_match DESC) row_number_ssvid_mult_len,

  ROW_NUMBER() OVER (PARTITION BY detect_id ORDER BY score_ave*prob_length_match DESC) as row_number_detect_id_ave_len,
  ROW_NUMBER() OVER (PARTITION BY concat (ssvid, source), scene_id ORDER BY score_ave*prob_length_match DESC) row_number_ssvid_ave_len,

from
  without_row_numbers  '''
    
    query_to_table(
        q, f"{project_id}.proj_global_sar.score_combined${the_date:%Y%m%d}"
    )


# +
### Run the first time to create the table

# command = f'bq mk --time_partitioning_type=DAY proj_global_sar.score_combined' 
# subprocess.call(command.split())
# -

with ThreadPoolExecutor(max_workers=32) as e:
    for d in the_dates:
        e.submit(make_combined_scored_tables_length, d)


def match_tables(the_date, 
                    score_field = "score", 
                   row_number_detect_id = "row_number_detect_id_mult",
                   row_number_ssvid_field = "row_number_ssvid_mult",
                   output_table = "proj_global_sar.matched_mult",
                   scored_table = "proj_global_sar.score_combined"):
    
    q = f'''Create temp function the_date() as 
    (  date("{the_date:%Y-%m-%d}"));

CREATE TEMP FUNCTION YYYYMMDD(d DATE) AS (
      # Format a date as YYYYMMDD
      # e.g. DATE('2018-01-01') => '20180101'
      FORMAT_DATE('%Y%m%d',
        d) );

WITH
scores_ranked as ( 
  select 
    concat(ssvid, source) as ssvid_source, 
    {score_field} as score,
    {row_number_detect_id} as row_number_detect_id,
    {row_number_ssvid_field} row_number_ssvid,
    * except(score) 
  from
    {scored_table} 
  where DATE(_PARTITIONTIME) =  the_date()
    and source = "AIS"
),

ssvid_scores as (
  select 
    ssvid, source, scene_id, sum(score) ssvid_score_sum from scores_ranked
  group by 
    ssvid, scene_id, source
),

detect_id_scores as (
  select 
    detect_id, scene_id, sum(score) detect_id_score_sum from scores_ranked
  group by 
    detect_id, scene_id
),

objects_table as (
  select
    distinct
    detect_id,
    scene_id,
    detect_lon,
    detect_lat,
    TIMESTAMP_ADD(start_time,
                  INTERVAL  cast(timestamp_diff(end_time,
                                start_time, SECOND)/2 as int64) SECOND)
  as 
    detect_timestamp
  from 
    `proj_sentinel1_v20210924.detect_scene_raw_*`
  where
    _table_suffix = YYYYMMDD(the_date())
),

top_matches as (
  select * from scores_ranked
  where row_number_detect_id = 1 and row_number_ssvid = 1
),

second_matches_ranked as (
  select 
    *, row_number() over
     (partition by detect_id order by score desc) row_number_detect_id_2nd,
    row_number() over
      (partition by ssvid, scene_id order by score desc) row_number_ssvid_2nd
  from 
    scores_ranked
  where 
    concat(ssvid, scene_id) not in (select concat(ssvid, scene_id) from top_matches)
    and detect_id not in (select detect_id from top_matches)
),

second_matches as (
  select * 
  from 
    second_matches_ranked 
  where 
    row_number_detect_id_2nd = 1 and row_number_ssvid_2nd = 1
),

third_matches_ranked as
(
    select *, row_number() over
  (partition by detect_id order by score desc) row_number_detect_id_3rd,
    row_number() over
  (partition by ssvid, scene_id order by score desc) row_number_ssvid_3rd
from 
  second_matches_ranked
where
  concat(ssvid, scene_id) not in (select concat(ssvid, scene_id) from second_matches)
  and detect_id not in (select detect_id from second_matches)
),

third_matches as (
  select 
    * 
  from 
    third_matches_ranked 
  where row_number_detect_id_3rd = 1 and row_number_ssvid_3rd = 1
),

forth_matches_ranked as (
  select 
    *, 
    row_number() over
       (partition by detect_id order by score desc) row_number_detect_id_4th,
    row_number() over
        (partition by ssvid, scene_id order by score desc) row_number_ssvid_4th
  from 
    third_matches_ranked
  where
    concat(ssvid, scene_id) not in (select concat(ssvid, scene_id) from third_matches)
    and detect_id not in (select detect_id from third_matches)
),

fourth_matches as (
  select * 
  from 
    forth_matches_ranked 
  where row_number_detect_id_4th = 1 and row_number_ssvid_4th = 1
),

top_4_matches as (
  select 
    * from fourth_matches
  union all
   select *, null as row_number_detect_id_4th,
    null as row_number_ssvid_4th from third_matches
    union all
    select *,
    null as row_number_detect_id_4th,
    null as row_number_ssvid_4th,
    null as row_number_detect_id_3rd,
    null as row_number_ssvid_3rd
  from 
    second_matches
  union all
  select *,
    null as row_number_detect_id_4th,
    null as row_number_ssvid_4th,
    null as row_number_detect_id_3rd,
    null as row_number_ssvid_3rd,
    null as row_number_detect_id_2nd,
    null as row_number_ssvid_2nd
  from 
    top_matches
  order by
    row_number_detect_id,
    row_number_ssvid
),


joined_back_with_detections as (
  select
  b.scene_id scene_id,
  ssvid,
  source,
  detect_id,
  b.detect_timestamp detect_timestamp,
  b.detect_lon detect_lon,
  b.detect_lat detect_lat,
  * except(scene_id, ssvid, source, score, detect_lon, detect_lat, detect_timestamp, ssvid_source,
row_number_detect_id, row_number_ssvid, row_number_detect_id_2nd, row_number_ssvid_2nd,
row_number_detect_id_3rd, row_number_ssvid_3rd, row_number_detect_id_4th, row_number_ssvid_4th,
detect_id),
  ifnull(score, 0) score
  from 
    top_4_matches a
  full outer join 
    objects_table b
  using(detect_id)
)

select  
  scene_id, ssvid, source, detect_id, detect_timestamp, detect_lat, detect_lon, score,
  * except(scene_id, ssvid, source, detect_id, detect_timestamp, detect_lon, detect_lat, score),
   safe_divide(score, pow(ssvid_score_sum*detect_id_score_sum, .5) ) confidence
from 
  joined_back_with_detections
left join 
  ssvid_scores
using(ssvid, scene_id, source)
left join 
  detect_id_scores
using(detect_id, scene_id) '''
#     pyperclip.copy(q)
    
    query_to_table(
        q, f"{project_id}.{output_table}${the_date:%Y%m%d}"
    )



# +
## to be run the first time to create the tables

# command = f'bq mk --time_partitioning_type=DAY proj_global_sar.matched_mult'
# subprocess.call(command.split())
# command = f'bq mk --time_partitioning_type=DAY proj_global_sar.matched_ave'
# subprocess.call(command.split())
# command = f'bq mk --time_partitioning_type=DAY proj_global_sar.matched_mult_recall'
# subprocess.call(command.split())
# command = f'bq mk --time_partitioning_type=DAY proj_global_sar.matched_mult_recall_length'
# subprocess.call(command.split())
# command = f'bq mk --time_partitioning_type=DAY proj_global_sar.matched_mult_length'
# subprocess.call(command.split())
# command = f'bq mk --time_partitioning_type=DAY proj_global_sar.matched_ave_length'
# subprocess.call(command.split())



# -

with ThreadPoolExecutor(max_workers=16) as e:
    for d in the_dates:
        e.submit(match_tables, d, score_field = "score_ave", 
                   row_number_detect_id = "row_number_detect_id_ave",
                   row_number_ssvid_field = "row_number_ssvid_ave",
                   output_table = "proj_global_sar.matched_ave",
                   scored_table = "proj_global_sar.score_combined")

with ThreadPoolExecutor(max_workers=16) as e:
    for d in the_dates:
        e.submit(match_tables, d, score_field = "score", 
                   row_number_detect_id = "row_number_detect_id_mult",
                   row_number_ssvid_field = "row_number_ssvid_mult",
                   output_table = "proj_global_sar.matched_mult",
                   scored_table = "proj_global_sar.score_combined")

with ThreadPoolExecutor(max_workers=16) as e:
    for d in the_dates:
        e.submit(match_tables, d, score_field = "score*recall*prob_inside", 
                   row_number_detect_id = "row_number_detect_id_mult_recall",
                   row_number_ssvid_field = "row_number_ssvid_mult_recall",
                   output_table = "proj_global_sar.matched_mult_recall",
                   scored_table = "proj_global_sar.score_combined")

with ThreadPoolExecutor(max_workers=16) as e:
    for d in the_dates:
        e.submit(match_tables, d, score_field = "score*recall*prob_inside*prob_length_match", 
                   row_number_detect_id = "row_number_detect_id_mult_recall_len",
                   row_number_ssvid_field = "row_number_ssvid_mult_recall_len",
                   output_table = "proj_global_sar.matched_mult_recall_length",
                    scored_table = "proj_global_sar.score_combined")

with ThreadPoolExecutor(max_workers=16) as e:
    for d in the_dates:
        e.submit(match_tables, d, score_field = "score*prob_length_match", 
                   row_number_detect_id = "row_number_detect_id_mult_len",
                   row_number_ssvid_field = "row_number_ssvid_mult_len",
                   output_table = "proj_global_sar.matched_mult_length",
                    scored_table = "proj_global_sar.score_combined")

with ThreadPoolExecutor(max_workers=16) as e:
    for d in the_dates:
        e.submit(match_tables, d, score_field = "score_ave*prob_length_match", 
                   row_number_detect_id = "row_number_detect_id_ave_len",
                   row_number_ssvid_field = "row_number_ssvid_ave_len",
                   output_table = "proj_global_sar.matched_ave_length",
                    scored_table = "proj_global_sar.score_combined")


