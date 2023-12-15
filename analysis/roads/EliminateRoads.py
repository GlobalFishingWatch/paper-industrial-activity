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

# # EliinateRoads
#
# This notebook finds, for every SAR secene, the area where fast moving trucks might appear in the ocean due to the doppler shift. Because SAR uses the frequency of the return signal to spatialy resolve where an object is in the scene, a vessel moving toward or away from the satellite will cause a shift in the frequency of return, and the the object will "appear" in the image in a different place than it actually was. This offset is accounted for in our matching algorithm. An explanation is also provided [here](https://geoawesomeness.com/eo-hub/moving-objects-and-their-displacement-in-sar-images/). 
#
# To calcluate this, we we unioned all footprints and then found all highways and primary roads (we ignored secondary roads, as trucks are less likely to drive quickly on these) that were within 3km of the area imaged by Sentinel-1.
#
# Then we turned each road into a series of point pairs, calculated a course from these, and then estimated where a truck moving at 85 miles per hour on a highway or 50 miles per hour on a primary road would appear in the SAR scene. For each scene we then created a polygon that is bounded by where these vessels might appear, and this is an area to exclude for our analyses.

# +
import sys
sys.path.append('../utils')
from bigquery_helper_functions import (
    update_table_description,
    query_to_table,
)

from datetime import datetime, timedelta
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
# -

# # Union all the footprints...

q_template = '''SELECT st_union_agg(
  st_geogfromtext(
  footprint_wkt)) as unioned FROM `project-id.proj_sentinel1_v20210924.detect_foot_raw_{suffix}` 
  where safe.st_geogfromtext(
  footprint_wkt) is not null'''


def process_day(date):
    table_id = f'project-id.scratch_david.s1f_{date:%Y%m%d}'
    query_to_table(q_template.format(suffix=f"{date:%Y%m%d}"), table_id)


the_dates = np.arange(
    datetime(2017, 1, 1),
    datetime(2021, 12, 31) + timedelta(days=1),
    timedelta(days=1),
).astype(datetime)

with ThreadPoolExecutor(max_workers=16) as e:
    for d in the_dates:
        e.submit(process_day, d)

# +

q_grouped = '''select st_union_agg(st_simplify(unioned,500)) as unioned from `scratch_david.s1f_{YYYYMM}*`'''

for year in range(2018,2022):
    for month in range(1,13):
        d = datetime(year,month,1)
        YYYYMM = f"{d:%Y%m}"

        table_id = f'project-id.scratch_david.s1m_{YYYYMM}'
        query_to_table(q_grouped.format(YYYYMM=f"{YYYYMM}"), table_id)

# -

for year in range(2017,2022):
    q_grouped_year = f'''
    select st_union_agg(st_simplify(unioned,500)) as unioned 
    from `scratch_david.s1m_{year}*`'''
    table_id = f'project-id.scratch_david.s1y_{year}'
    query_to_table(q_grouped_year, table_id)


q = ''' 
select st_union_agg(unioned) as unioned 
    from `project-id.scratch_david.s1y_*`
    where _table_suffix between '2017' and '2021' 
'''
table_id =  f'project-id.scratch_david.s1_footprint_2017_2022'
query_to_table(q, table_id)




q = '''with roads as

(select safe.st_geogfromtext(wkt) road from `scratch_david.GRIP4_region*` 
where safe.st_geogfromtext(wkt) is not null),

interesected_roads as 
(select st_intersection(road,unioned) as bridges
from roads
cross join
scratch_david.s1_footprint_2017_2022
)

select * from interesected_roads


'''
table_id =  f'project-id.scratch_david.s1_footprint_bridges'
query_to_table(q, table_id)


# +
q = '''select 
st_union_agg(st_buffer(bridges,3500)) as near_bridges_3500,
st_union_agg(st_buffer(bridges,3000)) as near_bridges_3000,
st_union_agg(st_buffer(bridges,2500)) as near_bridges_2500,
st_union_agg(st_buffer(bridges,2000)) as near_bridges_2000,
st_union_agg(st_buffer(bridges,1500)) as near_bridges_1500,
st_union_agg(st_buffer(bridges,1000)) as near_bridges_1000,
from
scratch_david.s1_footprint_bridges
'''

table_id =  f'project-id.scratch_david.area_near_bridges'
query_to_table(q, table_id)

# -

# ## Get rid of all non-polygons
# The aggregation gets a bunch of multistring objects that are not things we want.
# We will download, remove, and upload again.

# +
q = '''select * from scratch_david.s1_footprint_2017_2022'''

df = pd.read_gbq(q)
# -

import shapely
s = shapely.wkt.loads(df.unioned.values[0])

types = []
wkt=[]
for ob in s:
    if type(ob)==shapely.geometry.multipolygon.MultiPolygon:    
        for o in ob:
            wkt.append({"wkt":o.wkt})
dff = pd.DataFrame(wkt)
len(dff)

dff.to_gbq("scratch_david.s1_footprint_split_2017_2022", if_exists='replace')

# +
q = '''with area_buffered as 
(

select st_simplify(st_union_agg(buffered),500) as unioned_3500 from
(select st_buffer(st_geogfromtext(wkt),3500) buffered
 from scratch_david.s1_footprint_split_2017_2022))
 
 select * from area_buffered
 '''

df = pd.read_gbq(q)


# -

s = shapely.wkt.loads(df.unioned_3500.values[0])
len(s)

len(s.wkt)

types = []
wkt=[]
for ob in s:
    wkt.append({"wkt":ob.wkt})
dff = pd.DataFrame(wkt)
len(dff)

dff.to_gbq('scratch_david.s1_footprint_split_2017_2022')

# # Buffer out 3.5 km from each scene

# +
q = '''with area_buffered as 
(

select st_simplify(st_union_agg(buffered),500) as unioned_3500 from
(select st_buffer(st_geogfromtext(wkt),3500) buffered
 from scratch_david.s1_footprint_split_2017_2022)),

  roads as

(select * except(wkt), safe.st_geogfromtext(wkt) road
from `scratch_david.GRIP4_region*` 
where safe.st_geogfromtext(wkt) is not null
and  GP_RTP in (1,2))

select st_intersection(road,unioned_3500) road_clipped, *
from roads
cross join
area_buffered
where st_distance(road,unioned_3500) = 0
'''

table_id =  f'project-id.scratch_david.s1_footprint_bridges_3500buffer_highway_primary'
query_to_table(q, table_id)

# this takes a long time!
# -

# # Now get roads, simplified

# +
# st_simplify was turning lines into points, which I didn't like! Not sure
# why it was doing that

q = '''select
st_astext(case when st_length(road_100)>0 then road_100
when st_length(road_50) >0 then road_50
else road_clipped end
) as road_wkt,
GP_RTP
from
(select 
st_simplify(road_clipped,50) as road_50,
st_simplify(road_clipped,100) as road_100,
road_clipped,
GP_RTP
from 
scratch_david.s1_footprint_bridges_3500buffer_highway_primary)'''


q = '''
(select 
st_astext(road_clipped) road_wkt,
GP_RTP
from 
scratch_david.s1_footprint_bridges_3500buffer_highway_primary)'''

dfr = pd.read_gbq(q)
# -

len(dfr)

len(dfr)

dfr.head()

r = dfr.road_wkt.values[0]
s = shapely.wkt.loads(r)
s

for c in s.coords:
    print(c)

for i in range(len(s.coords)-1):
    lon1 = s.coords[i][0]
    lat1 = s.coords[i][1]
    lon2 = s.coords[i+1][0]
    lat2 = s.coords[i+1][1]




new_rows = []
for index, row in dfr.iterrows():
    road_id = index
    s = shapely.wkt.loads(row.road_wkt)
    if type(s) == shapely.geometry.multilinestring.MultiLineString:
        for geom in s:
            c = geom.coords
            for i in range(len(c)):
                lon1 = c[i][0]
                lat1 = c[i][1]
                if i == len(c)-1:
                    adjust = - 2
                else:
                    adjust = 0
                lon2 = c[i+1 + adjust][0]
                lat2 = c[i+1 + adjust][1]
                x = (lon2-lon1)*np.cos(np.pi*lat1/180)
                y = lat2-lat1
                if i == len(c)-1:
                    x = -x
                    y = -y
                course = np.arctan2(x,y)/np.pi*180
                new_rows.append({'road_id':road_id,
                                'lon':lon1,
                                'lat':lat1,
                                'course':course,
                                'GP_RTP':row.GP_RTP})
            
    elif type(s) == shapely.geometry.linestring.LineString:
        c = s.coords
        for i in range(len(c)):
            lon1 = c[i][0]
            lat1 = c[i][1]
            # for the last point, use the previous one in the sequence
            if i == len(c)-1:
                adjust = - 2
            else:
                adjust = 0
            lon2 = c[i+1 + adjust][0]
            lat2 = c[i+1 + adjust][1]
            x = (lon2-lon1)*np.cos(np.pi*lat1/180)
            y = lat2-lat1
            if i == len(c)-1:
                x = -x
                y = -y
            course = np.arctan2(x,y)/np.pi*180
            new_rows.append({'road_id':road_id,
                            'lon':lon1,
                            'lat':lat1,
                            'course':course,
                            'GP_RTP':row.GP_RTP})
    else:
        print(type(s))
        break

df_road_points = pd.DataFrame(new_rows)

df_road_points.head()

df_road_points.to_gbq('scratch_david.highway_primary_road_pairs', if_exists='replace')

# # Now we have what we want. A series of points with courses. Just add speed and apply doppler offset!

q_road_doppler_template ='''create temp function radians(x float64) as (
  3.14159265359 * x / 180
);

create temp function degrees(x float64) as (
    x * 180 / 3.14159265359
);

create temp function meters_per_nautical_mile() as (
  1852
);

create temp function get_course(point1 geography, point2 geography) as ((
  -- Equation are from http://www.movable-type.co.uk/scripts/latlong.html
  -- assumes a spherical earth, which, of course, is only mostly right

  --  BEARING
  -- (which is measured, apparently, counterclockwise from due east, so
  -- we edited to make it clockwise from due north
  --        const y = Math.sin(λ2-λ1) * Math.cos(φ2);
  -- const x = Math.cos(φ1)*Math.sin(φ2) -
  --           Math.sin(φ1)*Math.cos(φ2)*Math.cos(λ2-λ1);
  -- const θ = Math.atan2(y, x);
  -- const brng = (θ*180/Math.PI + 360) % 360; // in degrees
  -- λ1 is lon1,  λ2 is lon2, φ1 is lat1, φ2 is lat2, measured in radians

        select (90 - degrees(atan2(x, y))) as course
        from
            (select
                    sin(rlon2 - rlon1) * cos(rlat2) as y,
                    cos(
                        rlat1
                    ) * sin(
                        rlat2
                    ) - sin(rlat1) * cos(rlat2) * cos(rlon2 - rlon1) as x
                from
                    (select
                            radians(st_x(point1)) as rlon1,
                            radians(st_y(point1)) as rlat1,
                            radians(st_x(point2)) as rlon2,
                            radians(st_y(point2)) as rlat2))

));


create temp function earth_radius_km(lat float64) as
-- this is super overkill. You could just use the average
-- radius of the earth. But I wanted to see if it made a difference.
-- It matters if you want > 4 significant digits.
-- But, once I made it, I didn't feel like deleting it.
-- Equation taken from https://rechneronline.de/earth-radius/
((
select
  --  R = √ [ (r1² * cos(B))² + (r2² * sin(B))² ] / [ (r1 * cos(B))² + (r2 * sin(B))² ]
  pow(
  ( pow(r1*r1 * cos(B), 2) + pow(r2*r2 * sin(B), 2) )
  /
  ( pow(r1 * cos(B), 2) + pow(r2 * sin(B), 2) )
  , .5)
    from
    (select
    6371.001 as r1,
    6356.752 as r2,
    Radians(lat) as B)
    limit 1
));



with

sat_positions as ( select
                  time,
                  UPPER(substr(sat,2,1)) as sat,
                  lon, lat, altitude
                  from
       `project-id.satellite_positions_v20190208.sentinel_1_positions*`
                  where
                  _table_suffix between "{start_YYYYMMDD}" and
                                         "{end_YYYYMMDD}"
                                         ),

scenes as (select *,
             st_geogfromtext(footprint_wkt) AS footprint
              from  (select  distinct SUBSTR(scene_id, 3, 1) sat, scene_id,
                      footprint_wkt,
                      TIMESTAMP_ADD(start_time, INTERVAL
                        cast(timestamp_diff(end_time, start_time,
                        SECOND)/2 as int64) SECOND) scene_timestamp from 
`project-id.proj_sentinel1_v20210924.detect_foot_raw_{YYYYMMDD}`
)
),

extrapolate_table_raw as (
 select road_id, lat, lon, course as likely_course, st_geogpoint(lon,lat) pos, GP_RTP,
 case when GP_RTP = 1 then 85*0.868976 -- 85 miles per hour times conversion. toknots
 when GP_RTP = 2 then 50*0.868976 -- let's say a primary road they go 50 miles per hour
 end likely_speed
  from scratch_david.highway_primary_road_pairs

),

extrapolate_table as (
select road_id, lat, lon, pos, likely_course, scene_id, GP_RTP, likely_speed
from extrapolate_table_raw
cross join
scenes
where st_distance(footprint, pos) < 3500


),


-- the position of the satellite 30 seconds before the scene timestamp
start_pos AS (
    SELECT
         scene_id,
         lat AS start_lat,
         lon AS start_lon,
         altitude AS start_altitude,
         TIMESTAMP_SUB(scene_timestamp, INTERVAL 30 second) AS start_time
    FROM sat_positions a
    JOIN scenes b
          ON
        TIMESTAMP_SUB(TIMESTAMP_TRUNC(scene_timestamp, second), INTERVAL 30 second) = time
        AND a.sat = b.sat -- two satellites, make sure it is the right one
),

-- end position of the satellite 30 seconds after the scene timestamp
end_pos AS (
    SELECT
         scene_id,
         lat AS end_lat,
         lon AS end_lon,
         altitude AS end_altitude,
         TIMESTAMP_ADD(scene_timestamp, INTERVAL 30 second) AS end_time
    FROM sat_positions a
    JOIN scenes b
          ON
        TIMESTAMP_ADD(TIMESTAMP_TRUNC(scene_timestamp, second), INTERVAL 30 second) = time
        AND a.sat = b.sat
),


-- calcuate the location, and altitude of the satellite
sat_directions AS (
    SELECT
         scene_id,
         start_lat AS sat_start_lat,
         start_lon AS sat_start_lon,
         start_altitude,
         end_altitude,
         end_lat AS sat_end_lat,
         end_lon AS sat_end_lon,
         -- sat course, measured clockwise from due north, in degrees
         get_course(st_geogpoint(start_lon, start_lat), st_geogpoint(end_lon, end_lat)) sat_course,
         TIMESTAMP_DIFF(end_time, start_time, second) AS seconds,
--     end_lat / 2 + start_lat / 2 AS avg_lat
    FROM end_pos
    JOIN
         start_pos
         USING (scene_id)),


-- Calculate speed of satellite for each scene
-- speed of satellite varies a small ammount, so don't really need to calculate
-- for each scene. But hey -- why not calculate it directly?
sat_directions_with_speed AS (
    SELECT
         *,
         -- distance between start and end of satellite positions in meters
         ST_DISTANCE(ST_GEOGPOINT(sat_start_lon, sat_start_lat), ST_GEOGPOINT(sat_end_lon, sat_end_lat))
         -- multiply by a factor to account for satellite altitude
         * (EARTH_RADIUS_KM(sat_end_lat) + start_altitude / 1000) / EARTH_RADIUS_KM(sat_end_lat)
         / seconds  -- make it meters per second
         -- 1 nautical mile / 1852 meters * 3600 seconds/ hour
         * 1 / METERS_PER_NAUTICAL_MILE() * 3600
         AS sat_knots
         -- how often do you get to measure satellite speed in knots?
    FROM sat_directions),


-- Get distance from the likely position of the vessel to the satellite,
-- and the speed of the vessel perpendicular to the satellite.
extrapolated_positions_compared_to_satellite AS (
    SELECT
        *,
        likely_speed * SIN(
            RADIANS( likely_course - sat_course)
        ) AS vessel_speed_perpendicular_to_sat,
        -- likely location of vessel
        ST_DISTANCE(pos,
            -- line of satellite
            ST_MAKELINE(
                safe.ST_GEOGPOINT(sat_start_lon, sat_start_lat),
                (safe.ST_GEOGPOINT(sat_end_lon, sat_end_lat) )
            )
        )
        -- convert from meters to nautical miles, because
        / meters_per_nautical_mile()
        AS vessel_distance_to_sat_nm
    FROM
        extrapolate_table
    JOIN
        sat_directions_with_speed
        USING (scene_id)
),


-- using satellite speed, vessel speed perpendicular to satellite direction of travel,
-- and the distance of the vessel to the satellite, calculate the distance the vessel
-- will be offset in the direction of the satellite is traveling.
extrapolated_positions_adjust_formula AS (
    SELECT
        *,
        -- multiply by sin(look angle) to get the speed perpendicular to the satellite
        vessel_speed_perpendicular_to_sat * SIN(
            ATAN2(
                vessel_distance_to_sat_nm,
                start_altitude / METERS_PER_NAUTICAL_MILE()
            )
        )
        / sat_knots
        -- calculate the hypotenuse -- the distance from the vessel to the satellite
        -- using pathangerean theorm to get the approximate distance to the satellite in
        -- nautical miles.
        * POW(
            (
                POW(vessel_distance_to_sat_nm, 2)
                + POW(start_altitude/meters_per_nautical_mile(), 2)
            ),
            .5
        )
        AS adjusted_nautical_miles_parallel_to_sat,
        # The look angle is the angle from vertical to the object, from the satellite
        DEGREES(
            ATAN2(
                vessel_distance_to_sat_nm,
                start_altitude / METERS_PER_NAUTICAL_MILE()
            )
        ) AS look_angle
    FROM
        extrapolated_positions_compared_to_satellite
),


positions_adjusted AS (
    SELECT
        *,
        -- 60 nautical miles per degree
        - adjusted_nautical_miles_parallel_to_sat * COS(
            RADIANS(sat_course)
        ) / 60 AS lat_doppler_offset,
        -- 60 nautical miles * cos(lat) per degree
        - adjusted_nautical_miles_parallel_to_sat * SIN(
            RADIANS(sat_course)
        ) / (60 * COS(RADIANS(lat))) AS lon_doppler_offset
    FROM
        extrapolated_positions_adjust_formula
),

all_dopper_points as (
select scene_id, road_id, 
st_geogpoint( lon + lon_doppler_offset, lat + lat_doppler_offset) pos,  
st_geogpoint( lon + lon_doppler_offset*2, lat + lat_doppler_offset*2) pos2
from  positions_adjusted 
union all
select scene_id, road_id, 
st_geogpoint( lon - lon_doppler_offset, lat - lat_doppler_offset) pos,  
st_geogpoint( lon - lon_doppler_offset*2, lat - lat_doppler_offset*2) pos2  
from  positions_adjusted 
), 

by_road_and_scene as 
(select 
scene_id,
road_id,
st_convexhull(st_union_agg(pos)) doppler_area,
st_convexhull(st_union_agg(pos2)) doppler_area2
from 
all_dopper_points 
group by scene_id, road_id ),

by_scene as 
(
select scene_id, 
st_union_agg(doppler_area) doppler_area,
st_union_agg(doppler_area2) doppler_area2
from 
by_road_and_scene
group by scene_id
)

-- select st_astext(doppler_area) wkt from by_scene

select 
scene_id,
st_intersection(doppler_area,footprint) doppler_area,
st_intersection(doppler_area2,footprint) doppler_area2,
from 
by_scene
join
scenes
using(scene_id)
where st_intersects(doppler_area2,footprint)

'''



def process_day(d):
    """d is a datetime object"""
    table_id = f"project-id.scratch_david.doppler_removed_{d:%Y%m%d}"
    query_to_table(
        q_road_doppler_template.format(
            YYYYMMDD=f"{d:%Y%m%d}",
            start_YYYYMMDD=f"{d-timedelta(days=1):%Y%m%d}",
            end_YYYYMMDD=f"{d+timedelta(days=1):%Y%m%d}",
        ),
        table_id,
    )


# +
the_dates = np.arange(
    datetime(2018, 1, 1),
    datetime(2021, 12,31) + timedelta(days=1),
    timedelta(days=1),
).astype(datetime)


the_dates


# +
# process_day(the_dates[0])
# -

with ThreadPoolExecutor(max_workers=16) as e:
    for d in the_dates:
        e.submit(process_day, d)

import subprocess
command ="bq mk --time_partitioning_type=DAY proj_global_sar.doppler_road_area"
subprocess.call(command.split())

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
sys.path.append('../utils')
from bigquery_helper_functions import query_to_table
import numpy as np


def copydoppler(the_date):
    q = f'''select * from scratch_david.doppler_removed_{the_date:%Y%m%d}'''
    
    query_to_table(q, f"project-id.proj_global_sar.doppler_road_area${the_date:%Y%m%d}")



# +
the_dates = np.arange(
    datetime(2017, 1, 2),
    datetime(2021, 12, 31) + timedelta(days=1),
    timedelta(days=1),
).astype(datetime)


with ThreadPoolExecutor(max_workers=2) as e:
    for d in the_dates:
        e.submit(copydoppler, d)


# -

# # make table of detections within the SAR regions

def make_doppler_detection_table(the_date):
    q = f'''
     select 
       detect_lat, detect_lon, detect_id, scene_id, 
       ifnull(st_contains(doppler_area, st_geogpoint(detect_lon, detect_lat)), False) as in_road_doppler,
       ifnull(st_contains(doppler_area2, st_geogpoint(detect_lon, detect_lat)), False) as in_road_doppler2,
     from 
       `project-id.proj_sentinel1_v20210924.detect_scene_raw_{the_date:%Y%m%d}`
     left join 
       proj_global_sar.doppler_road_area 
     using(scene_id)
     where date(_partitiontime) = "{the_date:%Y-%m-%d}" ''' 
    
    query_to_table(q, f"project-id.proj_global_sar.doppler_road_area_detections${the_date:%Y%m%d}")



command ="bq mk --time_partitioning_type=DAY proj_global_sar.doppler_road_area_detections"
subprocess.call(command.split())

# +
the_dates = np.arange(
    datetime(2017, 2, 1),
    datetime(2021, 12, 31) + timedelta(days=1),
    timedelta(days=1),
).astype(datetime)


with ThreadPoolExecutor(max_workers=16) as e:
    for d in the_dates:
        e.submit(make_doppler_detection_table, d)
# -


