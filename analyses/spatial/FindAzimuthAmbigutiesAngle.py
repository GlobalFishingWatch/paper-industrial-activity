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

import proplot as pplt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
sys.path.append('../utils') 
from bigquery_helper_functions import query_to_table, update_table_description
from proj_id import project_id


# # Find Azimuth Ambiguties
#
# This notebook finds azimuth ambiguities by counting pairs of detections, separated by an angle relative to the satellite ("ambiguity angle" in this notebook, or amb_angle), which are along a line parallel to the direction of the satellite travel. The ambiguities show up at constant "ambiguity angles" in each of the three subswaths of Sentinel-1. The locations of these subswaths can be found by using the off-nadir angle (and we find that the off nadir angle is a very good proxy for the boundaries of these subswaths). Note that throughout this notebook I use "look angle" when really I mean "off nadir angle."
#

# +
q= '''create temp function radians(x float64) as (
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


sar_detects as 

(
select *, st_geogpoint(detect_lon, detect_lat) as loc,
earth_radius_km(detect_lat)/1.852 as earth_radius_nm


from 
  `proj_sentinel1_v20210924.detect_scene_match`
  join
  `proj_sentinel1_v20210924.detect_scene_pred_*`
  using(detect_id)
  where date(_partitiontime) between "2017-01-01" and "2021-12-31"
  and _table_suffix between '20161231' and '20220101'

),

matched_sar as 

(
  select scene_id,
  length_m,
  SUBSTR(scene_id, 3, 1) sat,
  detect_id,
  detect_lon,
  detect_lat,
  detect_timestamp as scene_timestamp,
  earth_radius_nm
  from
sar_detects
  where
  -- where length_m < 100
  -- and 
  score > 1e-3




),


sat_positions as ( select
                    time,
                    UPPER(substr(sat,2,1)) as sat,
                    lon, lat, altitude
                from
                    `satellite_positions_v20190208.sentinel_1_positions*`
                where
                    _table_suffix between '20161231' and '20230101'),

scenes as (select distinct sat, scene_id, scene_timestamp from matched_sar),


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
        TIMESTAMP_SUB(TIMESTAMP_TRUNC(scene_timestamp, second), INTERVAL 30 second) = TIMESTAMP_TRUNC(time, second)
        AND lower(a.sat) = lower(b.sat) -- two satellites, make sure it is the right one
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
        TIMESTAMP_ADD(TIMESTAMP_TRUNC(scene_timestamp, second), INTERVAL 30 second) = TIMESTAMP_TRUNC(time, second)
        AND lower(a.sat) = lower(b.sat)
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
detections_compared_to_satellite AS (
    SELECT
        *,
        -- likely_speed * SIN(
        --     RADIANS( likely_course - sat_course)
        -- ) AS vessel_speed_perpendicular_to_sat,
        -- likely location of vessel
        ST_DISTANCE(safe.ST_GEOGPOINT(detect_lon, detect_lat),
            -- line of satellite
            ST_MAKELINE(
                safe.ST_GEOGPOINT(sat_start_lon, sat_start_lat),
                (safe.ST_GEOGPOINT(sat_end_lon, sat_end_lat) )
            )
        )
        -- convert from meters to nautical miles, because
        / meters_per_nautical_mile()
        AS vessel_distance_to_sat_ground_nm,

        sqrt( pow(ST_DISTANCE(safe.ST_GEOGPOINT(detect_lon, detect_lat),
                            -- line of satellite
                            ST_MAKELINE(
                                safe.ST_GEOGPOINT(sat_start_lon, sat_start_lat),
                                (safe.ST_GEOGPOINT(sat_end_lon, sat_end_lat) )
                            ), True
        ),2 ) + pow(start_altitude,2) ) 
        -- convert from meters to nautical miles, because
        / meters_per_nautical_mile()
        AS vessel_distance_to_sat_nm


    FROM
        matched_sar
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
        # The look angle is the angle from vertical to the object, from the satellite
        DEGREES(
            ATAN2(
                vessel_distance_to_sat_ground_nm,
                start_altitude / METERS_PER_NAUTICAL_MILE()
            )
            
        ) AS look_angle_old,
-- d is the distance along the surface you compute already and R is the radius of the earth (6.371e6 m) and h is the satellite height. Then I think the angle is
-- atan2(sin(d / R) * R, R + h - cos(d / R) * R)

        DEGREES(atan2(sin(vessel_distance_to_sat_ground_nm / earth_radius_nm ) * earth_radius_nm,
         earth_radius_nm + start_altitude/ METERS_PER_NAUTICAL_MILE() - 
        cos(vessel_distance_to_sat_ground_nm / earth_radius_nm) * earth_radius_nm))
        as look_angle

    FROM
        detections_compared_to_satellite
),


positions_adjusted AS (
    SELECT
        *,
        -- 60 nautical miles per degree
        - 5 * COS( -- 5 nautical miles
            RADIANS(sat_course)
        ) / 60 AS lat_adjust,
        -- 60 nautical miles * cos(lat) per degree
        - 5 * SIN( -- 5 nautical miles
            RADIANS(sat_course)
        ) / (60 * COS(RADIANS(detect_lon))) AS lon_adjust
    FROM
        extrapolated_positions_adjust_formula
),


with_lines as 

(select 
scene_id,
length_m,
look_angle,
vessel_distance_to_sat_nm,

st_geogpoint(detect_lon, detect_lat) loc,
            ST_MAKELINE(
                safe.ST_GEOGPOINT(detect_lon + lon_adjust, detect_lat + lat_adjust),
                safe.ST_GEOGPOINT(detect_lon - lon_adjust, detect_lat - lat_adjust) 
            ) as line
  
from positions_adjusted
)



select 
-- look_angle,
floor(amb_angle*1000)/1000 as amb_angle,
floor(look_angle*10)/10 as look_angle,
avg(ambiguity_length) as ambiguity_length,
avg(amb_dist) avg_amb_dist,
avg(distance_to_sat_km) distance_to_sat_km,
count(*) detections
from

(select 
b.length_m as ambiguity_length,
look_angle,
st_distance(a.loc,b.loc )/1000 as amb_dist,
degrees(atan2( st_distance(a.loc,b.loc )/1000,vessel_distance_to_sat_nm*1.852)) as amb_angle,
vessel_distance_to_sat_nm*1.852 as distance_to_sat_km
from with_lines a
join
sar_detects b
using(scene_id)
where b.score < 1e-3
and st_distance(line, b.loc)<400
and st_distance(a.loc,b.loc ) between 3000 and 15000
and a.length_m > 100
-- and b.length_m > 100
)
where amb_angle between .2 and .7 

group by 
-- distance_to_sat_km, 
amb_angle, look_angle
order by amb_angle'''

df = pd.read_gbq(q)
# -

distances = df.distance_to_sat_km.unique()
distances.sort()
distances



angles = df.look_angle.unique()
angles.sort()

angles

# get the ambiguity angle at which the maximium number of detections was found
# at different off-nadir angles
amb_angles = []
for a in angles:
    d = df[df.look_angle == a]
    d = d.sort_values("detections",ascending=False)
    amb_angles.append(d.amb_angle.values[0])


# +
# amb_angles

# +

plt.figure(figsize=(5,5))
plt.title("angle to ambiguity")
plt.xlabel("off nadir-angle")
plt.ylabel("angle to ambiguity")

plt.plot(angles,amb_angles)
# Doted lines show where ESA says that the bouandaries between
# the sub-swaths is
plt.plot([30.91,30.91],[0,100],'--', color = 'black')
plt.plot([32.43,32.43],[0,100],'--', color = 'black')
plt.plot([36.92,36.92],[0,100],'--', color = 'black')
plt.plot([35.39,35.39],[0,100],'--', color = 'black')

plt.ylim(.29, .38)
# -

angles[55:67]

df.head()



# # How sharp is the cutoff in the 

# +
fig, axs = pplt.subplots(ncols=3, nrows=4, span=True, figsize=(6,7))

for i, angle in enumerate(angles[55:67]):
    if i >= 12:
        break    
    d = df.copy()
    d = d[d.look_angle==angle]
    d['detections_norm'] = d.detections/d.detections.sum()
    d = d.dropna()

    ax = axs[i]
    ax.scatter(d.amb_angle.values, d.detections_norm.values) 
    ax.set_ylim(0,.15)
    ax.set_title(f"{angle} degrees")

    
    


# +
fig, axs = pplt.subplots(ncols=3, nrows=4, span=True, figsize=(6,7))

for i, angle in enumerate(angles[x:x+12]):
    if i >= 12:
        break    
    d = df.copy()
    d = d[d.look_angle==angle]
    d['detections_norm'] = d.detections/d.detections.sum()
    d = d.dropna()

    ax = axs[i]
    ax.scatter(d.amb_angle.values, d.detections_norm.values) 
    ax.set_ylim(0,.15)
    ax.set_title(f"{angle} degrees")

# +
angle1 = 32.41
angle2 = 36.85

plt.figure(figsize=(10,4))

d = df[df.look_angle < angle1]
d = d.groupby('amb_angle').sum()
d['detections_norm'] = d.detections/d.detections.sum()
plt.plot(d.index.values,(d.detections_norm).values, label = f"<{angle1}")

d = d[(d.detections>2*d.detections.mean())&(d.detections-d.detections.mean()>d.detections.std())]
plt.scatter(d.index.values, d.detections_norm.values)
print(f"<{angle1}, {d.index.min()}-{d.index.max()}")


d = df[(df.look_angle >= angle1)&(df.look_angle < angle2)]

d = d.groupby('amb_angle').sum()
d['detections_norm'] = d.detections/d.detections.sum()

plt.plot(d.index.values,(d.detections/d.detections.sum()).values, label = f"{angle1}-{angle2}")

d = d[(d.detections>2*d.detections.mean())&(d.detections-d.detections.mean()>d.detections.std())]
plt.scatter(d.index.values, d.detections_norm.values)
print(f"{angle1}-{angle2}, {d.index.min()}-{d.index.max()}")



d = df[(df.look_angle >= angle2)]

d = d.groupby('amb_angle').sum()
d['detections_norm'] = d.detections/d.detections.sum()

plt.plot(d.index.values,(d.detections/d.detections.sum()).values, label = f">= {angle2}")

d = d[(d.detections>2*d.detections.mean())&(d.detections-d.detections.mean()>d.detections.std())]
plt.scatter(d.index.values, d.detections_norm.values)
print(f">{angle2}, {d.index.min()}-{d.index.max()}")

plt.xlim(.29, .38)

plt.legend(title="Off nadir angle,\ndegrees", frameon=False)
plt.ylabel("relative number of detections")
plt.xlabel("angle to ambiguity")
plt.title("number of detections at different off nadir angles")
# -

# # What source lengths matter?

# +
q = '''create temp function radians(x float64) as (
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


sar_detects as 

(
select *, st_geogpoint(detect_lon, detect_lat) as loc,
earth_radius_km(detect_lat)/1.852 as earth_radius_nm


from 
  `proj_sentinel1_v20210924.detect_scene_match`
  join
  `proj_sentinel1_v20210924.detect_scene_pred_*`
  using(detect_id)
  where date(_partitiontime) between "2017-01-01" and "2021-12-31"
  and _table_suffix between '20161231' and '20220101'

),

matched_sar as 

(
  select scene_id,
  length_m,
  SUBSTR(scene_id, 3, 1) sat,
  detect_id,
  detect_lon,
  detect_lat,
  detect_timestamp as scene_timestamp,
  earth_radius_nm
  from
sar_detects
  where
  -- where length_m < 100
  -- and 
  score > 1e-3




),


sat_positions as ( select
                    time,
                    UPPER(substr(sat,2,1)) as sat,
                    lon, lat, altitude
                from
                    `satellite_positions_v20190208.sentinel_1_positions*`
                where
                    _table_suffix between '20161231' and '20230101'),

scenes as (select distinct sat, scene_id, scene_timestamp from matched_sar),


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
        TIMESTAMP_SUB(TIMESTAMP_TRUNC(scene_timestamp, second), INTERVAL 30 second) = TIMESTAMP_TRUNC(time, second)
        AND lower(a.sat) = lower(b.sat) -- two satellites, make sure it is the right one
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
        TIMESTAMP_ADD(TIMESTAMP_TRUNC(scene_timestamp, second), INTERVAL 30 second) = TIMESTAMP_TRUNC(time, second)
        AND lower(a.sat) = lower(b.sat)
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
detections_compared_to_satellite AS (
    SELECT
        *,
        -- likely_speed * SIN(
        --     RADIANS( likely_course - sat_course)
        -- ) AS vessel_speed_perpendicular_to_sat,
        -- likely location of vessel
        ST_DISTANCE(safe.ST_GEOGPOINT(detect_lon, detect_lat),
            -- line of satellite
            ST_MAKELINE(
                safe.ST_GEOGPOINT(sat_start_lon, sat_start_lat),
                (safe.ST_GEOGPOINT(sat_end_lon, sat_end_lat) )
            )
        )
        -- convert from meters to nautical miles, because
        / meters_per_nautical_mile()
        AS vessel_distance_to_sat_ground_nm,

        sqrt( pow(ST_DISTANCE(safe.ST_GEOGPOINT(detect_lon, detect_lat),
                            -- line of satellite
                            ST_MAKELINE(
                                safe.ST_GEOGPOINT(sat_start_lon, sat_start_lat),
                                (safe.ST_GEOGPOINT(sat_end_lon, sat_end_lat) )
                            ), True
        ),2 ) + pow(start_altitude,2) ) 
        -- convert from meters to nautical miles, because
        / meters_per_nautical_mile()
        AS vessel_distance_to_sat_nm


    FROM
        matched_sar
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
        # The look angle is the angle from vertical to the object, from the satellite
        DEGREES(
            ATAN2(
                vessel_distance_to_sat_ground_nm,
                start_altitude / METERS_PER_NAUTICAL_MILE()
            )
            
        ) AS look_angle_old,
-- d is the distance along the surface you compute already and R is the radius of the earth (6.371e6 m) and h is the satellite height. Then I think the angle is
-- atan2(sin(d / R) * R, R + h - cos(d / R) * R)

        DEGREES(atan2(sin(vessel_distance_to_sat_ground_nm / earth_radius_nm ) * earth_radius_nm,
         earth_radius_nm + start_altitude/ METERS_PER_NAUTICAL_MILE() - 
        cos(vessel_distance_to_sat_ground_nm / earth_radius_nm) * earth_radius_nm))
        as look_angle

    FROM
        detections_compared_to_satellite
),


positions_adjusted AS (
    SELECT
        *,
        -- 60 nautical miles per degree
        - 5 * COS( -- 5 nautical miles
            RADIANS(sat_course)
        ) / 60 AS lat_adjust,
        -- 60 nautical miles * cos(lat) per degree
        - 5 * SIN( -- 5 nautical miles
            RADIANS(sat_course)
        ) / (60 * COS(RADIANS(detect_lon))) AS lon_adjust
    FROM
        extrapolated_positions_adjust_formula
),


with_lines as 

(select 
scene_id,
length_m,
look_angle,
vessel_distance_to_sat_nm,

st_geogpoint(detect_lon, detect_lat) loc,
            ST_MAKELINE(
                safe.ST_GEOGPOINT(detect_lon + lon_adjust, detect_lat + lat_adjust),
                safe.ST_GEOGPOINT(detect_lon - lon_adjust, detect_lat - lat_adjust) 
            ) as line
  
from positions_adjusted
)



select 
-- look_angle,
floor(amb_angle*1000)/1000 as amb_angle,
-- floor(look_angle*10)/10 as look_angle,

case when (look_angle < 32.41) then 'I'
when (look_angle between 32.41 and 36.85) then "II"
when (look_angle > 36.85) then "III"
end look_angle,

floor(length_m/10)*10 source_length,
avg(ambiguity_length) as ambiguity_length,
avg(amb_dist) avg_amb_dist,
avg(distance_to_sat_km) distance_to_sat_km,
count(*) detections
from

(select 
b.length_m as ambiguity_length,
look_angle,
a.length_m,
st_distance(a.loc,b.loc )/1000 as amb_dist,
degrees(atan2( st_distance(a.loc,b.loc )/1000,vessel_distance_to_sat_nm*1.852)) as amb_angle,
vessel_distance_to_sat_nm*1.852 as distance_to_sat_km
from with_lines a
join
sar_detects b
using(scene_id)
where b.score < 1e-3
and st_distance(line, b.loc)<400
and a.length_m > b.length_m
and st_distance(a.loc,b.loc ) between 3000 and 15000
-- and a.length_m > 100
-- and b.length_m > 100
)
where amb_angle between .2 and .7 

group by 
-- distance_to_sat_km, 
amb_angle, look_angle,source_length
order by amb_angle'''

dfs = pd.read_gbq(q)
# -

dfs.look_angle.unique()

# +
lengths = []
detections = []


fig, axs = pplt.subplots(ncols=5, nrows=8, span=True, figsize=(5,20))

for i, length in enumerate(np.arange(0,400,10)):
    
    ax = axs[i]    
    de = 0 
    for look_angle in ("I","II","III"):
        d = dfs[(dfs.look_angle == look_angle)&(dfs.source_length==length)]
        d = d.groupby('amb_angle').sum()
        d['detections_norm'] = d.detections/d.detections.sum()
        ax.plot(d.index.values,(d.detections_norm).values, label = f"{look_angle}")
        d = d[(d.detections>2*d.detections.mean())&(d.detections-d.detections.mean()>d.detections.std())]
        ax.scatter(d.index.values, d.detections_norm.values)
        de += d.detections.sum()
    lengths.append(length)
    detections.append(de)
        
#     plt.legend()
#     ax.set_ylim(0,.3)
#     ax.set_xlim(.29, .38)
    ax.set_title(length)
    
plt.show()
# -

plt.figure(figsize=(6,4))
plt.plot(lengths, detections)
plt.ylim(0,13000)
plt.title("number of potential ambiguties as function of sourcer length")
plt.xlabel("length of source, m")

plt.figure(figsize=(6,4))
plt.plot(lengths, np.array(detections).cumsum()/np.array(detections).sum())
plt.title("fraction of ambiguties from source above a given length")
plt.xlabel("length of source, m")
plt.ylabel("fraction of ambiguties from sources\nsmaller than a given size")

# # Findings
#  - more than 90% of potential ambiguities are from vessels > 100 m
#  - the false positive rate will be higher for these smaller boats as the peaks are lower



# # Find Ambiguities

# +
q = f'''create temp function radians(x float64) as (
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


sar_detects as 

(
select *, st_geogpoint(detect_lon, detect_lat) as loc,
earth_radius_km(detect_lat)/1.852 as earth_radius_nm
 from 
  `proj_sentinel1_v20210924.detect_scene_match`
  join
  `proj_sentinel1_v20210924.detect_scene_pred_*`
  using(detect_id)
  where date(_partitiontime) between "2017-01-01" and "2021-12-31"
  and _table_suffix between '20161231' and '20220101'

),

matched_sar as 

(
  select scene_id,
  length_m,
  SUBSTR(scene_id, 3, 1) sat,
  detect_id,
  detect_lon,
  detect_lat,
  detect_timestamp as scene_timestamp,
  earth_radius_nm
  from
sar_detects
  -- where
  -- where length_m < 100
  -- and 
  -- score > 1e-3




),


sat_positions as ( select
                    time,
                    UPPER(substr(sat,2,1)) as sat,
                    lon, lat, altitude
                from
                    `satellite_positions_v20190208.sentinel_1_positions*`
                where
                    _table_suffix between '20161231' and '20230101'),

scenes as (select distinct sat, scene_id, scene_timestamp from matched_sar),


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
        TIMESTAMP_SUB(TIMESTAMP_TRUNC(scene_timestamp, second), INTERVAL 30 second) = TIMESTAMP_TRUNC(time, second)
        AND lower(a.sat) = lower(b.sat) -- two satellites, make sure it is the right one
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
        TIMESTAMP_ADD(TIMESTAMP_TRUNC(scene_timestamp, second), INTERVAL 30 second) 
          = TIMESTAMP_TRUNC(time, second)
        AND lower(a.sat) = lower(b.sat)
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
detections_compared_to_satellite AS (
    SELECT
        *,
        -- likely_speed * SIN(
        --     RADIANS( likely_course - sat_course)
        -- ) AS vessel_speed_perpendicular_to_sat,
        -- likely location of vessel



        ST_DISTANCE(safe.ST_GEOGPOINT(detect_lon, detect_lat),
            -- line of satellite
            ST_MAKELINE(
                safe.ST_GEOGPOINT(sat_start_lon, sat_start_lat),
                (safe.ST_GEOGPOINT(sat_end_lon, sat_end_lat) )
            )
        )
        -- convert from meters to nautical miles, because
        / meters_per_nautical_mile()
        AS vessel_distance_to_sat_ground_nm,

        sqrt( pow(ST_DISTANCE(safe.ST_GEOGPOINT(detect_lon, detect_lat),
                            -- line of satellite
                            ST_MAKELINE(
                                safe.ST_GEOGPOINT(sat_start_lon, sat_start_lat),
                                (safe.ST_GEOGPOINT(sat_end_lon, sat_end_lat) )
                            ), True
        ),2 ) + pow(start_altitude,2) ) 
        -- convert from meters to nautical miles, because
        / meters_per_nautical_mile()
        AS vessel_distance_to_sat_nm

    FROM
        matched_sar
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
        -- vessel_speed_perpendicular_to_sat * SIN(
        --     ATAN2(
        --         vessel_distance_to_sat_nm,
        --         start_altitude / METERS_PER_NAUTICAL_MILE()
        --     )
        -- )
        -- / sat_knots
        -- -- calculate the hypotenuse -- the distance from the vessel to the satellite
        -- -- using pathangerean theorm to get the approximate distance to the satellite in
        -- -- nautical miles.
        -- * POW(
        --     (
        --         POW(vessel_distance_to_sat_nm, 2)
        --         + POW(start_altitude/meters_per_nautical_mile(), 2)
        --     ),
        --     .5
        -- )
        -- AS adjusted_nautical_miles_parallel_to_sat,
        # The look angle is the angle from vertical to the object, from the satellite

-- d is the distance along the surface you compute already and R is the radius of the earth (6.371e6 m) and h is the satellite height. Then I think the angle is
-- atan2(sin(d / R) * R, R + h - cos(d / R) * R)

        DEGREES(atan2(sin(vessel_distance_to_sat_ground_nm / earth_radius_nm ) * earth_radius_nm,
         earth_radius_nm + start_altitude/ METERS_PER_NAUTICAL_MILE() - 
        cos(vessel_distance_to_sat_ground_nm / earth_radius_nm) * earth_radius_nm))
        as look_angle
    FROM
        detections_compared_to_satellite
),


positions_adjusted AS (
    SELECT
        *,
        -- 60 nautical miles per degree
        - 5 * COS( -- 5 nautical miles
            RADIANS(sat_course)
        ) / 60 AS lat_adjust,
        -- 60 nautical miles * cos(lat) per degree
        - 5 * SIN( -- 5 nautical miles
            RADIANS(sat_course)
        ) / (60 * COS(RADIANS(detect_lon))) AS lon_adjust
    FROM
        extrapolated_positions_adjust_formula
),


with_lines as 

(select 
scene_id,
length_m,
vessel_distance_to_sat_nm,
look_angle,
st_geogpoint(detect_lon, detect_lat) loc,
            ST_MAKELINE(
                safe.ST_GEOGPOINT(detect_lon + lon_adjust, detect_lat + lat_adjust),
                safe.ST_GEOGPOINT(detect_lon - lon_adjust, detect_lat - lat_adjust) 
            ) as line
  
from positions_adjusted
),

values_binned as 
(select 
  b.detect_id detect_id,
  b.length_m as ambiguity_length,
  a.length_m as source_length,
  look_angle,
  st_distance(line, b.loc) line_dist,
  floor(vessel_distance_to_sat_nm*1.852/5)*5  as sat_distances,
  floor(st_distance(a.loc,b.loc)/1000*5)/5 as amb_distances, 

  degrees(atan2( st_distance(a.loc,b.loc )/1000,vessel_distance_to_sat_nm*1.852)) as amb_angle

from 
  with_lines a
join
  sar_detects b
using(scene_id)
where 
  st_distance(line, b.loc)<1000
  and st_distance(a.loc,b.loc ) between 3000 and 8000
  and a.length_m > 100
  and b.length_m < a.length_m
  )


select * from values_binned

where 

### THESE ARE THE KEY VALUES

(look_angle < 32.41 and amb_angle between 0.359 and 0.367)
or
(look_angle between 32.41 and 36.85 and amb_angle between 0.304 and 0.312)
or 
(look_angle > 36.85 and amb_angle between 0.355 and 0.363)



'''

table_id = f'{project_id}.proj_global_sar.potential_S1_amgibuities_v2_2017_2021'
query_to_table(q, table_id)
# -

d.head()

# +
q = '''select floor(line_dist/10)*10 line_dist, 
count(*) detects from proj_global_sar.potential_S1_amgibuities_v2_2017_2021
where ambiguity_length < source_length
group by line_dist order by line_dist'''

d = pd.read_gbq(q)
# -

plt.figure(figsize=(6,6))
plt.scatter(d.line_dist, d.detects)
plt.xlabel("distance from azimuth line to potential ambiguity\n(range offset)")
plt.ylabel("number of potential ambiguities")




# ## Findings
#  - at 200m distance, the number of detecitons is ~2x the background rate, suggesting that that is where we should have the cutoff to maximize false positives versus false negatives. 
#  
# ## What fraction of detections in our data are ambiguties?

# +
# now write a query to find out how many of our detections are ambiguties in the dataset
# get the final query, copy it to the clipboard, and then edit it
import pyperclip
import sys
sys.path.append('../utils') 
from vessel_queries import *
from proj_id import project_id

q = f'''
{final_query_static} -- from the file vessel_queries


select
  detect_lat as lat,
  detect_lon as lon,
  year,
  in_road_doppler,
  in_road_doppler2,
  matched_category,
  fishing_score,
  confidence,
  score,
  on_fishing_list,
  overpasses_2017_2021
from
  final_table'''
pyperclip.copy(q)

# this is copied to the clipboard and then edited in the console

# +
q ='''  



with
predictions_table as
(

  select 
    detect_id, 
    avg(fishing_33) fishing_score_low,
    avg( fishing_50) fishing_score, 
    avg(fishing_66) fishing_score_high
  from
    (select detect_id, fishing_33, fishing_50, fishing_66 from 
    `proj_sentinel1_v20210924.fishing_pred_even_v5*`
    union all
    select detect_id, fishing_33, fishing_50, fishing_66 from 
    `proj_sentinel1_v20210924.fishing_pred_odd_v5*`
    )
  group by 
    detect_id
    
),
vessel_info as (
select
  ssvid,
  if(on_fishing_list_known is not null, on_fishing_list_known, on_fishing_list_nn) as on_fishing_list
from
  `gfw_research.vi_ssvid_v20230701`
  -- don't do anything with identity spoofing vessels!
  where activity.overlap_hours_multinames < 24
),


detections_table as
(
  select
    detect_lat,
    detect_lon,
    detect_id,
    ssvid_mult_recall_length as ssvid,
    eez_iso3,
    score_mult_recall_length as score,  
    confidence_mult_recall_length as confidence,  
    7.4e-6 as dd_perkm2,
    overpasses_2017_2021,
    extract(year from detect_timestamp) year,
    date_24,
    in_road_doppler,
    in_road_doppler2,
    periods24_with_overpass,
    overpasses_24day,
    length_m,
    potential_ambiguity
    
  from
  `proj_global_sar.detections_w_overpasses_v20230922`
  where
  -- the following is very restrictive on repeated objects
  repeats_100m_180days_forward < 3 and
  repeats_100m_180days_back < 3 and
  repeats_100m_180days_center < 3
  -- get rid of scenes where more than half the detections
  -- are likely noise
  and (scene_detections <=5 or scene_quality > .5)
  and extract(date from detect_timestamp)
     between "2017-01-01" and "2021-12-31"
  -- at least 10 overpasses
  and overpasses_2017_2021 > 30
  -- our cutoff for noise -- this could be adjusted down, but makes
  -- very little difference between .5 and .7
  and presence > .7
  and not in_road_doppler
  and not close_to_infra
  and not
      (
       ( detect_lon > -120.0 and detect_lon < -46.8 and detect_lat> 50.5 and detect_lat < 80.5 ) or 
   ( detect_lon > -120.0 and detect_lon < -46.8 and detect_lat> 50.5 and detect_lat < 80.5 ) or 
   (
          (detect_lon > 39.5 or detect_lon < -46.8 ) and
          detect_lat> 65.0 and detect_lat < 90 )
      or   ( detect_lon > 15.95 and detect_lon < 36.23 and detect_lat> 59.02 and detect_lat < 66.57 ) or 
   ( detect_lon > -173.7 and detect_lon < -158.4 and detect_lat> 62.0 and detect_lat < 66.8 ) or 
   (
          (detect_lon > 130.5 or detect_lon < -174.2 ) and
          detect_lat> 50.6 and detect_lat < 67.8 )
      or   ( detect_lon > 3.5 and detect_lon < 31.9 and detect_lat> 78.1 and detect_lat < 85.0 ) or 
   ( detect_lon > -179.8 and detect_lon < -156.5 and detect_lat> 57.4 and detect_lat < 62.2 ) or 
   ( detect_lon > -44.82 and detect_lon < -29.05 and detect_lat> -57.93 and detect_lat < -50.61 ) or 
   ( detect_lon > 31.4 and detect_lon < 60.3 and detect_lat> 61.4 and detect_lat < 73.1 ) or 
   ( detect_lon > -27.61 and detect_lon < -19.47 and detect_lat> 68 and detect_lat < 68.62 ) )

  ) ,
  
final_table as (
select
  date_24,
  year,
  detect_lat,
  detect_lon,
  detect_id,
  overpasses_2017_2021,
  eez_iso3,
  fishing_score,
  fishing_score_low,
  fishing_score_high,
  case when score > dd_perkm2 and on_fishing_list then "matched_fishing"
   when score > dd_perkm2 and not on_fishing_list then "matched_nonfishing"
   when score > dd_perkm2 and on_fishing_list is null then "matched_unknown"
   when score < dd_perkm2 then "unmatched" end as matched_category,
  in_road_doppler,
  in_road_doppler2,
  confidence,
  score,
  on_fishing_list,
  ssvid, 
  length_m,
  potential_ambiguity
from
  detections_table a
left join
  vessel_info
using(ssvid)
left join
  predictions_table
using(detect_id)
) -- from the file vessel_queries


select
  countif(not potential_ambiguity) detections,
  countif(potential_ambiguity) ambiguities,
  countif(potential_ambiguity)/count(*)*100 percent_ambiguities

from
  final_table
  where not in_road_doppler

'''

pd.read_gbq(q)

# +
q ='''  



with
predictions_table as
(

  select 
    detect_id, 
    avg(fishing_33) fishing_score_low,
    avg( fishing_50) fishing_score, 
    avg(fishing_66) fishing_score_high
  from
    (select detect_id, fishing_33, fishing_50, fishing_66 from 
    `proj_sentinel1_v20210924.fishing_pred_even_v5*`
    union all
    select detect_id, fishing_33, fishing_50, fishing_66 from 
    `proj_sentinel1_v20210924.fishing_pred_odd_v5*`
    )
  group by 
    detect_id
    
),
vessel_info as (
select
  ssvid,
  if(on_fishing_list_known is not null, on_fishing_list_known, on_fishing_list_nn) as on_fishing_list
from
  `gfw_research.vi_ssvid_v20230701`
  -- don't do anything with identity spoofing vessels!
  where activity.overlap_hours_multinames < 24
),


detections_table as
(
  select
    detect_lat,
    detect_lon,
    detect_id,
    ssvid_mult_recall_length as ssvid,
    eez_iso3,
    score_mult_recall_length as score,  
    confidence_mult_recall_length as confidence,  
    7.4e-6 as dd_perkm2,
    overpasses_2017_2021,
    extract(year from detect_timestamp) year,
    date_24,
    in_road_doppler,
    in_road_doppler2,
    periods24_with_overpass,
    overpasses_24day,
    length_m,
    potential_ambiguity
    
  from
  `proj_global_sar.detections_w_overpasses_v20230803`
  where
  -- the following is very restrictive on repeated objects
  repeats_100m_180days_forward < 3 and
  repeats_100m_180days_back < 3 and
  repeats_100m_180days_center < 3
  -- get rid of scenes where more than half the detections
  -- are likely noise
  and (scene_detections <=5 or scene_quality > .5)
  and extract(date from detect_timestamp)
     between "2017-01-01" and "2021-12-31"
  -- at least 10 overpasses
  and overpasses_2017_2021 > 30
  -- our cutoff for noise -- this could be adjusted down, but makes
  -- very little difference between .5 and .7
  and presence > .7
  and not in_road_doppler
  and not close_to_infra
  and not
      (
       ( detect_lon > -120.0 and detect_lon < -46.8 and detect_lat> 50.5 and detect_lat < 80.5 ) or 
   ( detect_lon > -120.0 and detect_lon < -46.8 and detect_lat> 50.5 and detect_lat < 80.5 ) or 
   (
          (detect_lon > 39.5 or detect_lon < -46.8 ) and
          detect_lat> 65.0 and detect_lat < 90 )
      or   ( detect_lon > 15.95 and detect_lon < 36.23 and detect_lat> 59.02 and detect_lat < 66.57 ) or 
   ( detect_lon > -173.7 and detect_lon < -158.4 and detect_lat> 62.0 and detect_lat < 66.8 ) or 
   (
          (detect_lon > 130.5 or detect_lon < -174.2 ) and
          detect_lat> 50.6 and detect_lat < 67.8 )
      or   ( detect_lon > 3.5 and detect_lon < 31.9 and detect_lat> 78.1 and detect_lat < 85.0 ) or 
   ( detect_lon > -179.8 and detect_lon < -156.5 and detect_lat> 57.4 and detect_lat < 62.2 ) or 
   ( detect_lon > -44.82 and detect_lon < -29.05 and detect_lat> -57.93 and detect_lat < -50.61 ) or 
   ( detect_lon > 31.4 and detect_lon < 60.3 and detect_lat> 61.4 and detect_lat < 73.1 ) or 
   ( detect_lon > -27.61 and detect_lon < -19.47 and detect_lat> 68 and detect_lat < 68.62 ) )

  ) ,
  
final_table as (
select
  date_24,
  year,
  detect_lat,
  detect_lon,
  detect_id,
  overpasses_2017_2021,
  eez_iso3,
  fishing_score,
  fishing_score_low,
  fishing_score_high,
  case when score > dd_perkm2 and on_fishing_list then "matched_fishing"
   when score > dd_perkm2 and not on_fishing_list then "matched_nonfishing"
   when score > dd_perkm2 and on_fishing_list is null then "matched_unknown"
   when score < dd_perkm2 then "unmatched" end as matched_category,
  in_road_doppler,
  in_road_doppler2,
  confidence,
  score,
  on_fishing_list,
  ssvid, 
  length_m,
  potential_ambiguity
from
  detections_table a
left join
  vessel_info
using(ssvid)
left join
  predictions_table
using(detect_id)
) -- from the file vessel_queries


select
  countif(not potential_ambiguity) detections,
  countif(potential_ambiguity) ambiguities,
  countif(potential_ambiguity)/count(*)*100 percent_ambiguities

from
  final_table
  where not in_road_doppler

'''

pd.read_gbq(q)
# -


