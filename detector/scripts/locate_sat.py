# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Calculate Sentinel-1 Positions

from dataclasses_json import DataClassJsonMixin
from datetime import date, timedelta, datetime
from pathlib import Path
import sys

import ephem
import pandas as pd

import spacetrack.operators as op
from spacetrack import SpaceTrackClient

import git

CREDENTIALS = "../untracked/spacetrack.json"


def get_repo_basedir() -> Path:
    repo = git.Repo('.', search_parent_directories=True)
    return Path(repo.working_tree_dir)


@dataclass
class Credentials(DataClassJsonMixin):
    username: str
    password: str
    url: Optional[str] = None


with (CREDENTIALS).open() as fd:
    creds = Credentials.from_json(fd.read())

st = SpaceTrackClient(
    identity=creds.username, password=creds.password
)

the_date = sys.argv[1]  # "2017-09-03"


# +
date_1 = datetime.strptime(the_date, "%Y-%m-%d").date()
date_2 = (datetime.strptime(the_date, "%Y-%m-%d") + timedelta(5)).date()

decay_epoch = op.inclusive_range(date_1, date_2)
norad1a = 39634
norad1b = 41456
# -


def get_orb(norad_id, name):
    tles = st.tle(
        norad_cat_id=norad_id,
        orderby="epoch desc",
        limit=3,
        format="tle",
        epoch=decay_epoch,
    )
    #     print(tles)
    tle = tles.split("\n")
    line1, line2 = tle[0], tle[1]
    orb = ephem.readtle(name, line1, line2)
    return orb


# +
s1b_in_orbit = True

orb_1a = get_orb(norad1a, "1a")
try:
    orb_1b = get_orb(norad1b, "1b")
except:
    s1b_in_orbit = False
# -


# +
thedatetimes = [
    datetime.strptime(the_date, "%Y-%m-%d") + timedelta(0, i)
    for i in range(24 * 60 * 60)
]
lats = []
lons = []
sat = []
times = []
altitudes = []

for t in thedatetimes:
    # calculate for 1a
    orb_1a.compute(t.strftime("%Y/%m/%d %H:%M:%S"))
    lon = ephem.degrees(orb_1a.sublong) * 180 / 3.1416
    lat = ephem.degrees(orb_1a.sublat) * 180 / 3.1416
    altitude = orb_1a.elevation
    times.append(t)
    lats.append(lat)
    lons.append(lon)
    altitudes.append(altitude)
    sat.append("1a")
    # now for 1b
    if s1b_in_orbit:
        orb_1b.compute(t.strftime("%Y/%m/%d %H:%M:%S"))
        lon = ephem.degrees(orb_1b.sublong) * 180 / 3.1416
        lat = ephem.degrees(orb_1b.sublat) * 180 / 3.1416
        altitude = orb_1b.elevation
        times.append(t)
        lats.append(lat)
        lons.append(lon)
        altitudes.append(altitude)
        sat.append("1b")


# -


df = pd.DataFrame(
    list(zip(times, sat, lons, lats, altitudes)),
    columns=["time", "sat", "lon", "lat", "altitude"],
)

df.head()

df.to_gbq(
    "satellite_positions_v20190208.sentinel_1_positions{}".format(
        the_date.replace("-", "")
    ),
    project_id="project-id",
    if_exists="replace",
)


# +
# a query to use later for figuring out distance to vessel

q = """with sat_positions as (select time, sat , lon, lat from scratch_david.sentinel_1_positions),

start_pos as (
select id, start_time, 
lat as start_lat,
lon as start_lon,
sat as sat_start,
UPPER(substr(sat,2,1)),
SUBSTR(id, 3, 1)
from `gfw-research.sentinel_ds3_fmean250_e10_d70_s20_8xmean_ns_v20190306.exts20190401`
join sat_positions
on start_time = time
and SUBSTR(id, 3, 1) = UPPER(substr(sat,2,1))),



end_pos as (

select id, end_time, 
lat as end_lat,
lon as end_lon,
sat as sat_end
from `gfw-research.sentinel_ds3_fmean250_e10_d70_s20_8xmean_ns_v20190306.exts20190401`
join sat_positions
on end_time = time
and SUBSTR(id, 3, 1) = UPPER(substr(sat,2,1))),


deltas as (

select 
(end_lat - start_lat) * 111 as N_km,
(end_lon - start_lon) * 111 * cos( (end_lat/2 +start_lat/2)*3.1416/180 ) as E_km,
end_lat/2 +start_lat/2 as avg_lat,
start_time,
end_time
from end_pos
join
start_pos 
using(id))

-- select ST_DISTANCE(ST_GEOGPOINT(0, 0), ST_MAKELINE( ST_GEOGPOINT(1, 0), (ST_GEOGPOINT(0, 1) ) ) )

select 
floor(ATAN2(E_Km,N_km)*180/3.1416) course,
count(*) number
from deltas
group by course
order by course
"""
