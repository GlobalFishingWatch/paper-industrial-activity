CREATE OR REPLACE TABLE
`global-fishing-watch.paper_industrial_activity.vessels_meta_v20240605` (
  detect_id INTEGER OPTIONS (description = 'Unique identifier for each detection'),
  flag_iso STRING OPTIONS (description = 'Three letter ISO code for the flag state.  This is the country that issued the MMSI'),
  shipname STRING OPTIONS (description = 'Name of the vessel'),
  callsign STRING OPTIONS (description = 'Radio callsign'),
  imo STRING OPTIONS (description = 'IMO number'),
  elevation_m FLOAT64 OPTIONS (description = 'Vertical distance from mean sea level in meters'),
  distance_from_shore_m FLOAT64 OPTIONS (description = 'Distance from shore in meters'),
  MRGID_EEZ INT64 OPTIONS (description = 'Marine Regions identifier'),
  TERRITORY1 STRING OPTIONS (description = 'Marine Regions territory name'),
  ISO_TER1 STRING OPTIONS (description = 'Marine Regions ISO country identifier'),
  POL_TYPE STRING OPTIONS (description = 'Marine Regions political unit type'),
  `UNION` STRING OPTIONS (description = 'Marine Regions union')
) OPTIONS (
  description = """Extended metadata for vessels keyed by MMSI.

  Note that many vessels will change their identifiers over the 5-year study period, and the only identifier that
  is unchangeable is the IMO number.  This table contains the best identifiers associated with each MMSI over the
  course of a calendar year.  However these can change and it is possible for more than one vessel to use the same
  MMSI at the same time.

  Not all MMSI will have all the associated identity fields.  The AIS messages that contain the position information
  (lat, lon, timestamp) that are used to match to the SAR detects only contain MMSI.  The other fields are matched
  to the MMSI from other non-location messages.

  elevation_m is from the GEBCO gridded bathymetry data set
    https://globalfishingwatch.org/data-download/datasets/public-bathymetry-v1
    https://www.gebco.net/data_and_products/gridded_bathymetry_data/

  distance_from_shore_m is from this data set
    https://globalfishingwatch.org/data-download/datasets/public-distance-from-shore-v1
    https://pae-paha.pacioos.hawaii.edu/thredds/dist2coast.html?dataset=dist2coast_1deg

  The eez fields come from Marine Regions, Flanders Marine Institute (2019). Maritime Boundaries Geodatabase: Maritime Boundaries and Exclusive Economic Zones (200NM), version 11. Available online at https://www.marineregions.org/. https://doi.org/10.14284/386

  NB: There are about 13k detect_ids in this table that occur more than once because of overlapping EEZ areas which represent competing claims and join management areas.
  """
);

INSERT INTO
  global-fishing-watch.paper_industrial_activity.vessels_meta_v20240605
WITH vessels as (
  SELECT
    *,
    format("lon:%+07.2f_lat:%+07.2f", round(lon/0.01)*0.01, round(lat/0.01)*0.01) as gridcode,
    ST_GEOGPOINT(lon, lat) as geometry
  FROM `global-fishing-watch.paper_industrial_activity.vessels_v20231013`
),
spatial_measures as (
  SELECT
    gridcode, elevation_m, distance_from_shore_m
  FROM `world-fishing-827.pipe_static.spatial_measures_20201105`
),
gridded_eez as (
  SELECT distinct
    CAST(eez as INT64) as MRGID_EEZ,
    gridcode
  FROM `world-fishing-827.pipe_static.spatial_measures_20201105`
  cross join unnest(regions.eez) as eez
),
marine_regions as (
  select
    MRGID_EEZ, TERRITORY1, ISO_TER1, POL_TYPE, `UNION`, geometry
  from `world-fishing-827.pipe_regions_layers.EEZ_land_union_v3_202003`
),
vessel_identity as (
  SELECT
    ssvid as mmsi_str,
    year,
    best.best_flag as flag,
    ais_identity.n_shipname_mostcommon.value as shipname,
    ais_identity.n_callsign_mostcommon.value as callsign,
    ais_identity.n_imo_mostcommon.value as imo

  FROM `world-fishing-827.pipe_ais_v3_published.vi_ssvid_byyear_v20240401`
  WHERE year in (2017, 2018, 2019, 2020, 2021)
),
vessels_meta as (
  SELECT
    v.detect_id,
    m.* except(gridcode),
    r.* except(geometry),
    i.* except(mmsi_str, year)
  FROM vessels v
  LEFT JOIN spatial_measures m
  USING (gridcode)
  LEFT JOIN marine_regions r
  ON ST_CONTAINS(r.geometry, v.geometry)
  LEFT JOIN vessel_identity i
  ON(
    CAST(mmsi AS STRING) = mmsi_str
    AND EXTRACT(YEAR FROM v.timestamp) = i.year
  )
)

SELECT * from vessels_meta
