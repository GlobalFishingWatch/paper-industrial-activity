CREATE OR REPLACE TABLE
global-fishing-watch.paper_industrial_activity.offshore_infrastructure_meta_v20231106 (
  structure_id INTEGER OPTIONS (description = 'Unique identifier for all detections of the same structure'),
  elevation_m FLOAT64 OPTIONS (description = 'Vertical distance from mean sea level in meters'),
  distance_from_shore_m FLOAT64 OPTIONS (description = 'Distance from shore in meters'),
  MRGID_EEZ INT64 OPTIONS (description = 'Marine Regions identifier from '),
  TERRITORY1 STRING OPTIONS (description = 'Marine Regions territory name'),
  ISO_TER1 STRING OPTIONS (description = 'Marine Regions ISO country identifier'),
  POL_TYPE STRING OPTIONS (description = 'Marine Regions political unit type'),
  `UNION` STRING OPTIONS (description = 'Marine Regions union')
) OPTIONS (
  description = """Extended metadata for structures based on the lat/lon location for each structure id.

elevation_m is from the GEBCO gridded bathymetry data set
  https://globalfishingwatch.org/data-download/datasets/public-bathymetry-v1
  https://www.gebco.net/data_and_products/gridded_bathymetry_data/

distance_from_shore_m is from this data set
  https://globalfishingwatch.org/data-download/datasets/public-distance-from-shore-v1
  https://pae-paha.pacioos.hawaii.edu/thredds/dist2coast.html?dataset=dist2coast_1deg

The eez fields come from Marine Regions, Flanders Marine Institute (2019). Maritime Boundaries Geodatabase: Maritime Boundaries and Exclusive Economic Zones (200NM), version 11. Available online at https://www.marineregions.org/. https://doi.org/10.14284/386

NB: There are 4 structures that appear twice in the meta data table because Marine Regions has overlapping boundaries for Peru and Ecuador and these
structures are located in the overlap region
  """
);

INSERT INTO
global-fishing-watch.paper_industrial_activity.offshore_infrastructure_meta_v20231106
with structures as (
  SELECT
    *,
    format("lon:%+07.2f_lat:%+07.2f", round(lon/0.01)*0.01, round(lat/0.01)*0.01) as gridcode,
    ST_GEOGPOINT(lon, lat) as geometry
  FROM `global-fishing-watch.paper_industrial_activity.offshore_infrastructure_v20231106`
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
structures_meta as (
  SELECT
    s.structure_id,
    m.* except(gridcode),
    r.* except(geometry)
  FROM structures s
  LEFT JOIN sparial_measures m
  USING (gridcode)
  LEFT JOIN marine_regions r
  ON ST_CONTAINS(r.geometry, s.geometry)
)

SELECT distinct * FROM structures_meta
