CREATE OR REPLACE TABLE
`global-fishing-watch.paper_industrial_activity.vessels_meta_v20231106` (
  mmsi STRING OPTIONS (description = 'Mobile Maritime Service Identifier. This is the primary ID that the vessel broadcasts'),
  year INT64 OPTIONS (description = 'Calendar year this record applies to'),
  flag_iso STRING OPTIONS (description = 'Three letter ISO code for the flag state.  This is the country that issued the MMSI'),
  shipname STRING OPTIONS (description = 'Name of the vessel'),
  callsign STRING OPTIONS (description = 'Radio callsign'),
  imo STRING OPTIONS (description = 'IMO number')
) OPTIONS (
  description = """Extended metadata for vessels keyed by MMSI.

  Note that many vessels will change their identifiers over the 5-year study period, and the only identifier that
  is unchangeable is the IMO number.  This table contains the best identifiers associated with each MMSI over the
  course of a calendar year.  However these can change and it is possible for more than one vessel to use the same
  MMSI at the same time.

  Not all MMSI will have all the associated identity fields.  The AIS messages that contain the position information
  (lat, lon, timestamp) that are used to match to the SAR detects only contain MMSI.  The other fields are matched
  to the MMSI from other non-location messages.
  """
);

INSERT INTO
  global-fishing-watch.paper_industrial_activity.vessel_meta_v20231106
SELECT
  ssvid as mmsi,
  year,
  best.best_flag as flag,
  ais_identity.n_shipname_mostcommon.value as shipname,
  ais_identity.n_callsign_mostcommon.value as callsign,
  ais_identity.n_imo_mostcommon.value as imo

FROM `world-fishing-827.pipe_ais_v3_published.vi_ssvid_byyear_v20240401`
WHERE year in (2017, 2018, 2019, 2020, 2021)



