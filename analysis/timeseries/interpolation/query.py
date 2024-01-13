from google.cloud import bigquery

# Old query
q_ = """
SELECT
  date_24,
  lat_index,
  lon_index,
  overpasses,
  IFNULL(matched_fishing + matched_unknown_likelyfish,0) AS ais_fishing,
  IFNULL(matched_nonfishing + matched_unknown_likelynonfish,0) AS ais_nonfishing,
  IFNULL(unmatched_fishing,0) AS dark_fishing,
  IFNULL(unmatched_nonfishing,0) AS dark_nonfishing
FROM
  scratch_david.detections_24_w_zeroes
WHERE
  overpasses = 0
  OR detections > 0
-- LIMIT
--     10000
"""

# Updated query
q = """
SELECT
  date_24,
  lat_index,
  lon_index,
  overpasses,
  IFNULL(matched_fishing + matched_unknown_likelyfish,0) AS ais_fishing,
  IFNULL(matched_nonfishing + matched_unknown_likelynonfish,0) AS ais_nonfishing,
  IFNULL(unmatched_fishing,0) AS dark_fishing,
  IFNULL(unmatched_nonfishing,0) AS dark_nonfishing,
  IFNULL(unmatched_nonfishing_100,0) AS dark_nonfishing_100,
  IFNULL(matched_nonfishing_100 + matched_unknown_likelynonfish_100,0) AS ais_nonfishing_100,
FROM
  -- scratch_david.detections_24_w_zeroes_v3
  scratch_david.detections_24_w_zeroes_v4
WHERE
  overpasses = 0
  OR detections > 0
"""

bq = bigquery.Client()
df = bq.query(q).result().to_dataframe()
df.to_feather("data/gridded_detections_v4.feather")

print(df.head())
print('DONE')
