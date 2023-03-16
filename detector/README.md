# DEPRECIATED

Need to write this. All the information below is invalid!

# sentinel_1_vessel_detection_vm
For running vessel detections on virtual machines


The full sentinel-1 processing pipeline has many steps. For each day, the following is run:

1. Run detections in earth engine. These are run by ee-sar-v20200410.py, which has three outputs: 1) a geojson file with all the detections, 2) the footprint of the radar images, and 3) the footprints of the radar images, but clipped to x km from shore (x can be 1 to 5, depending on the version of the script). These outputs are all saved to GCS.

2. Upload the outputs to BigQuery. 

3. Rasterize the scene footprints for the day. This is a query run off of the footprints that were uploaded to bigquery.

4. Generate sentinel-1 satellite locations. These locations will be used in the matching query to account for the doppler shift

5. The queries that match radar detections to AIS/VMS





## Notes on the Earth Engine Script

### Key earth engine assets:
 - users/brianwong-gfw/ikea/ne_110m_ocean
 - users/brianwong-gfw/ikea/olr/ocean-land-mask-100m-v20190514

read more about this mask here:
https://docs.google.com/document/d/1_whvRkmPyM4vcuncLcObqC7Jb0EurHtbx7XqomStDb0/edit
see the mask on ee here: https://code.earthengine.google.com/ef68f4c9377266041d9b58c8ebd2fb0e
and how it was created: https://code.earthengine.google.com/ae2f7f4914b2f7ea75ab2b11f630735b 


var minor_islands = ee.FeatureCollection("users/brianwong-gfw/ikea/ne_10m_minor_islands"),
    reefs = ee.FeatureCollection("users/brianwong-gfw/ikea/unep-wcmc-global-reefs"),
    sayres_small_islands = ee.FeatureCollection("users/brianwong-gfw/prj-geeoid/oceanlandraster/sayres_et_al_small_islands_table_30m"),
    sayres_big_islands = ee.FeatureCollection("users/brianwong-gfw/prj-geeoid/oceanlandraster/sayres_et_al_big_islands_table_30m")
  
var gshhg_f = ee.FeatureCollection('users/skytruth-data/offshoreInfrastructureSupplementalShapefiles/GSHHG_f_L1');
var gshhg_h = ee.FeatureCollection('users/skytruth-data/offshoreInfrastructureSupplementalShapefiles/GSHHG_h_L1');
var osm = ee.FeatureCollection('users/skytruth-data/offshoreInfrastructureSupplementalShapefiles/osmLandPolygons');
var esri = ee.FeatureCollection('users/christian/General_Data/Terrestrial/WorldCountries_ESRI');

### [Brian's Guide to Sar](https://docs.google.com/document/d/13eUsBSrPEVRsjZKZPi1ic6R5qlH4PqVDgdSxAi0VwOs/edit#)


This is where we track what the vms are running: https://docs.google.com/spreadsheets/d/1FHR5jBrf7QGi3JTdy9ASeCPsBwllwfNnULOjyFueRn0/edit#gid=0
