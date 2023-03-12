# Analyses

This folder contains notebooks that produce analyses and figures for the paper. These notebooks include:

### AISFishingByContinent.py
Used to calculate supplemental table 1, AIS fishing by continent

### AreaImaged.py
Produces the supplemental figure showing the area of the ocean imaged by Sentinel-1 over our time series

### Concentration.py
This notebook calcluates how concentrated fishing and non-fishing activity is in the ocean. It shows that half of the vessels are in just 2.7% of the area.

### CoverageMaps.py
This notebook produces the supplemental figure that shows the area of the ocean imaged by Sentinel-1

### DetectionsWithOverpasses.py
This notebook contains a number of queries that provide metadata and detail about each SAR detection. Importantly, it includes, for each detection, how many times Sentinel-1 imaged that part of the ocean (measured at 1/200th of a degree). The result of this notebook is a table that is then used for most of the analyses. 

### EEZAnalysis.py
This notebook calcluates a number of statistics by EEZ (and the high seas), including the concentration of vessels in waters shallower than 200 meters. It also calcluates the concentration of vessels in the western North Korean EEZ for 2017-2019 and 2020-2021. 

### Fig1-CreateVesselActivityRasterDataframe.py
This notebook creates the dataframe that is used to create the raster of fishing and non-fishing activity in figure 1. 

### Fig1BarchartData.py
This notebook creates the dataframe that is used to create the bar chart in figure 1. It also produces the supplemental figure that shows number of vessels by EEZ. It also calculates some important statistics for the paper, such as the number of vessels present on average

### Fig3-DownloadAllVesselDetections.py
This creates a table and downloads a dataframe to a compressed CSV that contains all likely vessel detections. These detections are in turn used to produce the figures that show individual vessels.


