# Spatail Analyses

This folder contains notebooks that produces a number of the spatial analyses and figures for the paper, and also produces a number of data files that are then used in figures. These notebooks include:

### AISFishingByContinent.py
Used to calculate supplemental table 1, AIS fishing by continent

### Concentration.py
This notebook calcluates how concentrated fishing and non-fishing activity is in the ocean. It shows that half of the vessels are in just 2.7% of the area.

### CreateStudyArea
Creates a polygon outlining the areas with at least 30 images across 2017-2021 and also outside of ice areas.

### CreateVesselActivityRasterDataframe.py
This notebook creates the dataframe that is used to create the raster of fishing and non-fishing activity in figure 1. 

### DetectionsWithOverpasses.py
This notebook contains a number of queries that provide metadata and detail about each SAR detection. Importantly, it includes, for each detection, how many times Sentinel-1 imaged that part of the ocean (measured at 1/200th of a degree). The result of this notebook is a table that is then used for most of the analyses. 

### EEZAnalysis.py
This notebook calcluates a number of statistics by EEZ (and the high seas), including the concentration of vessels in waters shallower than 200 meters. It also calcluates the concentration of vessels in the western North Korean EEZ for 2017-2019 and 2020-2021. 

### DownloadAllVesselDetections.py
This creates a table and downloads a dataframe to a compressed CSV that contains all likely vessel detections. These detections are in turn used to produce the figures that show individual vessels.



```python

```
