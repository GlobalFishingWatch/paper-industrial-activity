# Model Features
This folder contains analyses that produce featuers for the nerual net model. We also built a random forest model that draws on many of these same features. 

### GenerateFeatures.py
This notebook produces features for the random forest and the neural net classifier that determines if a vessel is a fishing vessel. Note that for the paper we did not use the random forest classifer.


### UploadNPP.py
This notebook processes NPP data from https://oceandata.sci.gsfc.nasa.gov and uploads the data to BigQuery for other analyses.