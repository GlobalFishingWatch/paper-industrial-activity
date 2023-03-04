# Matching Scripts

This folder contains scripts that can match vessels detected in SAR to vessels broadcasting AIS. It builds on the methods described in [Kroodsma et al. 2022](https://www.nature.com/articles/s41598-022-23688-7).

We first run the `extrapolate` and `score` steps that are outlined in [this folder](https://github.com/GlobalFishingWatch/paper-longline-ais-sar-matching/tree/master/5_SAR_AIS_Matching) of the repository associated with the above paper. The extrapoloate step gets the vessels that could appear in each SAR scene, and the AIS position before and after the scene, and several other statics that help in matching. The score step calculates the score based on averaging the two probabilty raseters (see methods of the previously mentioned paper). We did not run the multiply step, as it was too costly using the queries in this repo, and instead updated it using the notebooks below. 


### LikelihoodInScene.py

This notebook calculates, for every AIS vessel that could have appeared in the scene, the likelihood it appeared in the scene based on multiplying a probability raster associated with the AIS position before and after the scene. This method is described in the methdos of [Kroodsma et al. 2022](https://www.nature.com/articles/s41598-022-23688-7).

### RecallCurve.py

This notebook calculates the recall for our Sentinel-1 vessel detection algorithm as a function of vessel length and the distance between a vessel and the closest other vessel with AIS. 


### MatchSARMultiplied.py

This notebook calcluates the "muliplied" score for potential AIS to SAR detections (as opposed to the "averaged" score), and then combines this score with the averaged score, the likelihood of a vessel appearing in a scene, the expected recall of a vessel, and the match between the vessel and detection lengths. It then choooses the most likely matches between SAR detections and AIS, and assigns each match a score.


### GetMatchingThreshold.py

This notebook cacluates how many vessels we expect to have detected with SAR based on 1) the recall curve calculated in `RecallCurve.py` and the likelihood vessels are in scenes, calculated with the notebook `LikelhoodInScene.py`. It then picks the score threshold for matching that gives that number of matches between SAR and AIS. 

