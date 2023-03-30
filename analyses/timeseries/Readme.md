# Time Series Analyses

This folder contains the code usesed to calculate a time series of vessel activity.

### ExtrapolateTimeSeries.py
This notebook creates the table proj_global_sar.detections_24_w_zeroes_v20230219, which is then used to extrapolate missing positions. After dividing our time series into 24 day increments, this then identivies, for each 200th of a degree by 200th of a a degree cell, the number of detections in each time increment and the number of overpasses. The trick is that it includes when there are 0 overpasses in a given cell for a time period. 


### 24DayAugmented.py
This notebook calcluates a 24-day, rolling average of the number of detections (fishing and non-fishing) for each EEZ and globally, and creates the CSV 24day_rolling_augmented_v20230220.feather, which is used in Figure 4. This notebook draws on the interpolated time series that fills in time periods where we are missing data for the time series. It also produces the supplemental figure showing fishing activity in the western North Korean EEZ. 
