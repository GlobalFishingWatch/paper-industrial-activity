# Interpolate missing pixels using a sparse 3D tensor.

Linearly interpolate the time series at the missing pixel locations.

Edit parameters in the header of each script, and run (in the following order):

Run query to get pixels to interpolate -> feather dataframe

    python query.py

Interpolate one 3D field at a time:

    python interp.py

    Fields:
    ais_fishing
    ais_nonfishing
    dark_fishing
    dark_nonfishing
    ais_nonfishing_100
    dark_nonfishing_100

Merge the interpolated 3D fields and upload to BigQuery.

    python merge.py
