## Code for analysis and figures

    .
    ├── data            # expected location of data for notebooks
    ├── features        # environmental rasters for model features
    ├── figures         # notebooks of paper figures
    ├── infrastructure  # analysis of infrastructure detections
    ├── matching        # AIS to SAR detections matching pipeline
    ├── rasterize       # rasterization of SAR footprints
    ├── roads           # filtering of ambiguities from vehicles
    ├── spatial         # spatial analysis and creation of data for figures
    ├── timeseries      # moving window and interpolation of time series
    ├── uncertainty     # bootstrap standard error of percent change
    └── utils           # shared functionality for analyses


## Building conda environment

The packages required to run the analysis code are included in the `env.yaml`. To create an environment from the env.yml with the default name `rad`:
`conda env create -f env.yaml`

You can specify the environment name using:
`conda env create -n name -f env.yaml`


## Download the data

Download data from [here](https://figshare.com/s/8157ca5d6f0014226d7f) into the `./data/' directory.