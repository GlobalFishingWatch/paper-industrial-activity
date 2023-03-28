## Code for the analyses and figures

    .
    ├── features        # environmental rasters for model features
    ├── figures         # notebooks of paper figures
    ├── infrastructure  # analysis of infrastructure detections
    ├── matching        # AIS to SAR detections matching pipeline
    ├── rasterize       # rasterization of SAR footprints
    ├── roads           # filtering of ambiguities from vehicles
    ├── spatial         # ???
    └── timeseries      # moving window and interpolation of time series


## Building conda environment

The packages required to run the anlysis code are included in the `env.yaml`. 

1. To create an environment from the env.yml with the default name `rad`:
`conda env create -f env.yaml`

You can specify the environment name using:
`conda env create -n name -f env.yaml`

## Installable python packages and `setup.cfg`

Turning your repo into an installable python package allows for your core functions to be accessed throughout the repo without having to define paths. 

Follow these steps to turn your repo into an installable python package:
1. Setup and activate the conda environment (See [Building conda environment](#building-conda-environment))
2. Build your environment then use `pip install -e .[all]`. This will create a folder titled `pkg.egg-info` and will allow you to access the code within your `pkg` folder from outside of that folder.
