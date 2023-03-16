# What does the code do
## Public repository for the souce code used in the paper

[Satellite mapping reveals extensive undisclosed industrial activity at sea](http://#)

by [Global Fishing Watch](http://#) _et al_.

## Content

`detector/` - Source code of SAR detection system in Earth Engine

`nnets/` - Source code of Deep Learning models and training

`analyses/` - Source code and notebooks of analyses and figures

# How to install or setup environment
Most of the code in this repository relies on a common environment used by the Global Fishing Watch Research and Innovation team.

## Usage
### Creating a new environment
To create an environment with the default name of rad execute:

conda env create -f radenv.yaml 

To create an environment with a different name use:

conda env create -f radenv.yaml -n ENV_NAME

### Turning the repo into an installable python packages and `setup.cfg`

Turning your repo into an installable python package allows for your core functions to be accessed throughout the repo without having to define paths. It will make your imports very clean and store core functionality in a central place so that multiple users of the repo can benefit from functions that may have previously lived only in one person's notebook.

Follow these steps to turn your repo into an installable python package:
1. Setup and activate the conda environment.
2. Build your environment then use `pip install -e .`. This will create a folder titled `pkg.egg-info` and will allow you to access the code within your `pkg` folder from outside of that folder.


# How to run things

# Citing this work
Register and update with Zenodo
