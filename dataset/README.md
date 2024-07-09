# Dataset Preparation

## Setup

Following this section read `/conf/config.yaml` if you get an error or some unexpected result.

- Create `/data` directory
- Copy or parse KazHydroMet data into `/data/kazhydromet` directory (`meteo` data has ot be called `basin_forcing` and `hydro` data has to be called `streamflow`)
- Copy `selected_basins.csv` into `/data` directory

### Google Earth Engine Data

Create `/data/GEE` directory and place all GEE attribute files inside

## Dataset

To prepare the dataset run:

```bash
python pipeline.py
```

This script will create `/data/CAMELS_KZ` dataset directory with the following structure:

- `CAMELS_KZ`
    - `attributes` (contains static attributes split into multiple files)
    - `mean_basin_forcing`
    - `streamflow` 
    - `time_series` (`.nc` files of combined dynamic attributes of meteo and hydro data)

Files contained in `time_series` are the end files used for training the models.
