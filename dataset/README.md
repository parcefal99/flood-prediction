# Dataset Preparation

## Setup

Following this section read `/conf/config.yaml` if you get an error or some unexpected result.

- Create `/data` directory
- Copy or parse KazHydroMet data into `/data/kazhydromet` directory (`meteo` data has ot be called `basin_forcing` and `hydro` data has to be called `streamflow`)
- Copy `selected_basins.csv` into `/data` directory

### Google Earth Engine Data

Create `/data/GEE` directory and place all GEE attribute files inside

## Pipeline

To prepare static attributes from dynamic ones run:

```bash
python pipeline.py
```
