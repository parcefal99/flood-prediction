# Dataset Preparation

## Dynamic and Static Features

Upload the shapefiles to Google Earth Engine and use the following `.js` scripts to obtain raw meteo data (from ERA5) and catchment attributes (with global datasets specified):
- Dynamic features:
    - `era5.js`
- Static features: 
    - `area_gages2.js`
    - `elev_mean.js`
    - `forest_frac.js`
    - `gvf_max_diff.js`
    - `lai_max_diff.js`
    - `max_water_content.js`
    - `pet_mean.js`
    - `sand_silt_clay.js`
    - `soil_conductivity.js`


## Streamflow 

There are two options: Web Portal and PDF Yearbooks (both from KazHydroMet). The former could be accessed using the [web_parser_meteo.py](./kazhydromet_parser/web_parser_meteo.py), which obtains meteo data from all basins of Kazakhstan. The latter is parsed using LlamaParse, check [llama_parse](../llama_parse/) for detailed information with our codes.

## Preprocessing 

Use the following scripts - [raw_to_fancy.ipynb](./raw_to_fancy.ipynb) and [static_clim.ipynb](./gee_scripts/preprocessing_clim/static_clim.ipynb) - to obtain dataset for multiple basins in .csv format.

Note that streamflow values should be normalized by the area of subsequent basin. Check this script [streamflow_normalization.ipynb](./streamflow_normalization.ipynb).

## NetCDF 

To access the NeuralHydrology models and start training, the dataset should create `timeseries` in NetCDF format, which is essentially concatenation of dynamic/static features and streamflow data. Files contained in `time_series` are the end files used for training the models.
The script is available here - [convert_to_nc.py](./convert_to_nc.py). 
