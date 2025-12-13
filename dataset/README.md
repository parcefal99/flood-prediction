# KazFlow85 Dataset Documentation

## Overview

### Dataset Name
KazFlow85 Dataset

### Description
This dataset consists of meteorological (time series) and geophysical (catchment attributes) data of 85 basins in Kazakhstan. It is intended for use in weather forecasting or modeling, as well as flood prediction based on the attributes provided.

We developed basin-scale hydrometeorological forcing data for 85 basins in the conterminous Kazakhstan basin subset. Retrospective model forcings are computed from ERA5-Land forcing data run from 1 Jan 2000 to 31 Dec 2022. Model time series output is available for the same time periods as the forcing data.


| ID     | Latitude  | Longitude  | Location |
|--------|----------|-----------|--------------------------------|
| 11001  | 48.004531  | 85.222161  | с.Боран |
| 11063  | 48.434364  | 85.838581  | Теректы  с. Мойылды (Николаевка) |
| 11068  | 48.133153  | 85.196547  | с.Калжыр |
| 11077  | 47.283392  | 84.106714  | с.Сарыолен |
| 11094  | 49.009844  | 82.788169  | с. Джумба |
| 11108  | 48.606183  | 83.869081  | с. Вознесенка |
| 11117  | 49.206781  | 84.517825  | с.Улькен Нарын |
| 11124  | 49.365844  | 86.448403  | с. Берель |
| 11126  | 49.331689  | 85.171111  | с. Печи (с.Барлык) |
| 11129  | 49.829589  | 84.329319  | с. Лесная пристань |
| 11131  | 49.2269    | 85.8861    | Черновое |
| 11143  | 49.378308  | 85.410133  | с. Белое |
| 11146  | 49.510986  | 84.391953  | с. Средигорное |
| 11147  | 49.891314  | 84.077142  | с. Кутиха |
| 11160  | 49.393872  | 82.560883  | с. Алгабас |
| 11163  | 50.276683  | 83.377817  | г.Риддер |
| 11164  | 50.010614  | 82.841592  | с. Ульби Перевалочная |
| 11170  | 50.190344  | 82.580083  | с.Белокаменка |
| 11187  | 49.985264  | 82.262047  | с. Отрадное |
| 11188  | 50.247281  | 82.244192  | с. Предгорное |
| 11199  | 50.7686    | 83.0375    | Каракожа |
| 11207  | 50.612714  | 81.876767  | с. Шемонаиха |
| 11233  | 49.163231  | 81.969733  | аул Кентарлау |
| 11661  | 49.834997  | 82.645803  | с. Самсоновка |
| 11242  | 51.736944  | 72.305     | с.Новомарковка |
| 11272  | 51.537222  | 71.937778  | с.Приречное |
| 11275  | 52.494444  | 73.285     | свх.Изобильное |
| 11291  | 53.099444  | 69.012222  | с.Павловка |
| 11395  | 50.62      | 73.026667  | с.Приишимское |
| 11424  | 51.785833  | 69.463889  | с. Калкутан |
| 11432  | 52.536944  | 68.789167  | с. Балкашино |
| 11433  | 51.797778  | 68.368889  | г. Атбасар |
| 11453  | 52.747222  | 67.669722  | Бурлукс.Гусаковка |
| 11461  | 53.651389  | 67.413056  | с.Соколовка |
| 11468  | 52.579167  | 67.680833  | с.Ковыльное |
| 11469  | 52.794167  | 66.759444  | с.Возвышенка |
| 13048  | 48.384444  | 67.847778  | с.Малшыбай |
| 13061  | 49.35      | 74.466111  | с.Бесоба |
| 13064  | 49.9675    | 74.051944  | с.Шешенкара |
| 13090  | 49.113889  | 73.468611  | п.Шопан |
| 13091  | 49.256389  | 72.945556  | жд.ст.Карамурын |
| 13105  | 49.099167  | 75.86      | с.Новостройка |
| 13115  | 48.558056  | 70.446667  | рзд.189 км |
| 13128  | 48.668056  | 71.639444  | п.Атасу |
| 13142  | 49.871722  | 72.576086  | с.Каражар |
| 13148  | 50.254722  | 71.566667  | п.Киевка |
| 13198  | 49.668611  | 69.550278  | п.Баршино |


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
