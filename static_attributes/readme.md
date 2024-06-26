# Static Attributes Derived from Google Earth Engine

## JavaScript Files for Attribute Derivation

- **p_mean**
- **pet_mean**
- **aridity**
- **p_seasonality**
- **frac_snow_daily**
- **high_prec_freq**
- **high_prec_dur**
- **low_prec_freq**
- **low_prec_dur**
- **elev_mean**
- **slope_mean**
- **area_gages2**
- **forest_frac**
- **lai_max**
- **lai_diff**
- **gvf_max**
- **gvf_diff**
- **soil_porosity**
- **soil_conductivity**
- **max_water_content**
- **sand_frac**
- **silt_frac**
- **clay_frac**



## Datasets and Static Attributes

| Dataset                    | Static Attribute       | Description                                                                                 |
|----------------------------|------------------------|---------------------------------------------------------------------------------------------|
| Kazhydromet*               | **p_mean**             | Mean daily precipitation.                                                                   |
| MODIS/006/MOD16A2          | **pet_mean**           | Mean daily potential evapotranspiration.                                                    |
| MODIS/006/MOD16A2          | **aridity**            | Ratio of mean PET to mean precipitation.                                                    |
| Kazhydromet                | **p_seasonality**      | Seasonality and timing of precipitation.                                                    |
| Kazhydromet                | **frac_snow_daily**    | Fraction of precipitation falling on days with temperatures below 0°C.                      |
| Kazhydromet                | **high_prec_freq**     | Frequency of high-precipitation days (≥ 5 times mean daily precipitation).                  |
| Kazhydromet                | **high_prec_dur**      | Average duration of high-precipitation events.                                              |
| Kazhydromet                | **low_prec_freq**      | Frequency of dry days (< 1 mm d⁻¹).                                                         |
| Kazhydromet                | **low_prec_dur**       | Average duration of dry periods.                                                            |
| MERIT/DEM/v1_0_3           | **elev_mean**          | Catchment mean elevation.                                                                   |
| MERIT/DEM/v1_0_3           | **slope_mean**         | Catchment mean slope.                                                                       |
| -                          | **area_gages2**        | Catchment area.                                                                             |
| ESA/WorldCover/v100        | **forest_frac**        | Forest fraction.                                                                            |
| MODIS/006/MCD15A3H         | **lai_max**            | Maximum monthly mean of leaf area index.                                                    |
| MODIS/006/MCD15A3H         | **lai_diff**           | Difference between the max. and min. mean of the leaf area index.                           |
| MODIS/006/MOD13Q1          | **gvf_max**            | Maximum monthly mean of green vegetation fraction.                                          |
| MODIS/006/MOD13Q1          | **gvf_diff**           | Difference between the maximum and minimum monthly mean of the green vegetation fraction.   |
| HiHydroSoilv2_0/ksat       | **soil_conductivity**  | Saturated hydraulic conductivity.                                                           |
| HiHydroSoilv2_0/wcsat      | **max_water_content**  | Maximum water content of the soil.                                                          |
| soilgrids-isric/sand_mean  | **sand_frac**          | Fraction of sand in the soil.                                                               |
| soilgrids-isric/silt_mean  | **silt_frac**          | Fraction of silt in the soil.                                                               |
| soilgrids-isric/clay_mean  | **clay_frac**          | Fraction of clay in the soil.                                                               |

\* Attributes with dataset Kazhydromet calculated in files: `catchment_attributes_calc.ipynb`

## Reference

Kratzert, F., Klotz, D., Shalev, G., Klambauer, G., Hochreiter, S., and Nearing, G.: Towards learning universal, regional, and local hydrological behaviors via machine learning applied to large-sample datasets, Hydrol. Earth Syst. Sci., 23, 5089–5110, https://doi.org/10.5194/hess-23-5089-2019, 2019.