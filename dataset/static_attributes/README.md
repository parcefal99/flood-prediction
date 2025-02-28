# 1. GEE Scripts.

## 1.1 Data Attributes Table
## 5 dynamic and 22 static attributes were computed based on the guidelines in Krazert et al(2019b). 
(While 27 static attributes were used in Kratzert et al. (2019b), five attributes, such as depth to bedrock, soil depth,
soil porosity, carbonate rocks fraction, and geological permeability were unavailable for Northern Kazakhstan.)

This table provides an overview of the data attributes, their definitions, and sources.

## Dynamic Attributes

| Data Attribute        | Definition                                              | Source       |
|----------------------|------------------------------------------------------|-------------|
| *t_max*            | Daily maximum air temperature [°C]                    | Era5-Land   |
| *t_min*            | Daily minimum air temperature [°C]                    | Era5-Land   |
| *prcp*             | Average daily precipitation [mm/day]                   | Era5-Land   |
| *srad*             | Surface-incident solar radiation [W/m²]                | Era5-Land   |
| *vp*               | Near-surface daily average vapor pressure [Pa]         | Era5-Land   |
| *discharge_shift1* | Water flow rate per unit basin area, shifted forward by one day [mm/day] | Kazhydromet |

## Static Attributes

| Data Attribute        | Definition                                              | Source       |
|----------------------|------------------------------------------------------|-------------|
| *elev_mean*        | Catchment mean elevation [m]                          | MERIT       |
| *slope_mean*       | Catchment mean slope [m/km]                           | MERIT       |
| *area_gages2*      | Catchment area [km²]                                  | MERIT       |
| *forest_frac*      | Fraction of catchment area covered by forest          | ESA         |
| *lai_max*         | Maximum monthly mean of leaf area index                | MODIS       |
| *lai_diff*        | Difference between maximum and minimum monthly mean leaf area index | MODIS       |
| *gvf_max*         | Maximum monthly mean of green vegetation fraction      | MODIS       |
| *gvf_diff*        | Difference between maximum and minimum monthly mean green vegetation fraction | MODIS       |
| *soil_conductivity* | Saturated hydraulic conductivity [cm/hr]              | HiHydroSoilv02 |
| *max.water_content* | Maximum water content of the soil [m]                 | HiHydroSoilv02 |
| *sand_frac*       | Fraction of sand in the soil [%]                        | SoilGrids-ISRIC |
| *silt_frac*       | Fraction of silt in the soil [%]                        | SoilGrids-ISRIC |
| *clay_frac*       | Fraction of clay in the soil [%]                        | SoilGrids-ISRIC |
| *p_mean*          | Mean daily precipitation [mm/day]                       | Era5-Land   |
| *pet_mean*        | Mean daily potential evapotranspiration [mm/day]        | MODIS       |
| *aridity*         | Ratio of mean PET to mean precipitation                 | MODIS       |
| *p_seasonality*   | Estimated by representing annual precipitation and temperature as sine waves | Era5-Land   |
| *frac_snow_daily* | Fraction of precipitation falling on days with temp < 0°C | Era5-Land   |
| *high_prec_freq*  | Frequency of days with prcp ≥ 5 × p_mean [days/year]   | Era5-Land   |
| *high_prec_dur*   | Average duration of high precipitation events [days]    | Era5-Land   |
| *low_prec_freq*   | Frequency of dry days with prcp < 1 mm/day [days/year]  | Era5-Land   |
| *low_prec_dur*    | Average duration of dry periods [days]                  | Era5-Land   |

## Target Attribute

| Data Attribute  | Definition                                      | Source       |
|----------------|----------------------------------------------|-------------|
| *discharge*   | Water flow rate per unit basin area [mm/day] | Kazhydromet |


\* Attributes with dataset Kazhydromet calculated in files: `catchment_attributes_calc.ipynb`

## Reference

Kratzert, F., Klotz, D., Shalev, G., Klambauer, G., Hochreiter, S., and Nearing, G.: Towards learning universal, regional, and local hydrological behaviors via machine learning applied to large-sample datasets, Hydrol. Earth Syst. Sci., 23, 5089–5110, https://doi.org/10.5194/hess-23-5089-2019, 2019.
