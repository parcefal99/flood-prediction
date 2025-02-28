# Sigma Dataset Documentation

## Overview

### Dataset Name
**Sigma Dataset**

### Description
This dataset consists of meteorological (time series) and geophysical (catchment attributes) data of 85 basins in Kazakhstan. It is intended for use in weather forecasting or modeling, as well as flood prediction based on the attributes provided.

We developed basin-scale hydrometeorological forcing data for 85 basins in the conterminous Kazakhstan basin subset. Retrospective model forcings are computed from **ERA5-Land** forcing data run from **1 Jan 2000 to 31 Dec 2022**. Model time series output is available for the same time periods as the forcing data.

### Data Sources and Attributes
- **Topographic Characteristics**
  - Retrieved from **MERIT** data.
  - Examples: Elevation, Slope.
  
- **Climatic Indices**
  - Computed using time series provided by **Newman et al. (2015)**.
  - Examples: Aridity, Frequency of dry days.

- **Hydrological Signatures**
  - Computed from time series.
  - Examples: Mean annual discharge, Baseflow index.

- **Soil Characteristics**
  - Characterized using **soilgrids-isric** and **HiHydroSoilv2_0** dataset.
  - Examples: Porosity, Soil depth.

- **Vegetation Characteristics**
  - Inferred using **MODIS** data.
  - Examples: Leaf area index, Rooting depth.

## Usage
This dataset is intended for use in:
- **Weather forecasting**
- **Hydrological modeling**
- **Flood prediction**
