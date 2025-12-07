# Rainfall–Runoff Modeling for Kazakhstan

Rainfall–Runoff Modeling for Northern Kazakhstan using LSTM and Deep Learning approaches.

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![NeuralHydrology](https://img.shields.io/badge/NeuralHydrology-Model-blue?style=for-the-badge)
![GEE](https://img.shields.io/badge/-white?style=for-the-badge&logo=googleearth&logoColor=white&label=Google%20Earth%20Engine&labelColor=A0522D)

# Description

Flooding constitutes a major hydro-meteorological risk across numerous regions of Kazakhstan, yet there is currently **no publicly accessible, ML-ready hydrological dataset or flood-prediction system** developed specifically for the country’s river basins. This repository addresses this gap by assembling, harmonizing, and structuring a multi-source dataset that integrates streamflow observations, meteorological forcings, soil and land-cover information, and catchment characteristics for several basins within Northern Kazakhstan. The resulting resource represents the **first machine-learning hydrology dataset tailored to the Kazakh context**, thereby enabling data-driven flood-forecasting research in a region where such tools have not previously been available.

The modeling workflow builds upon *NeuralHydrology*, a state-of-the-art deep learning framework for rainfall–runoff and hydrological prediction tasks (Kratzert et al., 2022). NeuralHydrology provides a modular pipeline for training LSTM-based and related architectures using simple YAML-based configuration files rather than bespoke implementations. More information on the framework is available at the project’s official page: [Neural Hydrology](https://research.google/pubs/neuralhydrology-a-python-library-for-deep-learning-research-in-hydrology/)

To ensure full compatibility with NeuralHydrology’s data loaders, extensive preprocessing was conducted. Local hydrological records (e.g., discharge data from KazHydroMet) were combined with catchment boundaries derived from shapefiles, and with global geospatial datasets, including satellite-derived forcings, reanalysis products, soil attributes, and land-cover classifications. These inputs were standardized, temporally aligned, and spatially aggregated following established rainfall–runoff modeling conventions.

This dataset constitutes a **novel scientific contribution** in several respects:

* It provides the **first publicly available, ML-ready hydrological dataset for Kazakhstan**, integrating national in-situ observations with static and gridded global data sources.
* It facilitates the application of contemporary rainfall–runoff modeling approaches, especially deep learning–based flood forecasting, in a region that has historically lacked such computational resources.
* It establishes a foundation for further research on hydrological extremes, operational early-warning systems, and sustainable water-resources planning in Central Asia, expanding the scope of machine-learning hydrology into an underrepresented geographic setting.


# Data

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
...   


To prepare the full dataset see the Dataset [README](./dataset/README.md) file. Another (faster) option is to download the ready dataset from [HuggingFace](https://huggingface.co/datasets/floodpeople/sigma_dataset/tree/main) and check the `KazFlow85_dataset.zip`. 


# Project Build Steps


## Dataset Collection

The collection of the data was performed in several steps from serveral sources. The main sources were KazHydroMet (for streamflow) and Google Earth Engine (for meteo data and static attributes).


### Streamflow Data Parsing

The web-parser script automates the extraction of streamflow data from the [KazHydroMet Web Portal](http://ecodata.kz:3838/app_hydro_en/) using Selenium. It navigates the portal, retrieves a list of hydrological posts, and downloads discharge data for each hydropost, saving it as CSV files in a structured directory. The script handles Cyrillic-to-Latin transliteration, cleans data by converting special characters and non-numeric entries (e.g., "нб" to "0", "-" to NaN), and ensures consistent formatting. The script includes error handling for timeouts and stale elements, with a retry mechanism to resume from the last processed station. To run, users should ensure a stable internet connection and adjust the script's timing parameters if needed to accommodate network or server response variations. The output provides a comprehensive list of CSV files of discharge records, ready for further steps. See [KazHydroMet Parser](./dataset/kazhydromet_parser/).

The streamflow data from KazHydroMet portal was in question, therefore, our team parsed PDF annual yearbooks provided by KazHydroMet. For that [llama_parse](https://www.llamaindex.ai/llamaparse) service was utilized. The parsed data were obtained in the `Markdown` format, later converted into `CSV` files, and merged after. See [llama_parse](./llama_parse/) folder.


### Basin Shapefiles

In order to obtain basin shapefiles the package [delineator](https://github.com/mheberger/delineator) was used. The coordinates of hydrological stations were provided to this package, which determined the basin shapes.

Delineator produced shapes containg holes, they were filled in QGis.


### Extracting Data from Google Earth Engine

Google Earth Engine (GEE) is a cloud-based platform for processing and analyzing large-scale geospatial data. It provides access to extensive satellite imagery and geospatial datasets, allowing users to perform advanced environmental and earth science analyses using JavaScript or Python APIs.

In order to obtain the meteorological forcings and catchment attributes, upload the basin shapefiles ([link](https://huggingface.co/datasets/floodpeople/sigma_dataset)) to GEE platform and start extracting the features using the guidelines stated [there](./dataset/README.md). Dynamic features (meteo) are obtained from ERA5_LAND dataset, and  static features are from MODIS, MERIT, ESA, HiHydroSoilv2_0, soilgrids-isric datasets using Google Earth Engine.
 

### Streamflow Data Availability

Notebook [plot_data_discharge.ipynb](./dataset/plot_data_discharge.ipynb) can be used to draw the streamflow data availability of all the specified basins in the dataset.


# Setup

## Virtual environment

Create a python virtual environment to install all the necessary packages:

```bash
python -m venv .venv
```

Alternatively, you can use conda:

```bash
conda create -n flood_proj python=3.10.12
```

## Installation

Install the required packages into the created virtual environment

```bash
pip install -r requirements.txt
```

## WandB Logging

Create `.env` file in the root directory and add `WANDB_ENTITY` entry.


# Training

[NeuralHydrology](https://github.com/neuralhydrology/neuralhydrology) library is used underneath this repo. Directory [ML](./ML/) contains training, evaluation, finetuning and other scripts related to ML models.

For details, see the ML [README](./ML/README.md) file.


# Data Availability

## Licensing and Use

* The **processed dataset** released here is available under a **CC BY 4.0 license**, allowing reuse for research, educational, and non-commercial purposes with proper attribution.
* **Raw data from KazHydroMet** (national hydrometeorological agency) cannot be redistributed due to licensing restrictions. Users wishing to access the original raw data should consult the KazHydroMet portal ([http://www.kazhydromet.kz/](http://www.kazhydromet.kz/)) and follow their data access procedures.

## Attribution

* When using this dataset, users should **cite both sources** if applicable:

  1. The **processed dataset** from this repository
  2. The **original KazHydroMet data**, if incorporated in analyses

## Reproducibility

* The repository includes **preprocessing scripts** that document how raw and global datasets were transformed into ML-ready inputs.
* Researchers with access to the original KazHydroMet data can **reproduce the processed dataset** by following these scripts.
* Global data sources (e.g., satellite or reanalysis products) are fully open-access and referenced in the scripts and documentation.