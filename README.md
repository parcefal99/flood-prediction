# Flood Prediction Project

Flood prediction using LSTM and Deep Learning approaches.

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)

# Setup

## Virtual environment

Create a python virtual environment to install all the necessary packages:

```bash
python -m venv .venv
```

or

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


# Data

To prepare the dataset see the [Dataset README file](./dataset/README.md).


# Training

For training models see the [ML README file](./ML/README.md).


# Project Build Steps


## Dataset Collection

The collection of the data was performed in several steps from serveral sources. The main sources were KazHydroMet (for streamflow) and Google Earth Engine (for meteo data and static attributes).


### Streamflow Data Parsing

The web-parser script automates the extraction of streamflow data from the [KazHydroMet Web Portal](http://ecodata.kz:3838/app_hydro_en/) using Selenium. It navigates the portal, retrieves a list of hydrological posts, and downloads discharge data for each hydropost, saving it as CSV files in a structured directory. The script handles Cyrillic-to-Latin transliteration, cleans data by converting special characters and non-numeric entries (e.g., "нб" to "0", "-" to NaN), and ensures consistent formatting. The script includes error handling for timeouts and stale elements, with a retry mechanism to resume from the last processed station. To run, users should ensure a stable internet connection and adjust the script's timing parameters if needed to accommodate network or server response variations. The output provides a comprehensive list of CSV files of discharge records, ready for further steps. See [KazHydroMet Parser](./dataset/kazhydromet_parser/).

The streamflow data from KazHydroMet portal was in question, therefore, our team parsed PDF annual yearbooks provided by KazHydroMet. For that [llama_parse](https://www.llamaindex.ai/llamaparse) service was utilized. The parsed data were obtained on Markdown format, which were converted in CSV files, and merged after. See [llama_parse](./llama_parse/) folder.


### Basin Shapefiles

In order to obtain basin shapefiles the package [delineator](https://github.com/mheberger/delineator) was used. The coordinates of hydrological stations were sent into this package, which provided the shape region basins.

Delineator produced shapes containg holes, which were filled in QGis.


### Extracting Data from Google Earth Engine

Google Earth Engine (GEE) is a cloud-based platform for processing and analyzing large-scale geospatial data. It provides access to extensive satellite imagery and geospatial datasets, allowing users to perform advanced environmental and earth science analyses using JavaScript or Python APIs.

In order to obtain the meteorological forcings and catchment attributes, upload the basin shapefiles ([link](https://huggingface.co/datasets/floodpeople/sigma_dataset)) to GEE platform and start extracting the features using the guidelines stated [there](./dataset/README.md). Dynamic features (meteo) are obtained from ERA5_LAND dataset, and  static features are from MODIS, MERIT, ESA, HiHydroSoilv2_0, soilgrids-isric datasets using Google Earth Engine.
 
