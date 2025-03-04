# PDF Annual Yearbooks Parsing

## Setup

Add `LLAMA_CLOUD_API_KEY` entry inside `.env` file in the project's root. Create it if it doesn't exist.

## Tables Extraction

Data from [KazHydroMet Portal for Annual Reports](https://www.kazhydromet.kz/ru/gidrologiya/ezhegodnye-dannye-o-rezhime-i-resursah-poverhnostnyh-vod-sushi-eds) must be downloaded first. Then [pdf_reports_parse.py](./pdf_reports_parse.py) script was used to extract the tables with streamflow data for the selected basins from the PDF contents. This script was also responsible for table cropping.


## Parsing the tables with `llama_parse`

* Notebook `split_tables.ipynb` was used to split the parsing dataset into batches containing <= 500 samples.
* To perform the parsing run script [parse.py](./parse.py). The output from this script will produce text in `markdown`. 
* To convert it into `.csv` format, use [md2csv.py](./md2csv.py).
