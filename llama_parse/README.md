# PDF Annual Yearbooks Parsing

## Setup

1. Add `LLAMA_CLOUD_API_KEY` entry inside `.env` file in the project's root. Create it if it doesn't exist.
2. Download the preprocessed [annual reports](https://drive.google.com/drive/folders/1Osj-ZF_QueGmAGcpoyhYo0D7Wlt8DLO9) and place them into the `data` of root folder: `/data/annual_reports`.

## Tables Extraction

To preprocess the tables (`annual_reports`), data from [KazHydroMet Portal for Annual Reports](https://www.kazhydromet.kz/ru/gidrologiya/ezhegodnye-dannye-o-rezhime-i-resursah-poverhnostnyh-vod-sushi-eds) was downloaded first. Then [pdf_reports_parse.py](./pdf_reports_parse.py) script was used to extract the tables with streamflow data for the selected basins from the PDF contents. This script was also responsible for table cropping.


## Parsing the tables with `llama_parse`

* Notebook `split_tables.ipynb` was used to split the parsing dataset into batches containing <= 500 samples.
* To perform the parsing run script [parse.py](./parse.py). The output from this script will produce text in `markdown`. If some documents will not be parsed due to errors, write the names of such docs into a file and place it inside `./data/annual_reports/batch_split`, then run [parse2.py](./parse2.py) script with the flag `-i` and the path to a file containing the not parsed docs.
* To convert markdown files into `.csv` format, use [md2csv.py](./md2csv.py) script.
