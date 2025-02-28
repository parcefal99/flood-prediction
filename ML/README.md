# Machine Learning Models Training

## Structure

- `conf` contains the configuration files for specific models and datasets as well as the base config, which is common for all the models. The attributes were taken from [here](https://neuralhydrology.readthedocs.io/en/latest/usage/config.html).
- `eval.py` evaluate a model
- `basins` contains files of lists of basins onto which to train models
- `multirun.py` training script that supports Grid Search
- `hindcast.py` script to plot observed vs predicted values

## Training

To run the training execute:

```bash
python multirun.py --config-name <model_name> -m dataset=<dataset_name> discharge=[true|false] gpu=<number> experiment_name=<name_of_experiment>
```

Example for _lstm_ model training on _sigma_ dataset with (previous) discharge (as previous input):

```bash
python multirun.py --config-name lstm -m dataset=sigma discharge=true gpu=0 experiment_name=lstm41_d_seed
```

Note it is recommended to see `--help` since it contains important information regarding the program. Flag `-m` indicates to sequentially run jobs specified by hypdra sweeper in each model file config.


## Evaluation 

After training, if you need to evaluate the model again, you should run:

```bash
python eval.py --run_dir /path/to/run --epoch EPOCH --gpu GPU
```

To know more about the arguments run:

```bash
python eval.py -h
```

This script will produce two files in the specified run directory: `eval.csv` and `eval_stats.csv`. The first one contains NSE and KGE metrics for each basin and the second file contains mean and median computed on the first file.


## Hindcast evaluation

`hindcast.py` plots observed vs predicted data. 

To run:

```bash
python hindcast.py ---run_dir RUN_DIR --epoch EPOCH --gpu GPU_ID
```

To understand the arguments run:

```bash
python hindcast.py -h
```

Currently it feeds meteo data and previous discharge. Meteo data present the observed data (needs to be predicted too in the future), previous discharge for previous 365 days initially composed of observed data too, but in the process of hindcast, the predicted values are added as previous discharge. E.g. if we predict discharge for tomorrow, the discharge for today is fed as previous discharge, for predicting the discharge for the day after tomorrow, we fed the discharge for tomorrow as previous one.

### Output

The images will be saved in the specified `run_dir/img_log`.
