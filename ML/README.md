# Machine Learning Models Training

## Structure

- `conf` contains the configuration files for specific models. The attributes were taken from [here](https://neuralhydrology.readthedocs.io/en/latest/usage/config.html).
- `eval.ipynb` is an evaluation notebook for models
- `basins.txt` contains a list of basins on which to train models
- `finetune_basin.txt` contains a list of bains on which to finetune models
- `gpu.json` presents a list of gpus allowed to use
- `train.py` training script

## Training

To see info about the program and available options execute the script with `-h` (or `--help`) flag:

```bash
python train.py -h
```

To run the training with default options execute:

```bash
python train.py
```

Note it is recommended to see `--help` since it contains important information regarding the program.
