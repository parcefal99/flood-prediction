# Machine Learning Models Training

## Structure

- `conf` contains the configuration files for specific models. The attributes were taken from [here](https://neuralhydrology.readthedocs.io/en/latest/usage/config.html).
- `eval.ipynb` is an evaluation notebook for models
- `basins.txt` contains a list of basins on which to train models
- `finetune_basin.txt` contains a list of bains on which to finetune models
- `gpu.json` presents a list of gpus allowed to use
- `train.py` training script

## Training

To run the training execute:

```bash
python train.py
```

Options:

- gpu <gpu_id> (take from allowed gpus list at `gpu.json`)
- continue_training 