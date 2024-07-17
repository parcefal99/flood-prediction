import json
import argparse
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import trange

from neuralhydrology.modelzoo import get_model
from neuralhydrology.utils.config import Config
from neuralhydrology.datautils.utils import load_scaler
from neuralhydrology.datasetzoo import BaseDataset, get_dataset


sns.set_style("whitegrid")


def main() -> None:
    # check if CUDA available
    if torch.cuda.is_available():
        pass
    else:
        raise Exception("No GPU found!")

    # load allowed GPU ids
    f = open("gpu.json")
    gpus = json.load(f)
    f.close()

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="run directory containing the model",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        required=True,
        help="model epoch to use",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        help="gpu id",
        default=gpus[0],
        required=False,
    )

    args = parser.parse_args()

    if args.gpu not in gpus:
        raise Exception(
            f"Specified prohibited gpu id: `{args.gpu}`, allowed gpu ids are: `{gpus}`"
        )

    run_dir = Path(args.run_dir)
    # check whether run_dir exists
    if not run_dir.exists():
        raise Exception(f"Specified run_dir does not exists: `{run_dir}`")

    # get basins
    basins_path = Path("./basins.txt")
    basins = pd.read_csv(basins_path, header=None)[0].tolist()
    evaluate_basins(run_dir, basins, epoch=args.epoch, gpu=args.gpu)


def evaluate_basins(run_dir: Path, basins: list, epoch: int, gpu: int) -> None:
    """Saves plots of observed vs predicted data for specified basins"""

    cfg = get_cfg(run_dir)
    cfg.update_config({"device": f"cuda:{gpu}"})

    model = load_model(cfg, run_dir, epoch=epoch)
    mean, std = get_scaler_vals(run_dir)

    basins_iter = iter(basins)

    t = trange(len(basins))
    for i in t:
        basin = next(basins_iter)
        t.set_description(f"Basin {basin}")

        loader = get_basin_data(cfg, run_dir, basin_id=basin, period="test")
        y_hat, y, year = get_cmp(cfg, model, loader, mean, std)

        plot_cmp(run_dir, basin_id=basin, y_hat=y_hat, y=y, year=year)

        if i == len(basins) - 1:
            t.set_description("Done")


def get_cfg(run_dir: Path, update_dict: Optional[dict] = None) -> Config:
    cfg = Config(run_dir / "config.yml")
    if update_dict is not None:
        cfg.update_config(update_dict)
    return cfg


def get_basin_data(
    cfg: Config, run_dir: Path, basin_id: str, period: str = "test"
) -> DataLoader:
    ds = get_dataset(
        cfg=cfg,
        is_train=False,
        basin=str(basin_id),
        period=period,
        scaler=load_scaler(run_dir),
    )
    loader = DataLoader(ds, batch_size=1, num_workers=0, collate_fn=ds.collate_fn)
    return loader


def load_model(cfg: Config, run_dir: Path, epoch: int = 20) -> nn.Module:
    """Loads the model from specified `run_dir` trained on `epochs`"""
    model = get_model(cfg)
    # convert epoch to the required format
    epoch = str(epoch)
    if len(epoch) == 1:
        epoch = "00" + epoch
    elif len(epoch) == 2:
        epoch = "0" + epoch
    # load model parameters
    model.load_state_dict(torch.load(run_dir / f"model_epoch{epoch}.pt"))
    model = model.to(cfg.device)
    return model


def get_scaler_vals(run_dir: Path) -> tuple[float, float]:
    scaler = load_scaler(run_dir)
    mean: np.ndarray = scaler["xarray_feature_center"]["discharge"].data.tolist()
    std: np.ndarray = scaler["xarray_feature_scale"]["discharge"].data.tolist()
    return mean, std


def get_cmp(
    cfg: Config, model: nn.Module, loader: DataLoader, mean: float, std: float
) -> tuple[list, list, str]:
    """Get predicted and observed data"""

    predictions = []
    actual = []

    year = None

    for i, data in enumerate(loader):
        # wait for one year to pass (to have data for previous year)
        if i < 365:
            continue
        # stop after the second year
        if i == 365 * 2:
            break

        if i == 365:
            year = str(data["date"][0][-1]).split("-")[0]

        for key in data.keys():
            if not key.startswith("date"):
                data[key] = data[key].to(cfg.device)

        x = model.pre_model_hook(data, is_train=False)
        y = x["y"].detach().cpu().numpy()[0][-1][0]
        # denormalize observed data
        y = y * std + mean
        # save observed data
        actual.append(y)
        del x["y"]

        # change observed data on predicted data
        # it changes as many values as there are in the `predicted` array
        if i > 365:
            x["x_d"][0][-len(predictions) :][:, 8] = torch.tensor(predictions)

        # obtain and save predictions
        prediction = model(x)
        predictions.append(prediction["y_hat"].detach().cpu().numpy()[0][-1][0])

    # denormalize predictions
    predictions = np.array(predictions) * std + mean
    return predictions, actual, year


def plot_cmp(run_dir: Path, basin_id: str, y_hat: list, y: list, year: str) -> None:
    """Plot and save the comparison between observed vs predicted data"""
    df = pd.DataFrame(
        {
            "Day": [i for i in range(1, len(y) + 1)],
            "Observed": y,
            "Predicted": y_hat,
        }
    )
    ax = sns.lineplot(
        data=df, x="Day", y="Observed", linestyle="solid", label="Observed"
    )
    sns.lineplot(
        ax=ax, data=df, x="Day", y="Predicted", linestyle="solid", label="Predicted"
    )
    ax.set_title(f"{basin_id}, year {year}")
    ax.set_ylabel("Discharge")
    plt.savefig(run_dir / f"img_log/{basin_id}_test_year_{year}.png")
    plt.close()


if __name__ == "__main__":
    main()
