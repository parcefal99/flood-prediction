import os
import json
import argparse
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
import wandb.plot
import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import trange
from dotenv import load_dotenv

import wandb

from neuralhydrology.modelzoo import get_model
from neuralhydrology.datasetzoo import get_dataset
from neuralhydrology.evaluation import metrics
from neuralhydrology.utils.config import Config
from neuralhydrology.datautils.utils import load_scaler


sns.set_style("whitegrid")


def main() -> None:
    load_dotenv("./.env")

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
    parser.add_argument(
        "--period",
        type=str,
        default="test",
        choices=["validation", "test"],
        help="period for hindcast",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="specifies to log training results to wandb",
        default=False,
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

    plots_period_path: Path = run_dir / "evaluation" / "plots_log" / args.period
    plots_period_path.mkdir(parents=True, exist_ok=True)
    timeseries_period_path: Path = (
        run_dir / "evaluation" / "timeseries_log" / args.period
    )
    timeseries_period_path.mkdir(parents=True, exist_ok=True)

    # get basins
    basins_path = Path("./basins.txt")
    basins = pd.read_csv(basins_path, header=None)[0].tolist()
    evaluate_basins(
        run_dir,
        basins,
        epoch=args.epoch,
        period=args.period,
        gpu=args.gpu,
        wandb_log=args.wandb,
        wandb_entity=os.getenv("WANDB_ENTITY"),
    )


def evaluate_basins(
    run_dir: Path,
    basins: list,
    epoch: int,
    period: str,
    gpu: int,
    wandb_log: bool,
    wandb_entity: str,
) -> None:
    """Saves plots of observed vs predicted data for specified basins"""

    if wandb_log:
        wandb_path = Path(Path(run_dir) / "wandb_run.json")
        if not wandb_path.exists():
            raise Exception(
                f"No wandb file found for specified run directory! Probably you have started training without `--wandb` option!"
            )

        with open(Path(run_dir) / "wandb_run.json", "r") as f:
            wandb_run = json.load(f)

    cfg_path = Path(Path(run_dir) / "config.yml")
    # check if model config exists
    if not cfg_path.exists():
        raise Exception(f"No config file found for specified run directory!")

    cfg = Config(cfg_path)

    if wandb_log:
        # setup wandb
        run = wandb.init(
            id=wandb_run["id"],
            name=wandb_run["name"],
            project=cfg.experiment_name,
            entity=wandb_entity,
            config=cfg.as_dict(),
        )

    cfg = get_cfg(run_dir)
    cfg.update_config({"device": f"cuda:{gpu}"})

    model = load_model(cfg, run_dir, epoch=epoch)
    model = model.eval()
    mean, std = get_scaler_vals(run_dir)

    basins_iter = iter(basins)

    nse_basins = np.empty(len(basins), dtype=np.float32)

    t = trange(len(basins))
    for i in t:
        basin = next(basins_iter)
        t.set_description(f"Basin {basin}")

        loader = get_basin_data(cfg, run_dir, basin_id=basin, period=period)
        y_hat, y, year_start, year_end = get_cmp(cfg, model, loader, basin, mean, std)

        if wandb_log:
            run.log(
                {
                    f"hindcast/{basin}": wandb.plot.line_series(
                        xs=range(len(y)),
                        ys=[y, y_hat],
                        keys=["Observed", "Predicted"],
                        xname="Days",
                        title=basin,
                    )
                }
            )

        nse = compute_nse(y_hat, y)
        nse_basins[i] = nse
        t.set_postfix({"nse": nse})

        plot_cmp(
            run_dir,
            basin_id=basin,
            y_hat=y_hat,
            y=y,
            period=period,
            year_start=year_start,
            year_end=year_end,
        )

        if i == len(basins) - 1:
            t.set_description("Done")

    if wandb_log:
        run.finish()

    nse_basins = pd.DataFrame({"NSE": nse_basins})
    nse_basins_stats = pd.DataFrame(
        {
            "NSE_mean": [nse_basins.mean()],
            "NSE_median": [np.median(nse_basins[np.isfinite(nse_basins)])],
            "NSE_min": [nse_basins.min()],
            "NSE_max": [nse_basins.max()],
        }
    )
    nse_basins.to_csv(run_dir / "evaluation" / "nse_metrics.csv", index=False)
    nse_basins_stats.to_csv(
        run_dir / "evaluation" / "nse_metrics_stats.csv", index=False
    )


def compute_nse(y_hat: np.ndarray, y: np.ndarray) -> float:
    obs = xr.DataArray(data=y, name="discharge_obs")
    sim = xr.DataArray(data=y_hat, name="discharge_sim")
    return metrics.nse(obs, sim)


def get_cfg(run_dir: Path, update_dict: Optional[dict] = None) -> Config:
    cfg = Config(run_dir / "config.yml")
    if update_dict is not None:
        cfg.update_config(update_dict)
    return cfg


def get_basin_data(
    cfg: Config, run_dir: Path, basin_id: str, period: str
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
    cfg: Config,
    model: nn.Module,
    loader: DataLoader,
    basin_id: int,
    mean: float,
    std: float,
) -> tuple[list, list, str]:
    """Get predicted and observed data"""

    predictions = []
    observations = []

    year_start = None
    year_end = None

    for i, data in enumerate(loader):
        # wait for one year to pass (to have data for previous year)
        if i < cfg.seq_length:
            continue
        # stop after the second year
        # if i == cfg.seq_length * 2:
        #     break

        if i == cfg.seq_length:
            year_start = str(data["date"][0][-1]).split("-")[0]
        elif i == len(loader) - 1:
            year_end = str(data["date"][0][-1]).split("-")[0]

        for key in data.keys():
            if not key.startswith("date"):
                data[key] = data[key].to(cfg.device)

        x = model.pre_model_hook(data, is_train=False)
        y = x["y"].detach().cpu().numpy()[0][-1][0]
        # save observed data
        observations.append(y)
        del x["y"]

        # change observed data on predicted data
        # it changes as many values as there are in the `predicted` array
        if i > cfg.seq_length:
            x["x_d"][0][-len(predictions) :][:, -1] = torch.tensor(predictions)[
                -cfg.seq_length :
            ]

        # obtain and save predictions
        prediction = model(x)
        pred = prediction["y_hat"].detach().cpu().numpy()[0][-1][0]

        if np.isnan(pred) and i > cfg.seq_length:
            pred = predictions[-1]
        predictions.append(pred)

    # denormalize predictions
    predictions = np.array(predictions) * std + mean
    observations = np.array(observations) * std + mean
    return predictions, observations, year_start, year_end


def plot_cmp(
    run_dir: Path,
    basin_id: str,
    y_hat: list,
    y: list,
    period: str,
    year_start: str,
    year_end: str,
) -> None:
    """Plot and save the comparison between observed vs predicted data"""
    df = pd.DataFrame(
        {
            "Day": [i for i in range(1, len(y) + 1)],
            "Observed": y,
            "Predicted": y_hat,
        }
    )
    df.to_csv(
        run_dir / f"evaluation/timeseries_log/{period}/{basin_id}.csv",
        sep=";",
        index=False,
    )
    ax = sns.lineplot(
        data=df, x="Day", y="Observed", linestyle="solid", label="Observed"
    )
    sns.lineplot(
        ax=ax, data=df, x="Day", y="Predicted", linestyle="solid", label="Predicted"
    )
    ax.set_title(f"{basin_id}, {period} period ({str(int(year_start)+1)}-{year_end})")
    ax.set_ylabel("Discharge")
    plt.savefig(run_dir / f"evaluation/plots_log/{period}/{basin_id}.png")
    plt.close()


if __name__ == "__main__":
    main()
