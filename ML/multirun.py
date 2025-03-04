import os
import logging

import torch

from dotenv import load_dotenv

import hydra
import hydra.core
import hydra.core.hydra_config
from omegaconf import DictConfig, OmegaConf

from neuralhydrology.utils.config import Config

from eval import evaluate
from nh_run import start_run


LOGGER = logging.getLogger(__name__)


@hydra.main(config_name=None, config_path="conf/models", version_base=None)
def run(cfg: DictConfig) -> None:

    load_dotenv(dotenv_path="../.env")

    # check if CUDA available
    if torch.cuda.is_available():
        pass
    else:
        raise Exception("No GPU found!")

    cfg = OmegaConf.to_object(cfg)

    gpu = cfg.pop("gpu")
    wandb_log = cfg.pop("wandb")
    is_discharge = cfg.pop("discharge")

    # load dataset config
    dataset = cfg.pop("dataset")
    dataset = OmegaConf.load(f"./conf/datasets/{dataset}.yaml")
    # load base config
    base = OmegaConf.load("./conf/base.yaml")

    # get sweeper parameters
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    sweeper_params = hydra_cfg.sweeper.params

    current_params = {}
    for p in sweeper_params:
        current_params[p] = cfg[p]

    print(f"Running job with the following parameters: {current_params}")

    cfg = OmegaConf.merge(cfg, base, dataset)
    cfg = OmegaConf.to_object(cfg)

    if is_discharge:
        cfg["dynamic_inputs"].append("discharge_shift1")

    cfg = Config(cfg)

    # training with validations
    start_run(
        config=cfg,
        gpu=gpu,
        wandb_log=wandb_log,
        wandb_entity=os.getenv("WANDB_ENTITY"),
    )

    # run evaluations
    periods: list[str] = ["train", "validation", "test"]

    # final evaluation over the specified periods
    evaluate(
        run_dir=cfg.run_dir,
        epoch=cfg.epochs,
        gpu=gpu,
        periods=periods,
        wandb_log=wandb_log,
        wandb_entity=os.getenv("WANDB_ENTITY"),
    )


if __name__ == "__main__":
    run()
