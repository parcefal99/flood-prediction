import os
import json
import logging
import argparse
from pathlib import Path

import hydra.core
import hydra.core.hydra_config
import torch
from dotenv import load_dotenv

import hydra
from omegaconf import DictConfig, OmegaConf

from neuralhydrology.nh_run import continue_run
from neuralhydrology.utils.config import Config

from eval import evaluate
from nh_run import start_run


LOGGER = logging.getLogger(__name__)


@hydra.main(config_name=None, config_path="conf", version_base=None)
def run(cfg: DictConfig):

    load_dotenv(dotenv_path=".env")

    # check if CUDA available
    if torch.cuda.is_available():
        pass
    else:
        raise Exception("No GPU found!")

    # load allowed GPU ids
    f = open("gpu.json")
    gpus = json.load(f)
    f.close()

    cfg = OmegaConf.to_object(cfg)

    gpu = cfg.pop("gpu")
    wandb_log = cfg.pop("wandb")

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    sweeper_params = hydra_cfg.sweeper.params

    current_params = {}
    for p in sweeper_params:
        current_params[p] = cfg[p]

    print(f"Running job with the following parameters: {current_params}")
        
    cfg = Config(cfg)

    if gpu not in gpus:
        raise Exception(
            f"Specified prohibited gpu id: `{gpu}`, allowed gpu ids are: `{gpus}`"
        )

    start_run(
        config=cfg,
        gpu=gpu,
        wandb_log=wandb_log,
        wandb_entity=os.getenv("WANDB_ENTITY"),
    )

    # run evaluations
    periods: list[str] = ["train", "validation", "test"]

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