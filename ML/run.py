import os
import json
import logging
import argparse
from pathlib import Path

import torch
from dotenv import load_dotenv

from neuralhydrology.nh_run import continue_run
from neuralhydrology.utils.config import Config

from eval import evaluate
from nh_run import start_run


LOGGER = logging.getLogger(__name__)


def run():
    
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

    # add CLI arguments
    parser = argparse.ArgumentParser(
        prog="train.py",
        description="This program trains the specified model.",
        epilog="Note, all the flags are optional, see the options to understand the default values.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--gpu",
        type=int,
        help="gpu id",
        default=gpus[0],
        required=False,
    )
    parser.add_argument(
        "--model",
        type=str,
        help="model to use",
        default="lstm",
        choices=["lstm", "arlstm", "ealstm", "transformer", "mamba"],
        required=False,
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="specifies to NOT log training results to wandb",
        default=False,
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="specifies to NOT evaluate the model",
        default=False,
    )

    args = parser.parse_args()

    # path to the config file
    config_file = Path(f"./conf/{args.model}.yaml")
    # check if model config exists
    if not config_file.exists():
        raise Exception(
            f"No config file found for specified model `{args.model}`. Consider adding a config for this model or use one of available models, see --help."
        )

    cfg = Config(config_file)

    if args.gpu not in gpus:
        raise Exception(
            f"Specified prohibited gpu id: `{cfg.gpu}`, allowed gpu ids are: `{gpus}`"
        )

    start_run(
        config=cfg,
        gpu=args.gpu,
        wandb_log=(not args.no_wandb),
        wandb_entity=os.getenv("WANDB_ENTITY"),
    )

    # run evaluations
    periods: list[str] = ["train", "validation", "test"]

    evaluate(
        run_dir=cfg.run_dir,
        epoch=cfg.epochs,
        gpu=args.gpu,
        periods=periods,
        wandb_log=(not args.no_wandb),
        wandb_entity=os.getenv("WANDB_ENTITY"),
    )


if __name__ == "__main__":
    run()
