import os
import json
import logging
import argparse
import datetime
import itertools
from pathlib import Path

import torch
from dotenv import load_dotenv

from config import Config
from eval import evaluate
from nh_run import finetune


LOGGER = logging.getLogger(__name__)

def start_finetune():
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
        prog="finetune.py",
        description="This program finetunes the specified model.",
        epilog="Note, all the flags are optional, see the options to understand the default values.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--gpu",
        type=int,
        help="gpu id",
        default=gpus[0],
        required=True,
    )
    parser.add_argument(
        "--pretrained-dir",
        type=str,
        help="directory of the pretrained model",
        required=True,
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

    if args.gpu not in gpus:
        raise Exception(
            f"Specified prohibited gpu id: `{args.gpu}`, allowed gpu ids are: `{gpus}`"
        )

    run_dir = Path(f"./runs/{args.pretrained_dir}")
    config = Config(run_dir / "config.yml")
    finetune_config = Config(Path("./conf/finetune.yml"))
    finetune_config.base_run_dir = run_dir.absolute()

    if config.model not in ['cudalstm', 'ealstm']:
        raise Exception(
            f"Finetuning for the pretrained model '{config.model}' is not covered (yet)."
        )

    if config.model == 'cudalstm':
        modules = ['head', 'lstm']
    elif config.model == 'ealstm':
        modules = ['head', 'input_gate', 'dynamic_gates']

    module_combinations = []

    if finetune_config.finetune_modules:
        module_combinations.append(finetune_config.finetune_modules)
    else:
        for l in range(1, len(modules) + 1):
            for subset in itertools.combinations(modules, l):
                module_combinations.append(list(subset))

    for subset in module_combinations:
        finetune_config.finetune_modules = subset

        finetune(
            config_base=config,
            config_finetune=finetune_config,
            gpu=args.gpu,
            wandb_log=(not args.no_wandb),
            wandb_entity=os.getenv("WANDB_ENTITY"),
        )

        # run evaluations
        periods: list[str] = ["train", "validation", "test"]

        if not args.no_eval:
            evaluate(
                run_dir=config.run_dir,
                epoch=finetune_config.epochs,
                gpu=args.gpu,
                periods=periods,
                wandb_log=(not args.no_wandb),
                wandb_entity=os.getenv("WANDB_ENTITY"),
            )


if __name__ == "__main__":
    start_finetune()
