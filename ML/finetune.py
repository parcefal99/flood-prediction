import os
import json
import logging
import argparse
import datetime
import glob
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
        "--dataset",
        type=str,
        help="dataset to use",
        required=True,
    )
    parser.add_argument(
        "--pretrained-dir",
        type=str,
        help="directory of the pretrained model",
        required=True,
    )
    parser.add_argument(
        "--period",
        type=int,
        help="period to use",
        required=True,
    )
    parser.add_argument(
        "--learning-rate",
        type=int,
        help="lr to use",
        required=True,
    )
    parser.add_argument(
        "--name",
        type=str,
        help="name of the experiment",
        required=True,
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="seed to use",
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
    finetune_config.experiment_name = f"{args.name}_{args.dataset}_41"
    finetune_config.data_dir = Path(f"../data/{args.dataset}_dataset")
    finetune_config.epochs = 20
    finetune_config.seed = args.seed

    matching_file = glob.glob(f"basins/basins_sigma_41.txt")[0]
    finetune_config.train_basin_file = Path(matching_file)
    finetune_config.validation_basin_file = Path(matching_file)
    finetune_config.test_basin_file = Path(matching_file)
    finetune_config.validate_n_random_basins = int(matching_file.split('_')[-1].split('.')[0])

    if args.period == 1:
        finetune_config.train_start_date = '01/10/2012'
        finetune_config.train_end_date = '31/12/2022'
        finetune_config.validation_start_date = '01/10/2008'
        finetune_config.validation_end_date = '30/09/2012'
        finetune_config.test_start_date = '01/10/2004'
        finetune_config.test_end_date = '30/09/2008'
    elif args.period == 2:
        finetune_config.train_start_date = '01/10/2011'
        finetune_config.train_end_date = '30/09/2019'
        finetune_config.validation_start_date = '01/10/2008'
        finetune_config.validation_end_date = '30/09/2011'
        finetune_config.test_start_date = '01/10/2019'
        finetune_config.test_end_date = '31/12/2022'

    if args.learning_rate == 1:
        finetune_config.learning_rate = {0: 1e-2, 5: 1e-3, 10: 1e-4}
    elif args.learning_rate == 2:
        finetune_config.learning_rate = {0: 1e-3, 5: 1e-4, 10: 1e-5}
    elif args.learning_rate == 3:
        finetune_config.learning_rate = {0: 1e-3}
    elif args.learning_rate == 4:
        finetune_config.learning_rate = {0: 1e-4}
    elif args.learning_rate == 5:
        finetune_config.learning_rate = {0: 1e-2}

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
