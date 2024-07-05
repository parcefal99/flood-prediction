import os
import json
import argparse
from pathlib import Path

import torch
from neuralhydrology.nh_run import start_run, continue_run


def train():
    # path to the config file
    cfg_path = Path("./conf/lstm.yml")

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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--continue_train",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--gpu", type=int, help="gpu id", default=gpus[0], required=False
    )
    args = parser.parse_args()

    if args.gpu not in gpus:
        raise Exception(
            f"Specified prohibited gpu id: `{args.gpu}`, allowed gpu ids are: `{gpus}`"
        )

    # continue training if set so
    if args.continue_train:
        try:
            run_dir = Path(f"./runs/{sorted(os.listdir('./runs'))[-1]}")
        except:
            print(
                "No `runs` directory found, first train a model from scrath without option `--continue_train`."
            )
        print("Continue training")
        continue_run(run_dir=run_dir, gpu=args.gpu)

    else:
        start_run(config_file=cfg_path, gpu=args.gpu)


if __name__ == "__main__":
    train()
