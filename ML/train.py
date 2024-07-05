import os
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

    # add CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--continue_train",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()

    # continue training if set so
    if args.continue_train:
        try:
            run_dir = Path(f"./runs/{sorted(os.listdir('./runs'))[-1]}")
        except:
            print(
                "No `runs` directory found, first train a model from scrath without option `--continue_train`."
            )
        print("Continue training")
        continue_run(run_dir=run_dir)

    else:
        start_run(config_file=cfg_path)



if __name__ == "__main__":
    train()
