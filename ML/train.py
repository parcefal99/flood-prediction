import os
import argparse
from pathlib import Path

import torch
from neuralhydrology.nh_run import start_run, continue_run


def train():

    cfg_path = Path("./conf/lstm.yml")
    run_dir = Path(f"./runs/{sorted(os.listdir('./runs'))[-1]}")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--continue_train",
        action="store_true",
        default=False,
    )

    args = parser.parse_args()

    if args.continue_train:
        print("Continue training")
        continue_run(run_dir=run_dir)

    else:
        if torch.cuda.is_available():
            start_run(config_file=cfg_path)

        else:
            raise Exception("No GPU found!")


if __name__ == "__main__":
    train()
