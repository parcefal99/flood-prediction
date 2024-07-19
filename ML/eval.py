import os
import json
import argparse
from pathlib import Path

import pandas as pd

from neuralhydrology.nh_run import eval_run


def main() -> None:
    gpus = load_allowed_gpus()

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
        "--train",
        action="store_true",
        help="evaulate on training period",
        default=True,
    )
    parser.add_argument(
        "--validation",
        action="store_true",
        help="evaulate on validation period",
        default=True,
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="evaulate on test period",
        default=True,
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

    periods: list[str] = []
    if args.train:
        periods.append("train")
    if args.validation:
        periods.append("validation")
    if args.test:
        periods.append("test")

    evaluate(run_dir, epoch=args.epoch, gpu=args.gpu, periods=periods)


def evaluate(run_dir: Path, epoch: int, gpu: int, periods: list[str]) -> None:

    epoch = str(epoch)
    if len(epoch) == 1:
        epoch_str = "00" + epoch
    elif len(epoch) == 2:
        epoch_str = "0" + epoch

    df = None
    for i, period in enumerate(periods):
        print(f"Evaluation on {period} period.")
        eval_run(run_dir, period=period, epoch=epoch, gpu=gpu)
        df_period = eval_results(run_dir, period, epoch=epoch_str)
        df_period = df_period.rename(
            columns={"NSE": f"NSE_{period}", "KGE": f"KGE_{period}"}
        )
        if i == 0:
            df = df_period
        else:
            df = pd.merge(df, df_period, left_index=True, right_index=True)

    # save metrics
    df.to_csv(f"{str(run_dir)}/eval.csv")


def load_allowed_gpus() -> list[int]:
    # load allowed GPU ids
    f = open("gpu.json")
    gpus = json.load(f)
    f.close()
    return gpus


def eval_results(run_dir: Path, period: str, epoch: str = "010") -> pd.DataFrame:
    df = pd.read_csv(
        run_dir / f"{period}" / f"model_epoch{epoch}" / f"{period}_metrics.csv",
        dtype={"basin": str},
    )
    df = df.set_index("basin")

    # Compute the median NSE from all basins, where discharge observations are available for that period
    print(f"Median NSE of the {period} period {df['NSE'].median():.3f}")
    print(f"Mean NSE of the {period} period {df['NSE'].mean():.3f}")

    return df


if __name__ == "__main__":
    main()
