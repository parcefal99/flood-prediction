import os
import json
import argparse
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

import wandb

from neuralhydrology.nh_run import eval_run
from neuralhydrology.utils.config import Config


def main() -> None:
    load_dotenv("./.env")

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
        default=0,
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
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="specifies to log training results to wandb",
        default=False,
    )

    args = parser.parse_args()

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

    evaluate(
        run_dir,
        epoch=args.epoch,
        gpu=args.gpu,
        periods=periods,
        wandb_log=args.wandb,
        wandb_entity=os.getenv("WANDB_ENTITY"),
    )


def evaluate(
    run_dir: Path,
    epoch: int,
    gpu: int,
    periods: list[str],
    wandb_log: bool,
    wandb_entity: str,
) -> None:

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
            config=cfg.as_dict(),
            project=cfg.experiment_name,
            entity=wandb_entity,
        )

    epoch = str(epoch)
    if len(epoch) == 1:
        epoch_str = "00" + epoch
    elif len(epoch) == 2:
        epoch_str = "0" + epoch

    df = None
    df_stats = None
    for i, period in enumerate(periods):
        print(f"Evaluation on {period} period.")
        eval_run(run_dir, period=period, epoch=epoch, gpu=gpu)
        df_period, median, mean = eval_results(run_dir, period, epoch=epoch_str)

        if wandb_log:
            run.log(
                {
                    f"{period}/accuracy/NSE_mean": mean,
                    f"{period}/accuracy/NSE_median": median,
                }
            )

        df_period = df_period.rename(
            columns={"NSE": f"NSE_{period}", "KGE": f"KGE_{period}"}
        )
        if i == 0:
            df = df_period
        else:
            df = pd.merge(df, df_period, left_index=True, right_index=True)

    # save metrics
    df.to_csv(f"{str(run_dir)}/eval.csv")
    df_stats = pd.concat([df.median().to_frame().T, df.mean().to_frame().T])
    df_stats = df_stats.set_index([["median", "mean"]])
    df_stats.to_csv(f"{str(run_dir)}/eval_stats.csv", index=True)

    if wandb_log:
        run.finish()


def eval_results(
    run_dir: Path, period: str, epoch: str = "010"
) -> tuple[pd.DataFrame, float, float]:
    df = pd.read_csv(
        run_dir / f"{period}" / f"model_epoch{epoch}" / f"{period}_metrics.csv",
        dtype={"basin": str},
    )
    df = df.set_index("basin")

    median = df["NSE"].median()
    mean = df["NSE"].mean()

    # Compute the median NSE from all basins, where discharge observations are available for that period
    print(f"Median NSE of the {period} period {median:.3f}")
    print(f"Mean NSE of the {period} period {mean:.3f}")

    return df, median, mean


if __name__ == "__main__":
    main()
