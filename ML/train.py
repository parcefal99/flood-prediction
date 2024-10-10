import os
import json
import logging
import argparse
from pathlib import Path

import torch
from dotenv import load_dotenv

# from neuralhydrology.nh_run import start_run, continue_run
from neuralhydrology.nh_run import continue_run
from neuralhydrology.utils.config import Config
from neuralhydrology.training.basetrainer import BaseTrainer

import wandb


LOGGER = logging.getLogger(__name__)


def train():
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
        "--continue_train",
        action="store_true",
        help="specifies to continue train from the last epoch",
        default=False,
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
    args = parser.parse_args()

    if args.gpu not in gpus:
        raise Exception(
            f"Specified prohibited gpu id: `{args.gpu}`, allowed gpu ids are: `{gpus}`"
        )

    # path to the config file
    cfg_path = Path(f"./conf/{args.model}.yml")
    # check if model config exists
    if not cfg_path.exists():
        raise Exception(
            f"No config file found for specified model `{args.model}`. Consider adding a config for this model or use one of available models, see --help."
        )

    cfg = Config(cfg_path)

    if args.continue_train:
        # continue training from previous epoch
        try:
            run_dir = Path(f"./runs/{sorted(os.listdir('./runs'))[-1]}")
        except:
            print(
                "No `runs` directory found, first train a model from scrath without option `--continue_train`."
            )
        print("Continue training")
        continue_run(run_dir=run_dir, gpu=args.gpu)

    else:
        # training from scratch
        start_run(
            config_file=cfg_path, gpu=args.gpu, wandb_entity=os.getenv("WANDB_ENTITY")
        )


def start_run(config_file: Path, wandb_entity: str, gpu: int = None):
    """Start training a model.

    Parameters
    ----------
    config_file : Path
        Path to a configuration file (.yml), defining the settings for the specific run.
    gpu : int, optional
        GPU id to use. Will override config argument 'device'. A value smaller than zero indicates CPU.

    """

    config = Config(config_file)

    # check if a GPU has been specified as command line argument. If yes, overwrite config
    if gpu is not None and gpu >= 0:
        config.device = f"cuda:{gpu}"
    if gpu is not None and gpu < 0:
        config.device = "cpu"

    start_training(config, wandb_entity)


def start_training(cfg: Config, wandb_entity: str):
    """Start model training.

    Parameters
    ----------
    cfg : Config
        The run configuration.

    """
    # MC-LSTM is a special case, where the head returns an empty string but the model is trained as regression model.
    if cfg.head.lower() in ["regression", "gmm", "umal", "cmal", ""]:
        trainer = CustomTrainer(cfg=cfg)
    else:
        raise ValueError(f"Unknown head {cfg.head}.")
    trainer.initialize_training()
    trainer.custom_train_and_validate(wandb_entity)


class CustomTrainer(BaseTrainer):

    def __init__(self, cfg: Config):
        super(CustomTrainer, self).__init__(cfg)

    def custom_train_and_validate(self, wandb_entity: str):
        """Train and validate the model.

        Train the model for the number of epochs specified in the run configuration, and perform validation after every
        ``validate_every`` epochs. Model and optimizer state are saved after every ``save_weights_every`` epochs.
        """

        # setup wandb
        run = wandb.init(
            project=self.cfg.experiment_name,
            entity=wandb_entity,
            config=self.cfg.as_dict(),
        )

        wandb_run = {
            "id": run.id,
            "name": run.name,
        }

        with open(Path(self.cfg.run_dir) / "wandb_run.json", "w") as f:
            json.dump(wandb_run, f, ensure_ascii=False, indent=4)

        run.watch(self.model)

        for epoch in range(self._epoch + 1, self._epoch + self.cfg.epochs + 1):
            if epoch in self.cfg.learning_rate.keys():
                LOGGER.info(f"Setting learning rate to {self.cfg.learning_rate[epoch]}")
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.cfg.learning_rate[epoch]

            self._train_epoch(epoch=epoch)
            avg_losses = self.experiment_logger.summarise()

            # wandb log training
            run.log({"training": avg_losses})

            loss_str = ", ".join(f"{k}: {v:.5f}" for k, v in avg_losses.items())
            LOGGER.info(f"Epoch {epoch} average loss: {loss_str}")

            if epoch % self.cfg.save_weights_every == 0:
                self._save_weights_and_optimizer(epoch)

            if (self.validator is not None) and (epoch % self.cfg.validate_every == 0):
                self.validator.evaluate(
                    epoch=epoch,
                    save_results=self.cfg.save_validation_results,
                    save_all_output=self.cfg.save_all_output,
                    metrics=self.cfg.metrics,
                    model=self.model,
                    experiment_logger=self.experiment_logger.valid(),
                )

                valid_metrics = self.experiment_logger.summarise()

                # wandb log validation
                run.log({"validation": valid_metrics})

                print_msg = f"Epoch {epoch} average validation loss: {valid_metrics['avg_total_loss']:.5f}"
                if self.cfg.metrics:
                    print_msg += f" -- Median validation metrics: "
                    print_msg += ", ".join(
                        f"{k}: {v:.5f}"
                        for k, v in valid_metrics.items()
                        if k != "avg_total_loss"
                    )
                    LOGGER.info(print_msg)

        run.finish()

        # make sure to close tensorboard to avoid losing the last epoch
        if self.cfg.log_tensorboard:
            self.experiment_logger.stop_tb()


if __name__ == "__main__":
    train()
