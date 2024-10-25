import json
import logging
from pathlib import Path

from neuralhydrology.utils.config import Config
from neuralhydrology.training.basetrainer import BaseTrainer

import wandb


LOGGER = logging.getLogger(__name__)


def start_run(config: Config, wandb_log: bool, wandb_entity: str, gpu: int = None):
    """Start training a model.

    Parameters
    ----------
    config_file : Path
        Path to a configuration file (.yml), defining the settings for the specific run.
    gpu : int, optional
        GPU id to use. Will override config argument 'device'. A value smaller than zero indicates CPU.

    """

    # check if a GPU has been specified as command line argument. If yes, overwrite config
    if gpu is not None and gpu >= 0:
        config.device = f"cuda:{gpu}"
    if gpu is not None and gpu < 0:
        config.device = "cpu"

    start_training(config, wandb_log, wandb_entity)


def finetune(config_base: Config, config_finetune: Config, wandb_log: bool, wandb_entity: str, gpu: int = None):
    """Finetune a pre-trained model.

    Parameters
    ----------
    config_file : Path, optional
        Path to an additional config file. Each config argument in this file will overwrite the original run config.
        The config file for finetuning must contain the argument `base_run_dir`, pointing to the folder of the 
        pre-trained model, as well as 'finetune_modules' to indicate which model parts will be trained during
        fine-tuning.
    gpu : int, optional
        GPU id to use. Will override config argument 'device'. A value smaller than zero indicates CPU.

    """
    # load finetune config and check for a non-empty list of finetune_modules
    if not config_finetune.finetune_modules:
        raise ValueError("For finetuning, at least one model part has to be specified by 'finetune_modules'.")

    # extract base run dir, load base run config and combine with the finetune arguments
    config_base.update_config({'run_dir': None, 'experiment_name': None})
    config_base.update_config(config_finetune.as_dict())
    config_base.is_finetuning = True

    # if the base run was a continue_training run, we need to override the continue_training flag from its config.
    config_base.is_continue_training = False

    # check if a GPU has been specified as command line argument. If yes, overwrite config
    if gpu is not None and gpu >= 0:
        config_base.device = f"cuda:{gpu}"
    if gpu is not None and gpu < 0:
        config_base.device = "cpu"

    start_training(config_base, wandb_log, wandb_entity)


def start_training(cfg: Config, wandb_log: bool, wandb_entity: str):
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
    trainer.custom_train_and_validate(wandb_log, wandb_entity)


class CustomTrainer(BaseTrainer):

    def __init__(self, cfg: Config):
        super(CustomTrainer, self).__init__(cfg)

    def custom_train_and_validate(self, wandb_log: bool, wandb_entity: str):
        """Train and validate the model.

        Train the model for the number of epochs specified in the run configuration, and perform validation after every
        ``validate_every`` epochs. Model and optimizer state are saved after every ``save_weights_every`` epochs.
        """

        if wandb_log:
            is_discharge = (
                "Discharge"
                if "discharge_shift1" in self.cfg.dynamic_inputs
                else "No Discharge"
            )

            # setup wandb
            run = wandb.init(
                project=self.cfg.experiment_name,
                entity=wandb_entity,
                config=self.cfg.as_dict(),
                tags=[is_discharge],
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

            if wandb_log:
                # wandb log training
                run.log({"training": avg_losses, "epoch": epoch})

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

                if wandb_log:
                    # wandb log validation
                    run.log({"validation": valid_metrics, "epoch": epoch})

                print_msg = f"Epoch {epoch} average validation loss: {valid_metrics['avg_total_loss']:.5f}"
                if self.cfg.metrics:
                    print_msg += f" -- Median validation metrics: "
                    print_msg += ", ".join(
                        f"{k}: {v:.5f}"
                        for k, v in valid_metrics.items()
                        if k != "avg_total_loss"
                    )
                    LOGGER.info(print_msg)

        if wandb_log:
            run.finish()

        # make sure to close tensorboard to avoid losing the last epoch
        if self.cfg.log_tensorboard:
            self.experiment_logger.stop_tb()
