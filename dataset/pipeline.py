import logging

import hydra
from omegaconf import DictConfig

import utils
from attributes_calc import process_attributes


@hydra.main(config_name="config", config_path="conf", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main Pipeline function
    """

    log = logging.getLogger(__name__)

    # create project directories
    utils.make_dirs(cfg)

    # get data from GEE
    # parse kazhydromet
    # get static attributes
    process_attributes(cfg, log)


if __name__ == "__main__":
    main()
