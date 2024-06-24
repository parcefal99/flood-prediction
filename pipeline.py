import hydra
from omegaconf import DictConfig


@hydra.main(config_name="config", config_path="conf", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main Pipeline function
    """

    # get data from GEE
    # parse kazhydromet
    # get static attributes


if __name__ == "__main__":
    main()
