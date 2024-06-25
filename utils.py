from pathlib import Path
from omegaconf import DictConfig


def make_dirs(cfg: DictConfig) -> None:
    """Creates all the necessary directories for the dataset"""

    # create directory for dataset
    dataset_path = Path(cfg.dataset.path)
    if not dataset_path.exists():
        dataset_path.mkdir()

    # create directory for merged station forcings if not exists
    forcing_path = dataset_path / cfg.dataset.forcing
    if not forcing_path.exists():
        forcing_path.mkdir()

    # create directory for time_series output if not exists
    time_series_path = dataset_path / cfg.dataset.time_series
    if not time_series_path.exists():
        time_series_path.mkdir()

    # create directory for time_series output if not exists
    attributes_path = dataset_path / cfg.dataset.catchment_attributes.path
    if not attributes_path.exists():
        attributes_path.mkdir()
