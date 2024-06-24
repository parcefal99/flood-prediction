import os
import logging
import pathlib
from functools import reduce

import hydra
from omegaconf import DictConfig

import numpy as np
import pandas as pd
import scipy
from tqdm import trange


@hydra.main(config_name="config", config_path="conf", version_base=None)
def main(cfg: DictConfig) -> None:
    log = logging.getLogger(__name__)

    # create the necessary directories
    make_dirs(cfg)

    # get the list of selected basins
    selected_df = get_selected_basins(cfg.dataset.selected_basins)

    log.info(f"Processing stations")
    # calculate attributes based on forcing
    df = process_basins(cfg, selected_df)
    log.info(f"Stations were successfully processed")
    log.info(f"Meteo basin forcings were saved at: `../{cfg.dataset.forcing}`")
    log.info(f"Dataset `.nc` files were saved at: `../{cfg.dataset.time_series}`")

    # merge with GEE attributes
    log.info("Merging attributes from GEE with those calculated from KazHydroMet")
    df = merge_attributes(df, cfg)

    # calculate attributes based on attributes
    df = post_calc_attributes(df)
    save_attributes(df, cfg, log)


def make_dirs(
    cfg: DictConfig,
) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path, pathlib.Path]:
    """Creates all the necessary directories for the dataset"""

    # create directory for dataset
    dataset_path = pathlib.Path(cfg.dataset.path)
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

    return dataset_path, forcing_path, time_series_path, attributes_path


def is_data_okay(df: pd.DataFrame) -> bool:
    """Checks if DataFrame contains all the necessary columns"""
    if set(
        [
            "date",
            "t_mean",
            "t_max",
            "t_min",
            "prcp",
        ]
    ).issubset(df.columns):
        return True
    return False


def process_basins(cfg: DictConfig, selected_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates static atributes for the whole meteo stations

    `data_dir`: str
        Must be a path to the meteo directory
    `selected_df`: pd.DataFrame
        Config DataFrame with required stations to calculate
    """

    result: list[pd.DataFrame] = []

    # take station groups by basin
    basin_groups = selected_df.groupby("basin")
    basin_groups_iter = iter(basin_groups)

    dataset_path = pathlib.Path(cfg.dataset.path)
    kazhydromet_path = pathlib.Path(cfg.kazhydromet.path)
    # all present stations
    all_stations = os.listdir(kazhydromet_path / cfg.kazhydromet.meteo)

    # iterate over basins (basin groups)
    t = trange(len(basin_groups))
    for i in t:
        t.set_description(f"Basin {i+1}")

        # get the next basin group with stations
        basin_group = next(basin_groups_iter)[1]
        # list to save the stations from one basin
        stations: list[pd.DataFrame] = []
        # iterate over the stations
        for s in basin_group.iterrows():
            # take the value from `DataFrame`, 0 is index, 1 is value
            row = s[1]
            df = load_meteo_by_name(
                row, all_stations, kazhydromet_path / cfg.kazhydromet.meteo
            )
            stations.append(df)

        # if there are > 1 station in stations merge them with the strategy for each row:
        #   - take average if both stations have the value for a given date
        #   - fill the row with data from one station if the second one doesn´t have records for that date
        if len(stations) > 1:
            df_forcing = reduce(merge_stations, stations)
        else:
            df_forcing = stations[0]

        # insert solar radiation for each day
        gee_path = pathlib.Path(cfg.dataset.gee.path)
        df_srad = load_srad_by_basin(
            row["basin"], gee_path / cfg.dataset.gee.forcing[0]
        )
        df_forcing = insert_srad(df_forcing, df_srad)

        # save merged station forcings
        df_forcing.to_csv(
            f"{dataset_path / cfg.dataset.forcing}/{row['basin']}.csv", index=False
        )

        result.append(process_basin(df_forcing, row["basin"]))

        # get streamflow data
        df_streamflow = load_hydro_by_id(
            str(row["basin"]), kazhydromet_path / cfg.kazhydromet.hydro
        )

        # merge basin forcing and streamflow
        df = merge_timeseries(df_forcing, df_streamflow)
        save_timeseries(df, str(row["basin"]), dataset_path / cfg.dataset.time_series)

        if i == len(basin_groups) - 1:
            t.set_description("Done")

    # concatenate basins into one dataframe
    return pd.concat(result, axis=0)


def process_basin(df: pd.DataFrame, basin_id: str) -> pd.DataFrame:
    """Calculates static attributes for a given basin"""

    # add new column `year`
    df_year = df.copy()
    df_year["year"] = df["date"].dt.year

    df = df_year

    # start calculations
    p_mean = calc_p_mean(df)
    p_seasonality = calc_p_seasonality(df)
    snow_frac_daily = calc_show_frac_daily(df)
    high_prec_freq = calc_high_prec_freq(df, p_mean=p_mean)
    high_prec_dur = calc_high_prec_dur(df, p_mean=p_mean)
    low_prec_freq = calc_low_prec_freq(df)
    low_prec_dur = calc_low_prec_dur(df)

    # return the dataframe with a single basin
    return pd.DataFrame(
        data={
            "basin_id": basin_id,
            "p_mean": p_mean,
            "p_seasonality": p_seasonality,
            "frac_snow_daily": snow_frac_daily,
            "high_prec_freq": high_prec_freq,
            "high_prec_dur": high_prec_dur,
            "low_prec_freq": low_prec_freq,
            "low_prec_dur": low_prec_dur,
        },
        index=[0],
    )


def post_calc_attributes(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate attributes on GEE and KazHydroMet attributes"""
    df["aridity"] = calc_aridity(df)
    return df


def merge_stations(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    Merge multiple stations
    - if date rows have value take the average
    - if some stations don't have values for date fill with mean of those
        that have the value
    """

    df1["date"] = pd.to_datetime(df1["date"])
    df2["date"] = pd.to_datetime(df2["date"])

    # save the cols to sample later
    station_cols = df1.columns.tolist()
    station_cols.remove("date")
    if "station" in station_cols:
        station_cols.remove("station")

    # merge two dataframes
    result = df1.merge(df2, how="outer", on="date", suffixes=("_df1", "_df2"))
    # merge values of cols
    for col in station_cols:
        result[col] = result[[f"{col}_df1", f"{col}_df2"]].mean(axis=1)
    # return the required cols
    return result[["date", *station_cols]]


def insert_srad(df_forcing: pd.DataFrame, df_srad: pd.DataFrame) -> pd.DataFrame:
    """Inserts solar radiation column into mean basin forcing for each day"""
    return df_forcing.merge(
        df_srad, how="left", on="date", left_index=False, right_index=False
    )


def merge_attributes(df: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """Merge calculated attributes with other attributes obtained from other sources"""

    df = df.set_index("basin_id")

    # dataset_path = pathlib.Path(cfg.dataset.path)
    gee_path = pathlib.Path(cfg.dataset.gee.path)
    files = os.listdir(gee_path)

    # search for `.csv` files in GEE directory
    for file in files:
        if file.endswith(".csv"):
            attr_df = pd.read_csv(gee_path / file, sep=",")
            attr_df = attr_df.set_index("basin_id")
            df = df.merge(attr_df, how="left", left_index=True, right_index=True)

    return df


def save_attributes(df: pd.DataFrame, cfg: DictConfig, log: logging.Logger) -> None:
    """Save the processed catchment attributes"""

    with_errors: bool = False
    dataset_path = pathlib.Path(cfg.dataset.path)
    attributes_path = dataset_path / cfg.dataset.catchment_attributes.path

    df_cols = df.columns.tolist()

    for attribute in cfg.dataset.catchment_attributes.attributes:
        attr = cfg.dataset.catchment_attributes.attributes[attribute]
        attr_features = attr.features

        if not set(attr_features).issubset(set(df_cols)):
            log.error(
                f"Error: Some attributes for `{attr.name}` are missing: {list(set(attr_features) - set(df.columns.tolist()))}"
            )
            # take the intersection of both sets
            attr_features = list(set(attr_features) & set(df_cols))
            with_errors = True

        df_attr = df[attr_features]
        filepath = attributes_path / attr.name
        df_attr.to_csv(filepath, sep=",", index=True)

    if not with_errors:
        log.info(f"Camels Attributes were successfully saved at: `{filepath}`")
    else:
        log.warning(
            f"Camels Attributes were saved at: `{filepath}` with errors, see the console logs"
        )


def load_meteo_by_name(
    row: pd.Series, all_stations: list[str], path: str
) -> pd.DataFrame:
    """Load meteo station by its name"""

    station_name = None
    # find a station file given a subset of its filename
    for s in all_stations:
        if row["meteo_station"] in s.split("_")[1]:
            station_name = s

    df = pd.read_fwf(os.path.join(path, station_name))
    try:
        is_data_okay(df)
    except:
        raise Exception("Some required columns are not present in the dataset")

    # convert date to datetime
    df["date"] = pd.to_datetime(df["date"])
    # remove year 2024
    df = df[df["date"].dt.year.ne(2024)]
    # fill NaN with 0 in precipitation
    df.loc[:, "prcp"] = df["prcp"].fillna(0)
    return df


def load_hydro_by_id(basin_id: str, path: str) -> pd.DataFrame:
    """Loads hydro station by its ID"""
    # find hydro station filename
    stations: list[str] = os.listdir(path)
    filenames = [f for f in stations if basin_id in f]

    if len(filenames) > 1:
        raise Exception(f"Multiple stations for one basin found: `{basin_id}`")
    elif len(filenames) == 0:
        raise Exception(f"Hydro station with id `{basin_id}` was not found")

    filename = filenames[0]
    df = pd.read_fwf(os.path.join(path, filename))
    # convert date
    df["date"] = pd.to_datetime(df["date"])
    # set date as index
    df = df.set_index("date")
    # return only needed features
    return df[["discharge", "level"]]


def load_srad_by_basin(basin_id: str, srad_path: pathlib.Path) -> pd.DataFrame:
    """Loads solar radiantion data for specific basin"""
    filepath = srad_path / f"{basin_id}.csv"
    try:
        df = pd.read_csv(filepath)
    except:
        raise Exception(f"File `{filepath}` not found")

    df = df.rename(columns={"SSR": "srad"})
    df["date"] = pd.to_datetime(df["date"])
    # remove records for year 2024
    df = df[df["date"].dt.year.ne(2024)]
    df = df.set_index("date")
    return df


def merge_timeseries(
    df_forcing: pd.DataFrame, df_streamflow: pd.DataFrame
) -> pd.DataFrame:
    """Merge basin mean forcing and streamflow"""
    df_forcing["date"] = pd.to_datetime(df_forcing["date"])
    df_forcing = df_forcing.set_index("date")
    df = df_forcing.merge(df_streamflow, how="inner", left_index=True, right_index=True)
    return df


def save_timeseries(df: pd.DataFrame, basin_id: str, path: str) -> bool:
    """Save timeseries as .nc files"""
    ds = df.to_xarray()
    filepath = os.path.join(path, f"{basin_id}.nc")
    try:
        ds.to_netcdf(filepath)
    except:
        raise Exception(f"Failed to save a file at `{filepath}`")


def calc_p_mean(df: pd.DataFrame) -> float:
    """Calculates mean precipitation for the whole period"""
    # calculate mean precipitations for each year
    df = df.groupby(df.date.dt.year)["prcp"].mean()
    # return the average
    return df.mean()


def calc_p_seasonality(df: pd.DataFrame) -> float:
    """Calculates precipitation seasonality"""

    df["day_of_year"] = df["date"].dt.strftime("%m-%d")

    average_prcp = df.groupby("day_of_year")["prcp"].mean().reset_index()
    average_prcp.columns = ["day_of_year", "avg_prcp"]
    sorted_days = (
        pd.date_range(start="2019-05-01", end="2020-04-30").strftime("%m-%d").tolist()
    )
    average_prcp["day_of_year"] = pd.Categorical(
        average_prcp["day_of_year"], categories=sorted_days, ordered=True
    )
    average_prcp = average_prcp.sort_values("day_of_year").reset_index(drop=True)

    average_t_mean = df.groupby("day_of_year")["t_mean"].mean().reset_index()
    average_t_mean.columns = ["day_of_year", "avg_t_mean"]
    average_t_mean = average_t_mean.sort_values(by="day_of_year").reset_index(drop=True)
    average_t_mean["day_of_year"] = pd.Categorical(
        average_t_mean["day_of_year"], categories=sorted_days, ordered=True
    )
    average_t_mean = average_t_mean.sort_values("day_of_year").reset_index(drop=True)

    extended_prcp = pd.concat([average_prcp] * 3, ignore_index=True)
    extended_prcp["29_day_avg_prcp"] = (
        extended_prcp["avg_prcp"].rolling(window=29, center=True).mean()
    )
    centered_29_day_avg = extended_prcp.iloc[366:732].reset_index(drop=True)

    extended_t_mean = pd.concat([average_t_mean] * 3, ignore_index=True)
    extended_t_mean["29_day_avg_prcp"] = (
        extended_t_mean["avg_t_mean"].rolling(window=29, center=True).mean()
    )
    centered_29_day_avg_t_mean = extended_t_mean.iloc[366:732].reset_index(drop=True)

    average_prcp["29_day_avg_prcp"] = centered_29_day_avg["29_day_avg_prcp"]
    average_t_mean["29_day_avg_t_mean"] = centered_29_day_avg_t_mean["29_day_avg_prcp"]

    df = pd.merge(average_prcp, average_t_mean, on="day_of_year")

    t_data = df["29_day_avg_t_mean"].values
    p_data = df["29_day_avg_prcp"].values

    N = len(t_data)
    t = np.linspace(0, 2 * np.pi, N)

    t_guess_mean = np.mean(t_data)
    t_guess_amp = (np.max(t_data) - np.min(t_data)) / 2
    p_guess_mean = np.mean(p_data)
    p_guess_amp = (np.max(p_data) - np.min(p_data)) / 2
    guess_phase = 0

    t_optimize_func = lambda x: x[0] * np.sin(t + x[1]) + x[2] - t_data
    p_optimize_func = lambda x: x[0] * np.sin(t + x[1]) + x[2] - p_data

    t_est_amp, t_est_phase, t_est_mean = scipy.optimize.leastsq(
        t_optimize_func, [t_guess_amp, guess_phase, t_guess_mean]
    )[0]
    p_est_amp, p_est_phase, p_est_mean = scipy.optimize.leastsq(
        p_optimize_func, [p_guess_amp, guess_phase, p_guess_mean]
    )[0]

    p_est_amp = p_est_amp / p_est_mean

    p_seasonality = (
        p_est_amp * t_est_amp / abs(t_est_amp) * np.cos(p_est_phase - t_est_phase)
    )

    return p_seasonality


def calc_show_frac_daily(df: pd.DataFrame) -> float:
    """Calculates the fraction of snow precipitation to the total precipitations for the whole period"""
    # take days where max temperature were less than 0
    df_snow_frac_daily = df[df["t_mean"] < 0]
    # take rows where precipitation is not NaN and more than 0
    df_snow_frac_daily = df_snow_frac_daily[
        (df_snow_frac_daily["prcp"].notna()) & (df_snow_frac_daily["prcp"] > 0)
    ]
    # compute the fraction of snow to the total precipitations amount
    df_snow_frac_daily = df_snow_frac_daily["prcp"].sum() / df["prcp"].sum()
    return df_snow_frac_daily


def calc_aridity(df: pd.DataFrame) -> pd.Series:
    """Calculates aridity as ratio of `pet_mean` to `p_mean`"""
    return df["pet_mean"] / df["p_mean"]


def calc_high_prec_freq(df: pd.DataFrame, p_mean: float) -> float:
    """Calculates the number of high precipitation days and averages across years"""
    # take the days where precipitations were 5 times the precipitation daily mean
    high_prec_freq_df = df[df["prcp"] >= 5 * p_mean]
    # obtain the number of such days for each year
    high_prec_freq = high_prec_freq_df.groupby(high_prec_freq_df.date.dt.year).size()
    # return the average across years
    return high_prec_freq.mean()


def calc_high_prec_dur(df: pd.DataFrame, p_mean: float) -> float:
    """Calculates the average duration of high precipitation days across all years"""
    # filter rows by the amount of precipitations
    df = df[df["prcp"] >= 5 * p_mean]
    s = df.groupby("year").date.diff().dt.days.ne(1).cumsum()
    df = df.groupby(["year", s]).size().reset_index(level=1, drop=True)
    # find average consecutive days for each year
    df_high_prec_dur_years = df.groupby("year").mean()
    # return average consecutive days across all years
    return df_high_prec_dur_years.mean()


def calc_low_prec_freq(df: pd.DataFrame) -> float:
    """Calculates the number of dry days and averages across years"""
    # take the days where precipitations were less than 1 mm/day
    low_prec_freq_df = df[df["prcp"] < 1]
    # obtain the number of such days for each year
    low_prec_freq = low_prec_freq_df.groupby(low_prec_freq_df.date.dt.year).size()
    # return the average across years
    return low_prec_freq.mean()


def calc_low_prec_dur(df: pd.DataFrame) -> float:
    """Calculates the average duration of high precipitation days across all years"""
    # filter rows by the amount of precipitations
    df = df[df["prcp"] < 1]
    s = df.groupby("year").date.diff().dt.days.ne(1).cumsum()
    df = df.groupby(["year", s]).size().reset_index(level=1, drop=True)
    # find average consecutive days for each year
    df_low_prec_dur_years = df.groupby("year").mean()
    # return average consecutive days across all years
    return df_low_prec_dur_years.mean()


def get_selected_basins(path: str) -> pd.DataFrame:
    """Loads the config dataframe and selects only basin (hydro station) and meteo station names"""
    if not os.path.exists(path):
        raise Exception(
            f"File with basin list was not found at the specified location: `{path}`"
        )

    df = pd.read_csv(path, sep=",")
    return df[["basin", "meteo_station"]]


if __name__ == "__main__":
    main()
