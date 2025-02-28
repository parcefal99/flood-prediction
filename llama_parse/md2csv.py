import re
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import trange


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


class Message(Enum):
    SUCCESS = 0
    MT1T = 1
    LT31D = 2
    LTSMS = 3
    NCS = 4
    OP = 5

    def get_name(cls) -> str:
        if cls.value == 0:
            return "Success"
        elif cls.value == 1:
            return "More than 1 table"
        elif cls.value == 2:
            return "Less than 31 days"
        elif cls.value == 3:
            return "Less than specified months sequence"
        elif cls.value == 4:
            return "Non-consecutive day sequence"
        else:
            return "Other Problem"


def main() -> None:
    # get all parsed files
    basins_path = Path("../data/yearbooks_collection/markdown/")
    basins = sorted(list(basins_path.rglob("*.md")))
    # set output directory
    output_dir = Path("./output")
    output_dir.mkdir(exist_ok=True)
    # set error directory
    error_dir = Path("./errors")
    error_dir.mkdir(exist_ok=True)

    file_stats = {}
    for msg in Message:
        file_stats[msg] = []

    print(f"{bcolors.HEADER}Parsing the markdown files!{bcolors.ENDC}")
    t = trange(len(basins))
    for i in t:
        basin = basins[i]
        t.set_description(f"File: {basin.name}")

        # obtain the year of parsed data
        basin_filename_split = basin.name.split("-")
        basin_name = basin_filename_split[0]
        year = basin_filename_split[-2]

        try:
            df, flag = parse_basin(basin, year)
            if df is not None:
                df.to_csv(output_dir / f"{basin_name}-{year}.csv", sep=";")
        except:
            flag = Message.OP
        file_stats[flag].append(basin.name)

        if i == len(basins) - 1:
            t.set_description("Done")

    # merge all csv files of the same basin
    print(f"{bcolors.HEADER}Merging each basin into CSV!{bcolors.ENDC}")
    no_data_basins = merge_basins(output_dir)

    print()
    print(
        f"{bcolors.BOLD}{bcolors.OKGREEN}Parse Statistics:{bcolors.ENDC}{bcolors.ENDC}"
    )
    for k, v in file_stats.items():
        msg = k.get_name()
        if len(v) != 0:
            pd.DataFrame(data=v, columns=[msg]).to_csv(
                error_dir / f"{k.name}.csv", index=False
            )
        print(f"  {bcolors.OKBLUE}{msg}{bcolors.ENDC}: {len(v)}")
    print(
        f"  {bcolors.WARNING}Basins with no data:{bcolors.ENDC} {len(no_data_basins)}"
    )
    print()

    # save basins with no data
    pd.DataFrame(data=no_data_basins, columns=["Basins with no data"]).to_csv(
        error_dir / f"NDB.csv", index=False
    )

    # if no errors, then remove the directory
    if not any(error_dir.iterdir()):
        error_dir.rmdir()


def parse_basin(filepath: Path, year) -> tuple[pd.DataFrame, Message]:
    extracted_text = get_markdown(filepath)

    lines = []
    period_end = None
    tables_count = 0
    for i, line in enumerate(extracted_text.split("\n")):
        line = " ".join(line.split())
        if line.strip() != "":
            lines.append(line)
        if line.lower().startswith("| 20"):
            period_end = i
            tables_count += 1

    if tables_count > 1:
        return None, Message.MT1T

    pattern = r"^\|\s*\d+\s*\|"
    matching_rows = [i for i, line in enumerate(lines) if re.match(pattern, line)]
    period_start = matching_rows[0]
    period_end = matching_rows[-1]

    days = []
    result = []
    indices = []

    # check month sequence continuity
    month_seq = extract_month_numbers(extracted_text)
    if not is_growing_sequence(month_seq):
        return None, Message.MT1T

    for line in lines[period_start : period_end + 1]:
        # filter out redundant symbols
        # leaves only numbers, latin and cyrillic symbols, dots and dashes
        line = re.sub(r"[^0-9.,\sA-Za-zА-Яа-яЁё|-]", "", line)
        for _ in range(12):
            line = line.replace("| |", "|-1|")
        line = line.replace(",", ".")
        line = line.replace(" ", "")
        row = line.split("|")
        row = [r for r in row if r != ""]

        try:
            if len(row) == len(month_seq) + 1:
                day = row.pop(0)
                days.append(int(day))
                indices.append(day)
                result.append(row)
            else:
                return None, Message.LTSMS
        except:
            return None, Message.OP

    # verify the number of days
    if len(days) < 31:
        return None, Message.LT31D

    # verify the continuity of the days
    if not is_continuous(days):
        return None, Message.NCS

    df = pd.DataFrame(data=result, index=indices, columns=[i for i in month_seq])

    discharge = []
    dates = []

    # Iterate over columns to append them in order
    for month in df.columns:
        _df = df[month]
        discharge.extend(_df)
        date = [f"{year}-{month}-{day}" for day in _df.index]
        dates.extend(date)

    # Create a new DataFrame with the extended column
    df = pd.DataFrame({"date": dates, "discharge": discharge})
    df = df[~(df["discharge"] == "-1")]
    df.loc[:, "discharge"] = df["discharge"].replace("-", np.nan)
    df.loc[:, "discharge"] = df["discharge"].replace("нб", float(0))
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df.set_index("date", inplace=True)
    return df, Message.SUCCESS


def get_markdown(filepath) -> str:
    with open(filepath, "r") as f:
        md = f.read()
        return md


def merge_basins(output_dir: Path) -> list:
    basins = get_basins("../ML/basins_sigma_85.txt")

    streamflow_path = Path("./streamflow/")
    streamflow_path.mkdir(exist_ok=True)

    no_data_basins: list = []

    t = trange(len(basins))
    for i in t:
        basin = basins[i]
        t.set_description(f"Basin {basin}")
        basins_by_year = sorted(list(output_dir.rglob(f"*{basin}*")))

        if len(basins_by_year) <= 0:
            no_data_basins.append(basin)
            continue

        df = None
        for j, file in enumerate(basins_by_year):
            # print(file.name)
            _df = pd.read_csv(file, sep=";")
            _df["date"] = pd.to_datetime(_df["date"])
            _df.set_index("date", inplace=True)

            if j == 0:
                df = _df
                continue

            df = pd.concat([df, _df], axis=0)

        df = df.asfreq("D", fill_value=np.nan)
        df.to_csv(streamflow_path / f"{basin}.csv", sep=",")

        if i == len(basins) - 1:
            t.set_description("Done")

    return no_data_basins


def get_basins(filepath: str) -> list[str]:
    basins = pd.read_csv(filepath, header=None)
    basins = sorted(basins[0].tolist())
    basins = [str(b) for b in basins]
    return basins


def is_continuous(arr: list) -> bool:
    for i in range(1, len(arr)):
        if arr[i] - arr[i - 1] != 1:
            return False
    return True


def is_growing_sequence(arr: list) -> bool:
    for i in range(1, len(arr)):
        if arr[i] - arr[i - 1] < 1:
            return False
    return True


def extract_month_numbers(markdown_text):
    # Find all rows that might contain month numbers
    rows = markdown_text.strip().split("\n")
    header_row = None

    # Identify the row with month numbers
    for row in rows:
        if re.search(
            r"\|\s*\d+\s*\|", row
        ):  # Look for a row containing digits between '|'
            header_row = row
            break

    if not header_row:
        return []

    # Extract the numbers from the identified header row
    month_pattern = r"\b\d{1,2}\b"  # Matches one or two digit numbers
    month_numbers = re.findall(month_pattern, header_row)
    return [int(month) for month in month_numbers]


if __name__ == "__main__":
    main()
