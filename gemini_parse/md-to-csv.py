import re
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import trange

from calendar import monthrange


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
    IMD = 6 

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
        elif cls.value == 5:
            return "Other Problem"
        else:
            return "Invalid month days"


def main() -> None:
    # get all parsed files
    basins_path = Path("./markdown")
    basins = sorted(list(basins_path.rglob("*.md")))
    # set output directory
    output_dir = Path("./mt1t_md/output")
    output_dir.mkdir(exist_ok=True)
    # set error directory
    error_dir = Path("./mt1t_md/errors")
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
        if len(basin_filename_split) > 2:
            file_stats[Message.MT1T].append(basin.name)
            continue

        basin_name = basin_filename_split[0]
        year = basin_filename_split[-1].split(".")[0]

        try:
            df, flag = parse_basin(basin, year)
            if df is not None:
                df.to_csv(output_dir / f"{basin_name}-{year}.csv", sep=";")
        except:
            flag = Message.OP
        file_stats[flag].append(basin.name)

        if i == len(basins) - 1:
            t.set_description("Done")
        # break

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
    tables_count = 0
    for i, line in enumerate(extracted_text.split("\n")):
        line = " ".join(line.split())
        if line.strip() != "":
            lines.append(line)
        if line.lower().replace(" ", "").find("f=") != -1:
            tables_count += 1

    if tables_count > 1:
        return None, Message.MT1T

    pattern = r"^\|\s*\d+\s*\|"
    matching_rows = [i for i, line in enumerate(lines) if re.match(pattern, line)]
    if not matching_rows:
        return None, Message.OP

    period_start = matching_rows[0]
    period_end = matching_rows[-1]

    month_seq = extract_month_numbers(extracted_text)
    if not is_growing_sequence(month_seq):
        return None, Message.MT1T

    expected_cols = len(month_seq) + 1
    days = []
    result = []
    indices = []

    for i, line in enumerate(lines[period_start : period_end + 1]):
        original_line = line  # Keep for diagnostics
        # Cleanup
        line = line.replace("^", "").replace("~", "").replace(",", ".").replace('нб', '0').replace("—", "-").replace("–", "-").replace("но", "0")
        line = re.sub(r"[^\d.\sA-Za-zА-Яа-яЁё|\-]", "", line)
        for _ in range(12):
            line = line.replace("| |", "|-1|")
        line = line.replace(" ", "")

        row = line.split("|")
        row = [r for r in row if r != ""]

        # Error-tolerant row handling
        if abs(len(row) - expected_cols) <= 2:
            if len(row) < expected_cols:
                row += ["-1"] * (expected_cols - len(row))
            elif len(row) > expected_cols:
                row = row[:expected_cols]
            try:
                day = row.pop(0)
                days.append(int(day))
                indices.append(day)
                result.append(row)
            except:
                # log_bad_row(filepath, original_line, i + period_start)
                return None, Message.OP
        else:
            # log_bad_row(filepath, original_line, i + period_start)
            return None, Message.LTSMS

    # Extra check for missing or non-continuous days
    unique_days = sorted(set(days))
    if max(unique_days) < 28:
        return None, Message.LT31D
    if len(unique_days) != len(days) or not is_continuous(unique_days):
        return None, Message.NCS
    

    df = pd.DataFrame(data=result, index=indices, columns=[i for i in month_seq])
   
    temp = is_valid_ending_days(df)
    if temp is False:
        return None, Message.IMD
    
    discharge = []
    dates = []
    for month in df.columns:
        _df = df[month]
        discharge.extend(_df)
        date = [f"{year}-{month:02d}-{int(day):02d}" for day in _df.index]
        dates.extend(date)

    
    df = pd.DataFrame({"date": dates, "discharge": discharge})
    df = df[~(df["discharge"] == "-1")]
    df["discharge"] = df["discharge"].replace("-", np.nan)
    df["discharge"] = df["discharge"].replace("нб", 0).astype(float)
    df["date"] = pd.to_datetime(df["date"],format="%Y-%m-%d", errors="coerce")
    df = df.dropna(subset=["date"])
    df.set_index("date", inplace=True)

    return df, Message.SUCCESS



# def log_bad_row(filepath: Path, line: str, lineno: int) -> None:
#     error_dir = Path("../errors/bad_rows")
#     error_dir.mkdir(parents=True, exist_ok=True)
#     log_path = error_dir / f"{filepath.stem}.log"
#     with open(log_path, "a", encoding="utf-8") as f:
#         f.write(f"[Line {lineno}] {line}\n")



def get_markdown(filepath) -> str:
    with open(filepath, "r") as f:
        md = f.read()
        return md


def merge_basins(output_dir: Path) -> list:
    basins = get_basins("./basin_ids.csv")

    streamflow_path = Path("./mt1t_md/streamflow/")
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
            if len(file.name.split("-")) > 2:
                no_data_basins.append(basin)
                continue
            _df = pd.read_csv(file, sep=";")
            _df["date"] = pd.to_datetime(_df["date"])
            _df.set_index("date", inplace=True)

            if j == 0:
                df = _df
                continue

            df = pd.concat([df, _df], axis=0)
        
        # to remove this error cannot reindex on an axis with duplicate labels
        df = df[~df.index.duplicated(keep='first')]
        df = df.sort_index()

        # fill missing dates
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

# edited to extract month numbers from markdown text 
def extract_month_numbers(markdown_text: str) -> list[int]:
    import unicodedata

    roman_to_int = {
        "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6,
        "VII": 7, "VIII": 8, "IX": 9, "X": 10, "XI": 11, "XII": 12
    }

    def normalize_token(token: str) -> str:
        # Remove bold markers and normalize unicode
        token = re.sub(r"\*\*", "", token)
        return unicodedata.normalize("NFKD", token.strip().upper())

    rows = markdown_text.strip().split("\n")
    candidate_rows = []

    for row in rows:
        if row.count("|") >= 6:
            candidate_rows.append(row)

    for row in candidate_rows:
        tokens = [normalize_token(t) for t in row.split("|") if t.strip()]
        month_numbers = []

        for token in tokens:
            if token.startswith("МЕСЯЦ"):
                token = token.replace("МЕСЯЦ", "").strip()
            if token.isdigit():
                num = int(token)
                if 1 <= num <= 12:
                    month_numbers.append(num)
            elif token in roman_to_int:
                month_numbers.append(roman_to_int[token])

        if 3 <= len(month_numbers) <= 12:
            return month_numbers

    return []



def is_valid_ending_days(df:pd.DataFrame) -> bool:
    # check if the values of the last days of each month are valid 
    for month in df.columns:
        if int(month) == 2: # February
            last_day = df.iloc[-1, 1]
            pre_last_day = df.iloc[-2, 1]
            if last_day in ("-1", "-", "нб", '0') and pre_last_day in ("-1", "-", "нб", '0'):
                continue
            else:
                return False
        elif str(month) in ['4', '6', '9', '11']:
            last_day = df.iloc[-1, int(month) - 1]
            if last_day in ("-1", "-", "нб", '0'):
                continue
            else:
                return False 
    return True            
       

if __name__ == "__main__":
    main()
