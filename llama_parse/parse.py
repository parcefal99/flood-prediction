import os
import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader


load_dotenv(dotenv_path="../.env")

# ==== Arguments ====
parser = argparse.ArgumentParser()
parser.add_argument(
    "--batch",
    "-b",
    help="Specify batch to parse",
    type=int,
    choices=[1, 2, 3, 4],
)
args = parser.parse_args()


# ==== Directories ====
data_dir = Path("./data/annual_reports/pdf")

output_dir = Path("./output")
if not output_dir.exists():
    output_dir.mkdir()

output_dir = output_dir / str(args.batch)
output_dir.mkdir(exist_ok=True)

df = pd.read_csv(data_dir / f"batch_split/{str(args.batch)}.csv")
grouping_key = df.index // 10
grouped = df.groupby(grouping_key)


# ==== Parsing ====
parser = LlamaParse(
    result_type="markdown",
    premium_mode=True,
    language="ru",
    num_workers=8,
    verbose=False,
)

file_extractor = {".pdf": parser}

with tqdm(
    total=len(grouped),
    desc=f"{str(args.part).capitalize()}-Batch-{str(args.batch)}",
    unit="group",
) as pbar_outer:
    # iterate over each group
    for group_name, group_df in grouped:
        # display the group name in the outer progress bar
        pbar_outer.set_postfix({"Group": group_name})

        _files = group_df["path"].values.tolist()
        dir_parser = SimpleDirectoryReader(
            input_files=_files, file_extractor=file_extractor
        )

        documents = dir_parser.load_data(show_progress=False)

        for d in documents:
            temp = os.path.join(
                output_dir, d.metadata["file_name"].split(".")[0] + ".md"
            )
            with open(temp, "w") as file:
                file.write(d.text)

        pbar_outer.update(1)
