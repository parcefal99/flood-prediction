import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

from llama_parse import LlamaParse


load_dotenv(dotenv_path="../.env")

# ==== Arguments ====
parser = argparse.ArgumentParser()
parser.add_argument(
    "--input",
    "-i",
    help="Specify input file with documents to parse",
    type=str,
    required=True,
)
args = parser.parse_args()


# ==== Directories ====
output_dir = Path("./output")

df = pd.read_csv(str(args.input), sep=";")


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
    total=len(df),
    desc=f"Parsing",
    unit="document",
) as pbar_outer:
    # Iterate over each group
    for index, row in df.iterrows():
        # Display the group name in the outer progress bar
        # pbar_outer.set_postfix({"Group": group_name})
        pbar_outer.set_description(f"{row['part']}-{str(row['batch'])}")

        _file = row["path"]
        document = parser.load_data(_file)

        _output_dir = output_dir / row["part"] / str(row["batch"])
        filename = f"{Path(_file).name.split('.')[0]}.md"

        output_file = _output_dir / filename
        with open(output_file, "w") as file:
            file.write(document[0].text)

        pbar_outer.update(1)
