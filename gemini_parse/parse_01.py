from google import genai
from google.genai import types
import os 
from dotenv import load_dotenv
from pathlib import Path
from tqdm import tqdm
import argparse
import pandas as pd


# load the api key from the .env file
load_dotenv()
api_key = os.getenv("issai_api_key")
client = genai.Client(api_key=api_key)

# ==== Arguments ====
parser = argparse.ArgumentParser()
parser.add_argument(
    "--batch",
    "-b",
    help="Specify batch to parse",
    type=int,
    choices=[1, 2, 3, 4, 5, 6],
)
args = parser.parse_args()


# ==== Directories ====
data_dir = Path("../gemini_parse/tables_1_3")

output_dir = Path('../gemini_parse/output_issai_pro')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_dir = output_dir / str(args.batch)
output_dir.mkdir(exist_ok=True)

df = pd.read_csv(f"batch_files/{str(args.batch)}.csv")
_files = df["path"].values.tolist()


# ==== Parsing ====
with tqdm(
    total=len(_files),
    desc=f"Batch-{str(args.batch)}",
    unit="group",
) as pbar_outer:
    # iterate over each group
    for filepath in _files:
        # display the group name in the outer progress bar
        pbar_outer.set_description(f"Processing {filepath}")
        pbar_outer.update(1)

        with open(filepath, "rb") as f:
            pdf_file = f.read()
            filename = filepath.split("\\")[-1].split(".")[0]
        
        first_check = filename + ".md"
        # check if the file already exists
        if os.path.exists(os.path.join(output_dir, first_check)):
            pbar_outer.set_postfix({"status": "skipped"})
            print(f"File {first_check} already exists. Skipping...")
            continue

        # generate the content
        prompt = "Parse this pdf into a markdown file." 
        response = client.models._generate_content(
            model="gemini-2.5-pro-preview-05-06", 
            contents=[
                types.Part.from_bytes(
                    data=pdf_file,
                    mime_type="application/pdf",
                    ),
                    prompt]
        )
        # store the response in a markdown file 
        with open(os.path.join(output_dir, f"{filename}.md"), "w", encoding="utf-8") as f:
            f.write(response.text)