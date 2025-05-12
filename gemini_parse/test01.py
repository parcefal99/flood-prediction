from google import genai
from google.genai import types
import os 
from dotenv import load_dotenv
from pathlib import Path
from tqdm import tqdm
import argparse

# path to the gemini
gen_path = './tables_1_3'
output_path = './tables_1_3/output'

if not os.path.exists(output_path):
    os.makedirs(output_path)

# load the api key from the .env file
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

client = genai.Client(api_key=api_key)

pdf_file = os.path.join(gen_path, "ba-2022-103.pdf")
prompt = "This is a page from a PDF document. Extract all text content while preserving the structure. Pay special attention to tables, columns, headers, and any structured content. Maintain paragraph breaks and formatting."
response = client.models._generate_content(
    model="gemini-2.5-pro-exp-03-25", 
    contents=[
        types.Part.from_bytes(
            data=open(pdf_file, "rb").read(),
            mime_type="application/pdf",
            ),
            prompt]
)
print(response.text)

# store the response in a markdown file 
with open(os.path.join(output_path, "ba-2022-103.md"), "w", encoding="utf-8") as f:
    f.write(response.text)
