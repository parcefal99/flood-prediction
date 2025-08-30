# This script processes KazHydroMet Annual Yearbooks (PDF files) to extract basin IDs and other relevant information.

from pathlib import Path
import fitz
import pandas as pd
import csv
import os
import re
import pdfplumber
import tqdm 


def process_all_pdfs(base_dir: Path) -> None:
    """
    
    Recursively find all PDF files in the base directory
    
    """
    with tqdm.tqdm(
        total=len(list(base_dir.rglob("*.pdf"))), 
        desc="Processing PDFS"
    ) as pbar:
        for pdf_path in sorted(base_dir.rglob("*.pdf")):
            # the following PDFs are not processed because they are scans of printed documents OR table 1.1 does not exist in them
            # and should be OCRed first
            if pdf_path.name == "ural-2015.pdf" or pdf_path.name == "ural-2018.pdf" or pdf_path.name == "ertis-2015.pdf" or pdf_path.name == "ural-2000.pdf":
                continue
            year = int(pdf_path.name.split("-")[-1].split(".")[0])
            pbar.set_description(f"Processing {pdf_path}")
            process_1_1(pdf_path, year)
            pbar.update(1)
    
    print("----------------------------\nFinished processing 1.1\n----------------------------")


def process_1_1(pdf_path: Path, year: int) -> None:
    """
    
    Process the PDF file to extract basin IDs and names from Table 1.1.
    
    """

    search_term = "таблица 1.1"
    all_pages_1_1, matched_pages_1_1, added_empty_1_1, not_added_1_1 = get_pages_with_match(pdf_path, search_term)

    extracted_text = extract_text_from_pages(pdf_path, matched_pages_1_1)

    basin_ids = get_pairs(extracted_text, year)
    # print(basin_ids)
    # print(len(basin_ids))
    # save_to_csv(basin_ids, "basin_ids.csv")
    save_to_csv_id_to_name(basin_ids, Path(base_dir, "basin_id_to_name.csv"))
    save_to_csv_name_to_id(basin_ids, Path(base_dir, "name_to_basin_id.csv"))


def get_pages_with_match(pdf_path: Path, search_term: str) -> tuple:
    """
    Search for a term in the PDF and return pages where it is found, along with adjacent pages and empty pages.
    """

    pdf_document = fitz.open(pdf_path)
    matched_pages = set()
    empty_pages = set()

    for page_number in range(len(pdf_document)):
        page = pdf_document[page_number]
        text = page.get_text()
        if text.strip().replace(" ", "") == "":
            empty_pages.add(page_number + 1)
        if search_term.lower().replace(" ", "") in text.lower().strip().replace("–", "-").replace(" ", ""):
            if ("Список постов").lower().strip().replace("–", "-").replace(" ", "") in text.lower().strip().replace("–", "-").replace(" ", "") and ("код") in text.lower().strip().replace("–", "-").replace(" ", "") and ("объекта") in text.lower().strip().replace("–", "-").replace(" ", "") and ("таблица1.2") not in text.lower().strip().replace("–", "-").replace(" ", "") and ("знак") not in text.lower().strip().replace("–", "-").replace(" ", ""):
                matched_pages.add(page_number)
        elif ("код") in text.lower().strip().replace("–", "-").replace(" ", "") and ("водного") in text.lower().strip().replace("–", "-").replace(" ", "") and ("объекта") in text.lower().strip().replace("–", "-").replace(" ", "") and page_number in matched_pages:
            matched_pages.add(page_number)

    all_pages = add_adjacent_pages(matched_pages, empty_pages)

    if not all_pages and str(pdf_path) == "reports/ural_basins/ural-2000.pdf":
        all_pages = set([11, 12, 13, 14])

    if all_pages:
        pass
        # print(f"The search term '{search_term}' was found on pages: {', '.join(map(str, sorted(all_pages)))}")
    else:
        print(f"---------------------------------------\nThe search term '{search_term}' was not found in the PDF.\n---------------------------------------")

    return (all_pages, matched_pages, all_pages.difference(matched_pages), empty_pages.difference(all_pages))

def add_adjacent_pages(main_pages: set, empty_pages: set) -> set:
    added_pages = True
    main = main_pages.copy()
    empty = empty_pages.copy()
    
    while added_pages:
        added_pages = False
        new_pages = {
            page for page in empty 
            if page - 1 in main or page + 1 in main
        }
        if new_pages:
            main.update(new_pages)
            empty -= new_pages
            added_pages = True
    
    return main


def extract_text_from_pages(pdf_path: Path, pages: list) -> str:
    """
    Extract text from specified pages of a PDF file.

    """

    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page_num in pages:
            page = pdf.pages[page_num]
            text += page.extract_text() + "\n"
    return text

def get_pairs(data: str, year: int) -> list:
    """
    Extract (Код поста, Название) pairs from PDF text.
    Assumes: Name lines start with "<number>. <name>", and Код поста appears on a later line.
    """
    basin_ids = []
    pending_name = None

    lines = data.splitlines()
    for line in lines:
        line = line.strip()

        # 1. Look for lines that start with an index (e.g. "23. ..." or "142. ...")
        name_match = re.match(r'^(\d+)\.\s*(.+)', line)
        if name_match:
            pending_name = name_match.group(2).strip()
            continue  # move to next line to find ID

        # 2. Look for 5-digit Код поста (post ID)
        id_match = re.search(r'\b(\d{5})\b', line)
        if id_match and pending_name:
            post_id = id_match.group(1)
            basin_ids.append((post_id, pending_name, year))
            pending_name = None  # reset after using it

    return basin_ids


def save_to_csv_id_to_name(data: list, filename: str) -> None:
    """
    Save basin_id (int) -> [name1, name2, ...] to CSV.
    """
    existing_data = {}

    # Load existing data if file exists
    if os.path.exists(filename):
        with open(filename, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file, delimiter=';')
            for row in reader:
                basin_id = row['basin_id']
                name = eval(row['name'])
                existing_data[basin_id] = name

    # Update with new data
    for basin_id, name in data:
        if basin_id not in existing_data:
            existing_data[basin_id] = [name]
        elif name not in existing_data[basin_id]:
            existing_data[basin_id].append(name)

    # Save to CSV
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['basin_id', 'name'], delimiter=';')
        writer.writeheader()
        for basin_id, name in sorted(existing_data.items()):
            writer.writerow({'basin_id': basin_id, 'name': name})


def save_to_csv_name_to_id(data: list, filename: str) -> None:
    """
    Save name (str) -> [basin_id1, basin_id2, ...] to CSV.
    """
    existing_data = {}

    # Load existing data if file exists
    if os.path.exists(filename):
        with open(filename, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file, delimiter=';')
            for row in reader:
                name = row['name']
                ids = eval(row['basin_ids'])
                existing_data[name] = ids

    # Update with new data
    for basin_id, name in data:
        if name not in existing_data:
            existing_data[name] = [basin_id]
        elif basin_id not in existing_data[name]:
            existing_data[name].append(basin_id)

    # Save to CSV
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['name', 'basin_ids'], delimiter=';')
        writer.writeheader()
        for name, ids in sorted(existing_data.items()):
            writer.writerow({'name': name, 'basin_ids': ids})

# def save_to_csv_name_to_id(data: list, filename: str) -> None:
#     """
#     Save (basin_id, name, year) tuples into CSV with columns:
#     name, id_2000, id_2001, ..., id_2022
#     """
#     START_YEAR = 2000
#     END_YEAR = 2022
#     years = list(range(START_YEAR, END_YEAR + 1))
#     year_columns = [f'id_{year}' for year in years]

#     # Load existing data if file exists
#     existing_data = {}  # name -> {year: id}
#     if os.path.exists(filename):
#         with open(filename, mode='r', encoding='utf-8') as file:
#             reader = csv.DictReader(file)
#             for row in reader:
#                 name = row['name']
#                 existing_data[name] = {}
#                 for year in years:
#                     key = f'id_{year}'
#                     existing_data[name][year] = row.get(key) or ''

#     # Update with new data
#     for basin_id, name, year in data:
#         if name not in existing_data:
#             existing_data[name] = {y: '' for y in years}
#         existing_data[name][year] = basin_id

#     # Write updated data to CSV
#     with open(filename, mode='w', newline='', encoding='utf-8') as file:
#         fieldnames = ['name'] + year_columns
#         writer = csv.DictWriter(file, fieldnames=fieldnames)
#         writer.writeheader()

#         for name in sorted(existing_data):
#             row = {'name': name}
#             for year in years:
#                 row[f'id_{year}'] = existing_data[name].get(year, '')
#             writer.writerow(row)


if __name__ == "__main__":
    base_dir = Path("reports")
    process_all_pdfs(base_dir)
    print("All PDFs processed successfully.")