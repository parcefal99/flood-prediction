from pathlib import Path
from pdf2image import convert_from_path
import PyPDF2
import fitz
import pandas as pd
import csv
import os
import re
import pdfplumber

def process_all_pdfs(base_dir):
    count = 0
    for pdf_path in sorted(base_dir.rglob("*.pdf")):
        print(f"Processing 1.1: {pdf_path}")
        process_1_1(pdf_path)
    print("----------------------------\nFinished processing 1.1\n----------------------------")

    for pdf_path in sorted(base_dir.rglob("*.pdf")):
        print(f"Processing: {pdf_path}")
        new = process_pdf(pdf_path)
        count += new
    print(count)

def process_1_1(pdf_path):
    search_term = "таблица 1.1"
    all_pages_1_1, matched_pages_1_1, added_empty_1_1, not_added_1_1 = get_pages_with_match(pdf_path, search_term)

    extracted_text = extract_text_from_pages(pdf_path, matched_pages_1_1)

    basins = get_basins("basins_96.txt")
    basin2name = get_basin_names(extracted_text, basins)
    save_to_csv(basin2name, "basin_mapping.csv")

def process_pdf(pdf_path):
    search_term = "таблица 1.1"
    all_pages_1_1, matched_pages_1_1, added_empty_1_1, not_added_1_1 = get_pages_with_match(pdf_path, search_term)

    search_term = 'таблица 1.3'
    all_pages_1_3, matched_pages_1_3, added_empty_1_3, not_added_1_3 = find_pages_with_data(pdf_path, search_term)

    save_pages_to_pdf(pdf_path, all_pages_1_1, all_pages_1_3, not_added_1_1.intersection(not_added_1_3), "report_tables_all")

    basins = get_basins_dict("basin_mapping.csv")

    save_pages_to_pdf_96(pdf_path, all_pages_1_1, matched_pages_1_3, added_empty_1_3, "report_tables_106", basins)
    save_pages_to_pdf_crop(pdf_path, all_pages_1_1, matched_pages_1_3, added_empty_1_3, "report_tables_cropped", basins)

    # save_pages_to_images(pdf_path, matched_pages1, matched_pages2, output_path)

    return len(all_pages_1_1.union(all_pages_1_3))
            
def get_pages_with_match(pdf_path, search_term):
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
                matched_pages.add(page_number + 1)
        elif ("код") in text.lower().strip().replace("–", "-").replace(" ", "") and ("водного") in text.lower().strip().replace("–", "-").replace(" ", "") and ("объекта") in text.lower().strip().replace("–", "-").replace(" ", "") and page_number in matched_pages:
            matched_pages.add(page_number + 1)

    all_pages = add_adjacent_pages(matched_pages, empty_pages)

    if not all_pages and str(pdf_path) == "reports/ural_basins/ural-2000.pdf":
        all_pages = set([11, 12, 13, 14])

    if all_pages:
        pass
        # print(f"The search term '{search_term}' was found on pages: {', '.join(map(str, sorted(all_pages)))}")
    else:
        print(f"---------------------------------------\nThe search term '{search_term}' was not found in the PDF.\n---------------------------------------")

    return (all_pages, matched_pages, all_pages.difference(matched_pages), empty_pages.difference(all_pages))

def add_adjacent_pages(main_pages, empty_pages):
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

def extract_text_from_pages(pdf_path, page_numbers):
    extracted_text = {}
    
    with open(pdf_path, 'rb') as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        for page_number in page_numbers:
            page = reader.pages[page_number - 1]
            text = page.extract_text()
            extracted_text[page_number] = text
    
    return extracted_text

def get_basins(filepath):
    basins = pd.read_csv(filepath, header=None)
    basins = sorted(basins[0].tolist())
    basins = [str(b) for b in basins]
    return basins

def get_basin_names(extracted_text, basins):
    id2name = {}
    
    for page_number, text in extracted_text.items():
        lines = text.split("\n")
        for i, line in enumerate(lines):
            for basin in basins:
                if f" {basin}" in line.lower():
                    j = i - 1
                    while ('–' not in lines[j] and "-" not in lines[j]):
                        j = j - 1
                    id2name[basin] = lines[j].strip().replace("–", "-").replace(" ", "").replace(",", ".").replace(".", "<SEP>", 1).split("<SEP>")[-1].strip('*')
    return id2name

def save_to_csv(data, csv_path):
    existing_data = {}
    if os.path.exists(csv_path):
        with open(csv_path, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file, delimiter=';')
            for row in reader:
                basin_id = row['basin_id']
                names = eval(row['name'])
                existing_data[basin_id] = names

    for basin_id, name in data.items():
        if basin_id not in existing_data:
            existing_data[basin_id] = [name]
        elif name not in existing_data[basin_id]:
            existing_data[basin_id].append(name)

    with open(csv_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['basin_id', 'name'], delimiter=';')
        writer.writeheader()
        for basin_id, names in sorted(existing_data.items(), key=lambda item: item[0]):
            writer.writerow({'basin_id': basin_id, 'name': names})

def find_pages_with_data(pdf_path, search_term):
    pdf_document = fitz.open(pdf_path)
    matched_pages = set()
    empty_pages = set()

    for page_number in range(len(pdf_document)):
        page = pdf_document[page_number]
        text = page.get_text()
        if text.strip().replace(" ", "") == "":
            empty_pages.add(page_number + 1)
        if search_term.lower().replace(' ', '') in text.lower().replace('\n', '').replace(' ', ''):
            if "число" in text.lower().replace('\n', '').replace(' ', '') and "месяц" in text.lower().replace('\n', '').replace(' ', '') and "знак" not in text.lower().replace('\n', '').replace(' ', '') and "данный" not in text.lower().replace('\n', '').replace(' ', '') and "табл." not in text.lower().replace('\n', '').replace(' ', ''):
                matched_pages.add(page_number + 1)

    all_pages = add_adjacent_pages(matched_pages, empty_pages)

    if str(pdf_path) == "reports/ural_basins/ural-2000.pdf":
        all_pages -= set([99, 100, 101, 102])

    if all_pages:
        print(f"The search terms were found on pages: {', '.join(map(str, sorted(all_pages)))}")
        # print(matched_pages)
        # print(len(matched_pages))
    else:
        print(f"---------------------------------------\nThe search terms were not found in the PDF.\n---------------------------------------")

    return (all_pages, matched_pages, all_pages.intersection(empty_pages), empty_pages.difference(all_pages))

def save_pages_to_pdf(pdf_path, all_pages_1_1, all_pages_1_3, not_added, output_dir):
    reader = PyPDF2.PdfReader(pdf_path)

    for page_number in sorted(all_pages_1_1.union(all_pages_1_3).union(not_added)):
        writer = PyPDF2.PdfWriter()
        writer.add_page(reader.pages[page_number - 1])
        new_path = Path(output_dir) / ("tables_1_1" if page_number in all_pages_1_1 else ("tables_1_3" if page_number in all_pages_1_3 else "not_added")) / f"{pdf_path.stem}-{page_number}.pdf"
        new_path.parent.mkdir(parents=True, exist_ok=True)
        with open(new_path, "wb") as f:
            writer.write(f)

def get_basins_dict(csv_path):
    basin_data = {}
    with open(csv_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file, delimiter=';')
        for row in reader:
            basin_id = row['basin_id']
            names = eval(row['name'])
            basin_data[basin_id] = names
    return basin_data

def save_pages_to_pdf_96(pdf_path, all_pages_1_1, matched_pages_1_3, added_empty_1_3, output_dir, basins):
    reader = PyPDF2.PdfReader(pdf_path)
    pdf_document = fitz.open(pdf_path)

    for page_number in sorted(all_pages_1_1):
        writer = PyPDF2.PdfWriter()
        writer.add_page(reader.pages[page_number - 1])
        new_path = Path(output_dir) / "tables_1_1" / f"{pdf_path.stem}-{page_number}.pdf"
        new_path.parent.mkdir(parents=True, exist_ok=True)
        with open(new_path, "wb") as f:
            writer.write(f)

    for page_number in sorted(matched_pages_1_3):
        page = pdf_document[page_number - 1]
        text = page.get_text()

        match = re.search(r'с([1-2][0|9][0-9][0-9])г', text.strip().lower().replace('\n', '').replace(' ', '').replace(',', '').replace('.', ''))
        if not match:
            match = re.search(r'вып[0-9][0-9]([1-2][0|9][0-9][0-9])', text.strip().lower().replace('\n', '').replace(' ', '').replace(',', '').replace('.', ''))
        if not match:
            match = re.search(r'вып[0-9]([1-2][0|9][0-9][0-9])', text.strip().lower().replace('\n', '').replace(' ', '').replace(',', '').replace('.', ''))
        if not match:
            match = re.search(r'с([1-2][0|9][0-9])г', text.strip().lower().replace('\n', '').replace(' ', '').replace(',', '').replace('.', ''))
        year = int(pdf_path.stem.split("-")[1]) if not match else (int(match.group(1)) if int(match.group(1)) != 201 else 2011)
    
        matching_basin_ids = set()
        for basin_id, names in basins.items():
            for name in names:
                if name.lower().replace('.', '').replace('-', '') in text.strip().lower().replace('\n', '').replace(' ', '').replace(',', '').replace('.', '').replace("–", "").replace("-", ""):
                    matching_basin_ids.add(basin_id)
        
        writer = PyPDF2.PdfWriter()
        writer.add_page(reader.pages[page_number - 1])

        for basin_id in matching_basin_ids:
            new_path = Path(output_dir) / "tables_1_3" / f"{basin_id}-{pdf_path.stem.split('-')[1]}-{year}.pdf"
            new_path.parent.mkdir(parents=True, exist_ok=True)
            with open(new_path, "wb") as f:
                writer.write(f)

    for page_number in sorted(added_empty_1_3):
        writer = PyPDF2.PdfWriter()
        writer.add_page(reader.pages[page_number - 1])
        new_path = Path(output_dir) / "scans_1_3" / f"{pdf_path.stem}-{page_number}.pdf"
        new_path.parent.mkdir(parents=True, exist_ok=True)
        with open(new_path, "wb") as f:
            writer.write(f)

def save_pages_to_pdf_crop(pdf_path, all_pages_1_1, matched_pages_1_3, added_empty_1_3, output_dir, basins):
    reader = PyPDF2.PdfReader(pdf_path)
    pdf_document = fitz.open(pdf_path)

    for page_number in sorted(all_pages_1_1):
        writer = PyPDF2.PdfWriter()
        writer.add_page(reader.pages[page_number - 1])
        new_path = Path(output_dir) / "tables_1_1" / f"{pdf_path.stem}-{page_number}.pdf"
        new_path.parent.mkdir(parents=True, exist_ok=True)
        with open(new_path, "wb") as f:
            writer.write(f)

    for page_number in sorted(matched_pages_1_3):
        page = pdf_document[page_number - 1]
        text = page.get_text()

        match = re.search(r'с([1-2][0|9][0-9][0-9])г', text.strip().lower().replace('\n', '').replace(' ', '').replace(',', '').replace('.', ''))
        if not match:
            match = re.search(r'вып[0-9][0-9]([1-2][0|9][0-9][0-9])', text.strip().lower().replace('\n', '').replace(' ', '').replace(',', '').replace('.', ''))
        if not match:
            match = re.search(r'вып[0-9]([1-2][0|9][0-9][0-9])', text.strip().lower().replace('\n', '').replace(' ', '').replace(',', '').replace('.', ''))
        if not match:
            match = re.search(r'с([1-2][0|9][0-9])г', text.strip().lower().replace('\n', '').replace(' ', '').replace(',', '').replace('.', ''))
        year = int(pdf_path.stem.split("-")[1]) if not match else (int(match.group(1)) if int(match.group(1)) != 201 else 2011)
    
        matching_basin_ids = set()
        for basin_id, names in basins.items():
            for name in names:
                if name.lower().replace('.', '').replace('-', '') in text.strip().lower().replace('\n', '').replace(' ', '').replace(',', '').replace('.', '').replace("–", "").replace("-", ""):
                    matching_basin_ids.add(basin_id)

        top = find_string_coordinates(pdf_path, page_number, "декада")

        if top:
            for basin_id in matching_basin_ids:
                crop_pdf_to_string(pdf_path, page_number, top, basin_id, year)
        else:
            print(f"---------------------------------------\nString 'декада' not found on page {page_number}.\n---------------------------------------")
    
    for page_number in sorted(added_empty_1_3):
        writer = PyPDF2.PdfWriter()
        writer.add_page(reader.pages[page_number - 1])
        new_path = Path(output_dir) / "scans_1_3" / f"{pdf_path.stem}-{page_number}.pdf"
        new_path.parent.mkdir(parents=True, exist_ok=True)
        with open(new_path, "wb") as f:
            writer.write(f)

def find_string_coordinates(pdf_path, page_number, target_string):
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_number - 1]
        for i, word in enumerate(page.extract_words()):
            if word['text'].lower() == target_string:
                return word['top']
        return None
    
def crop_pdf_to_string(pdf_path, page_number, top, basin_id, year):
    for i in range(2):
        reader = PyPDF2.PdfReader(pdf_path)
        writer = PyPDF2.PdfWriter()
        page = reader.pages[page_number - 1]
        mediabox = page.mediabox
        page.mediabox.lower_left = (mediabox.lower_left[0], float(mediabox.upper_right[1]) - top if i == 0 else mediabox.lower_left[1])
        page.mediabox.upper_right = (mediabox.upper_right[0], mediabox.upper_right[1] if i == 0 else float(mediabox.upper_right[1]) - top + 1)
        writer.add_page(page)
        new_path = Path(f"report_tables_cropped/tables_1_3_{"top" if i == 0 else "bottom"}/{basin_id}-{pdf_path.stem.split('-')[1]}-{year}-{"top" if i == 0 else "bottom"}.pdf")
        new_path.parent.mkdir(parents=True, exist_ok=True)
        with open(new_path, "wb") as f:
            writer.write(f)

base_dir = Path("reports")
process_all_pdfs(base_dir)