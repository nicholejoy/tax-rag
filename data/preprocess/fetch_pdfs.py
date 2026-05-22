import json
import os
from pathlib import Path

import fitz
import requests

DATA_DIR = Path(__file__).resolve().parents[1]


def download_pdf(url, output_path):
    response = requests.get(url)
    response.raise_for_status()
    with open(output_path, "wb") as f:
        f.write(response.content)
    return output_path


def extract_pdf_text(pdf_path):
    doc = fitz.open(pdf_path)
    pages = [page.get_text("text") for page in doc]
    return "\n\n".join(pages)


def process_document(url, doc_title, output_path):
    print(f"Processing: {doc_title}")

    pdf_path = f"/tmp/{doc_title}.pdf"
    download_pdf(url, pdf_path)
    text = extract_pdf_text(pdf_path)
    os.remove(pdf_path)

    record = {"title": doc_title, "content": text, "url": url}
    with open(output_path, "a") as f:
        f.write(json.dumps(record) + "\n")

    print(f"Done: {len(text)} characters extracted")


if __name__ == "__main__":
    output_path = DATA_DIR / "input.jsonl"

    documents = [
        {"title": "VAT 404 Guide", "url": "https://www.sars.gov.za/wp-content/uploads/Ops/Guides/Legal-Pub-Guide-VAT404-VAT-404-Guide-for-Vendors.pdf"},
        {"title": "Income Tax Guide", "url": "https://www.sars.gov.za/wp-content/uploads/Ops/Guides/PAYE-GEN-01-G20-Guide-for-Employers-iro-Employees-Tax-for-2026-External-Guide.pdf"},
        {"title": "Residence Based Taxation Guide", "url": "https://www.sars.gov.za/wp-content/uploads/Ops/Guides/LAPD-IT-G02-Guide-on-the-Residence-Basis-of-Taxation-for-Individuals.pdf"},
        {"title": "income Tax Guide Individual", "url": "https://www.sars.gov.za/wp-content/uploads/Ops/Guides/Legal-Pub-Guide-IT01-Guide-on-Income-Tax-and-the-Individual.pdf"},
        {"title": "VAT 201 Guide", "url": "https://www.sars.gov.za/wp-content/uploads/Ops/Guides/GEN-ELEC-04-G01-Guide-for-completing-the-Value-Added-Tax-VAT201-Declaration-External-Guide.pdf"},
    ]

    for doc in documents:
        process_document(doc["url"], doc["title"], output_path)
