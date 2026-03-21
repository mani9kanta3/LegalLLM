"""
Step 2: Extract text from Indian labour law PDFs.

Why PyMuPDF (fitz)?
- It's the fastest Python PDF library
- Handles scanned + digital PDFs well
- Preserves text structure better than PyPDF2
- Legal docs have complex formatting (tables, sections, schedules)
  and PyMuPDF handles them cleanly

Usage: python scripts/extract_text.py
"""

import os
import pymupdf as fitz  # PyMuPDF — the import name is 'fitz' for historical reasons
import json


def extract_text_from_pdf(pdf_path):
    """
    Extract text from a single PDF file.
    
    How it works:
    - Opens the PDF
    - Iterates through each page
    - Extracts text from each page
    - Joins all pages with clear page markers
    
    Returns:
        dict with 'text', 'num_pages', 'num_words', 'filename'
    """
    doc = fitz.open(pdf_path)
    
    num_pages = len(doc)  # Save page count BEFORE closing
    
    pages_text = []
    for page_num in range(num_pages):
        page = doc[page_num]
        text = page.get_text()  # Extracts text in reading order
        
        if text.strip():  # Only add non-empty pages
            pages_text.append(text.strip())
    
    doc.close()
    
    full_text = "\n\n".join(pages_text)
    
    return {
        "filename": os.path.basename(pdf_path),
        "text": full_text,
        "num_pages": num_pages,
        "num_words": len(full_text.split()),
    }


def main():
    # Paths
    raw_dir = os.path.join("data", "raw")
    processed_dir = os.path.join("data", "processed")
    
    # Make sure processed directory exists
    os.makedirs(processed_dir, exist_ok=True)
    
    # Find all PDFs
    pdf_files = sorted([f for f in os.listdir(raw_dir) if f.endswith(".pdf")])
    
    if not pdf_files:
        print("ERROR: No PDF files found in data/raw/")
        print("Make sure you copied the 9 PDFs into the data/raw/ folder.")
        return
    
    print(f"Found {len(pdf_files)} PDF files\n")
    print("=" * 70)
    
    # Track stats for summary
    all_stats = []
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(raw_dir, pdf_file)
        print(f"Processing: {pdf_file}...")
        
        # Extract text
        result = extract_text_from_pdf(pdf_path)
        
        # Save extracted text as .txt file
        # e.g., "minimum_wages_act_1948.pdf" -> "minimum_wages_act_1948.txt"
        txt_filename = pdf_file.replace(".pdf", ".txt")
        txt_path = os.path.join(processed_dir, txt_filename)
        
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(result["text"])
        
        # Print stats for this document
        print(f"  Pages: {result['num_pages']}")
        print(f"  Words: {result['num_words']:,}")
        print(f"  Saved: {txt_path}")
        print()
        
        all_stats.append({
            "filename": pdf_file,
            "pages": result["num_pages"],
            "words": result["num_words"],
            "txt_file": txt_filename,
        })
    
    # Save extraction stats as JSON (useful for documentation later)
    stats_path = os.path.join(processed_dir, "extraction_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2)
    
    # Print summary table
    print("=" * 70)
    print(f"{'Document':<45} {'Pages':>6} {'Words':>8}")
    print("-" * 70)
    
    total_pages = 0
    total_words = 0
    for stat in all_stats:
        name = stat["filename"].replace(".pdf", "")
        print(f"{name:<45} {stat['pages']:>6} {stat['words']:>8,}")
        total_pages += stat["pages"]
        total_words += stat["words"]
    
    print("-" * 70)
    print(f"{'TOTAL':<45} {total_pages:>6} {total_words:>8,}")
    print("=" * 70)
    print(f"\nAll texts saved to: {processed_dir}/")
    print(f"Stats saved to: {stats_path}")


if __name__ == "__main__":
    main()