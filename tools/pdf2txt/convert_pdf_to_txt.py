"""Convert C228 PDF to clean UTF-8 text file.

Usage:
    python scratch/convert_pdf_to_txt.py

This script converts the C228 reference paper PDF to a clean UTF-8 text file
with proper Chinese encoding. It tries multiple PDF extraction methods in order
of preference: PyMuPDF (fitz) -> pdfplumber -> pypdf.

Output: scratch/c228_clean.txt (UTF-8 encoded, with page breaks)
"""
from pathlib import Path
import sys


def convert_with_pymupdf(pdf_path: Path, txt_path: Path) -> bool:
    """Convert using PyMuPDF (fitz) - fastest and most reliable."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return False
    
    print("Using PyMuPDF (fitz)...")
    doc = fitz.open(pdf_path)
    page_count = len(doc)
    text_parts = []
    
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text", sort=True)
        text_parts.append(text)
        # Add page break between pages (except after last page)
        if page_num < page_count:
            text_parts.append("\f")
    
    full_text = "".join(text_parts)
    txt_path.write_text(full_text, encoding="utf-8")
    doc.close()
    
    print(f"[OK] Converted {page_count} pages -> {txt_path}")
    print(f"  File size: {txt_path.stat().st_size:,} bytes")
    return True


def convert_with_pdfplumber(pdf_path: Path, txt_path: Path) -> bool:
    """Convert using pdfplumber - good for tables."""
    try:
        import pdfplumber
    except ImportError:
        return False
    
    print("Using pdfplumber...")
    with pdfplumber.open(pdf_path) as pdf:
        text_parts = []
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)
                text_parts.append("\f")
        
        full_text = "".join(text_parts)
        txt_path.write_text(full_text, encoding="utf-8")
        
        print(f"[OK] Converted {len(pdf.pages)} pages -> {txt_path}")
        print(f"  File size: {txt_path.stat().st_size:,} bytes")
        return True


def convert_with_pypdf(pdf_path: Path, txt_path: Path) -> bool:
    """Convert using pypdf (formerly PyPDF2) - fallback method."""
    try:
        from pypdf import PdfReader
    except ImportError:
        try:
            from PyPDF2 import PdfReader
        except ImportError:
            return False
    
    print("Using pypdf/PyPDF2...")
    reader = PdfReader(pdf_path)
    text_parts = []
    
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        text_parts.append(text)
        if page_num < len(reader.pages):
            text_parts.append("\f")
    
    full_text = "".join(text_parts)
    txt_path.write_text(full_text, encoding="utf-8")
    
    print(f"[OK] Converted {len(reader.pages)} pages -> {txt_path}")
    print(f"  File size: {txt_path.stat().st_size:,} bytes")
    return True


def main():
    # Determine workspace root
    script_dir = Path(__file__).parent
    root = script_dir.parent
    
    pdf_path = root / "docs" / "(C228)基于价格弹性的蔬菜类商品自动定价与补货决策.pdf"
    txt_path = script_dir / "c228_clean.txt"
    
    # Check PDF exists
    if not pdf_path.exists():
        print(f"[ERROR] PDF not found: {pdf_path}")
        sys.exit(1)
    
    print(f"Input:  {pdf_path}")
    print(f"Output: {txt_path}")
    print(f"PDF size: {pdf_path.stat().st_size:,} bytes\n")
    
    # Try conversion methods in order of preference
    methods = [
        ("PyMuPDF (fitz)", convert_with_pymupdf),
        ("pdfplumber", convert_with_pdfplumber),
        ("pypdf/PyPDF2", convert_with_pypdf),
    ]
    
    for name, convert_func in methods:
        if convert_func(pdf_path, txt_path):
            # Verify Chinese text is readable
            sample = txt_path.read_text(encoding="utf-8")[:500]
            if "蔬菜" in sample or "定价" in sample or "价格" in sample:
                print("\n[OK] Chinese text verified in output")
            else:
                print("\n[WARNING] Chinese text may not be correctly extracted")
            return
    
    # No method succeeded
    print("\n[ERROR] Failed: No PDF extraction library found.")
    print("\nPlease install one of:")
    print("  pip install PyMuPDF        # Recommended")
    print("  pip install pdfplumber     # Good for tables")
    print("  pip install pypdf          # Fallback")
    sys.exit(1)


if __name__ == "__main__":
    main()
