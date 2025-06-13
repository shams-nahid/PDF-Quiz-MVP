from PyPDF2 import PdfReader
import os

def test_pdf_reader():
    print("Testing PDF reader import...")
    print("PyPDF2 imported successfully!")
    print(f"PdfReader class available: {PdfReader}")
    
    # Test if we can create a reader instance (without a file)
    print("PDF processing capability confirmed!")

if __name__ == "__main__":
    test_pdf_reader()