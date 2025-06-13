from PyPDF2 import PdfReader

def extract_pdf_text(pdf_path):
    """Extract text from a PDF file"""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        
        print(f"PDF has {len(reader.pages)} pages")
        
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            text += page_text + "\n"
            print(f"Extracted text from page {page_num + 1}")
        
        print(f"Total characters extracted: {len(text)}")
        return text.strip()
        
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None

if __name__ == "__main__":
    # Test with a sample PDF file
    pdf_path = "sample.pdf"  # We'll create this next
    text = extract_pdf_text(pdf_path)
    
    if text:
        print("\nFirst 200 characters:")
        print(text[:200])
    else:
        print("No text extracted or file not found")