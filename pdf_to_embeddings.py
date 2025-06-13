from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
from openai import OpenAI

# Load environment variables
load_dotenv()

def extract_pdf_text(pdf_path):
    """Extract text from PDF"""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        
        for page in reader.pages:
            text += page.extract_text() + "\n"
        
        return text.strip()
    except Exception as e:
        print(f"PDF extraction error: {e}")
        return None

def create_chunks(text, chunk_size=250, overlap=50):
    """Split text into chunks"""
    words = text.split()
    chunks = []
    
    start = 0
    chunk_id = 0
    
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)
        
        chunks.append({
            "id": f"pdf_chunk_{chunk_id:03d}",
            "text": chunk_text,
            "word_count": len(chunk_words)
        })
        
        start = end - overlap
        chunk_id += 1
        
        if end >= len(words):
            break
    
    return chunks

def generate_embeddings_for_chunks(chunks):
    """Generate embeddings for all chunks"""
    try:
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Extract just the text from chunks
        texts = [chunk["text"] for chunk in chunks]
        
        print(f"Generating embeddings for {len(texts)} chunks...")
        
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=texts
        )
        
        embeddings = [data.embedding for data in response.data]
        
        # Add embeddings back to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = embedding
        
        return chunks
        
    except Exception as e:
        print(f"Embedding generation error: {e}")
        return None

if __name__ == "__main__":
    # Process our sample PDF
    print("Processing sample.pdf...")
    
    # Step 1: Extract PDF text
    text = extract_pdf_text("sample.pdf")
    if not text:
        print("Failed to extract PDF text")
        exit()
    
    print(f"Extracted {len(text)} characters")
    
    # Step 2: Create chunks
    chunks = create_chunks(text)
    print(f"Created {len(chunks)} chunks")
    
    # Step 3: Generate embeddings
    chunks_with_embeddings = generate_embeddings_for_chunks(chunks)
    if chunks_with_embeddings:
        print(f"Generated embeddings for {len(chunks_with_embeddings)} chunks")
        print(f"First chunk: {chunks_with_embeddings[0]['text'][:100]}...")
        print(f"Embedding dimension: {len(chunks_with_embeddings[0]['embedding'])}")
    else:
        print("Failed to generate embeddings")