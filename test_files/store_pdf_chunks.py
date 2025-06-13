from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
from openai import OpenAI
from pinecone import Pinecone

# Load environment variables
load_dotenv()

def extract_and_process_pdf(pdf_path):
    """Extract PDF, create chunks, and generate embeddings"""
    # Extract text
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    
    # Create chunks
    words = text.split()
    chunks = []
    chunk_size = 250
    overlap = 50
    start = 0
    chunk_id = 0
    
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)
        
        chunks.append({
            "id": f"pdf_chunk_{chunk_id:03d}",
            "text": chunk_text
        })
        
        start = end - overlap
        chunk_id += 1
        if end >= len(words):
            break
    
    # Generate embeddings
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    texts = [chunk["text"] for chunk in chunks]
    
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=texts
    )
    
    embeddings = [data.embedding for data in response.data]
    
    # Combine chunks with embeddings
    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding
    
    return chunks

def store_in_pinecone(chunks, pdf_name):
    """Store chunks in Pinecone"""
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    index = pc.Index(os.getenv('PINECONE_INDEX_NAME'))
    
    # Prepare vectors
    vectors = []
    for chunk in chunks:
        vector = {
            "id": f"{pdf_name}_{chunk['id']}",
            "values": chunk["embedding"],
            "metadata": {
                "text": chunk["text"],
                "source": pdf_name,
                "chunk_id": chunk["id"]
            }
        }
        vectors.append(vector)
    
    # Store in Pinecone
    upsert_response = index.upsert(vectors=vectors)
    print(f"Stored {upsert_response['upserted_count']} chunks in Pinecone")
    
    return True

def test_retrieval(query_text):
    """Test retrieving relevant chunks"""
    # Generate query embedding
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    query_response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=[query_text]
    )
    query_embedding = query_response.data[0].embedding
    
    # Search Pinecone
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    index = pc.Index(os.getenv('PINECONE_INDEX_NAME'))
    
    search_response = index.query(
        vector=query_embedding,
        top_k=3,
        include_metadata=True,
        filter={"source": "sample.pdf"}  # Only get our PDF chunks
    )
    
    print(f"\nFound {len(search_response['matches'])} relevant chunks for: '{query_text}'")
    for i, match in enumerate(search_response['matches']):
        print(f"\n{i+1}. Score: {match['score']:.4f}")
        print(f"Text: {match['metadata']['text'][:150]}...")
    
    return search_response['matches']

if __name__ == "__main__":
    print("Processing and storing PDF chunks...")
    
    # Process PDF
    chunks = extract_and_process_pdf("sample.pdf")
    print(f"Created {len(chunks)} chunks")
    
    # Store in Pinecone
    store_in_pinecone(chunks, "sample.pdf")
    
    # Test retrieval with different queries
    test_queries = [
        "project setup environment requirements",
        "quiz generation questions",
        "testing validation"
    ]
    
    for query in test_queries:
        test_retrieval(query)