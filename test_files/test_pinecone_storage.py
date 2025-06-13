from dotenv import load_dotenv
import os
from pinecone import Pinecone
from openai import OpenAI

# Load environment variables
load_dotenv()

def store_embeddings_in_pinecone(texts, embeddings):
    """Store embeddings with metadata in Pinecone"""
    try:
        # Initialize Pinecone
        pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        index = pc.Index(os.getenv('PINECONE_INDEX_NAME'))
        
        print(f"Storing {len(embeddings)} embeddings in Pinecone...")
        
        # Prepare vectors for upsert
        vectors = []
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            vector = {
                "id": f"test_chunk_{i:03d}",
                "values": embedding,
                "metadata": {
                    "text": text,
                    "source": "test_document",
                    "chunk_index": i
                }
            }
            vectors.append(vector)
        
        # Upsert vectors
        upsert_response = index.upsert(vectors=vectors)
        print(f"Upserted {upsert_response['upserted_count']} vectors")
        
        # Test retrieval
        print("\nTesting retrieval...")
        query_response = index.query(
            vector=embeddings[0],  # Use first embedding as query
            top_k=2,
            include_metadata=True
        )
        
        print(f"Retrieved {len(query_response['matches'])} matches")
        for match in query_response['matches']:
            print(f"ID: {match['id']}, Score: {match['score']:.4f}")
            print(f"Text: {match['metadata']['text'][:50]}...")
        
        return True
        
    except Exception as e:
        print(f"Pinecone storage error: {e}")
        return False

if __name__ == "__main__":
    # Generate sample embeddings
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    sample_texts = [
        "Machine learning is a subset of artificial intelligence.",
        "Neural networks are inspired by biological brain structures.",
        "Deep learning uses multiple layers to learn complex patterns."
    ]
    
    print("Generating embeddings...")
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=sample_texts
    )
    embeddings = [data.embedding for data in response.data]
    
    # Store in Pinecone
    success = store_embeddings_in_pinecone(sample_texts, embeddings)
    
    if success:
        print("\nPinecone storage and retrieval successful!")
    else:
        print("\nPinecone storage failed!")