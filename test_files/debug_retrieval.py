from dotenv import load_dotenv
import os
from pinecone import Pinecone
from openai import OpenAI

load_dotenv()

def debug_pinecone_content():
    """Check what's actually in Pinecone"""
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    index = pc.Index(os.getenv('PINECONE_INDEX_NAME'))
    
    # Get index stats
    stats = index.describe_index_stats()
    print("Index Stats:")
    print(f"Total vectors: {stats['total_vector_count']}")
    print(f"Namespaces: {stats.get('namespaces', {})}")
    
    # Try query without any filters
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    query_response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=["project setup"]
    )
    query_embedding = query_response.data[0].embedding
    
    print("\n--- Query WITHOUT filter ---")
    search_response = index.query(
        vector=query_embedding,
        top_k=5,
        include_metadata=True
    )
    
    print(f"Found {len(search_response['matches'])} results")
    for i, match in enumerate(search_response['matches']):
        print(f"\n{i+1}. ID: {match['id']}")
        print(f"Score: {match['score']:.4f}")
        if 'metadata' in match and match['metadata']:
            source = match['metadata'].get('source', 'No source')
            print(f"Source: {source}")
            text_preview = match['metadata'].get('text', 'No text')[:100]
            print(f"Text: {text_preview}...")
        else:
            print("No metadata found")

if __name__ == "__main__":
    debug_pinecone_content()