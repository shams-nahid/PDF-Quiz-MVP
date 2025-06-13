from dotenv import load_dotenv
import os
from openai import OpenAI

# Load environment variables
load_dotenv()

def generate_embeddings(texts):
    """Generate embeddings for a list of texts"""
    try:
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        print(f"Generating embeddings for {len(texts)} texts...")
        
        # Generate embeddings
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=texts
        )
        
        embeddings = [data.embedding for data in response.data]
        
        print(f"Generated {len(embeddings)} embeddings")
        print(f"Embedding dimension: {len(embeddings[0])}")
        
        return embeddings
        
    except Exception as e:
        print(f"Embedding generation error: {e}")
        return None

if __name__ == "__main__":
    # Test with sample texts
    sample_texts = [
        "This is the first test document about machine learning.",
        "This is the second test document about artificial intelligence.",
        "This is the third test document about natural language processing."
    ]
    
    embeddings = generate_embeddings(sample_texts)
    
    if embeddings:
        print(f"\nFirst embedding preview: {embeddings[0][:5]}...")
        print("Embedding generation successful!")
    else:
        print("Embedding generation failed!")