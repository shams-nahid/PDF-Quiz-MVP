from dotenv import load_dotenv
import os
from pinecone import Pinecone

# Load environment variables
load_dotenv()

def test_pinecone_connection():
    """Test basic Pinecone connection"""
    try:
        # Initialize Pinecone
        pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        
        print("Pinecone client created successfully!")
        
        # List existing indexes
        indexes = pc.list_indexes()
        print(f"Available indexes: {[idx.name for idx in indexes]}")
        
        # Check if our target index exists
        index_name = os.getenv('PINECONE_INDEX_NAME')
        print(f"Looking for index: {index_name}")
        
        index_exists = any(idx.name == index_name for idx in indexes)
        print(f"Target index exists: {index_exists}")
        
        return True
        
    except Exception as e:
        print(f"Pinecone connection error: {e}")
        return False

if __name__ == "__main__":
    test_pinecone_connection()