from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Test reading the variables
openai_key = os.getenv('OPENAI_API_KEY')
pinecone_key = os.getenv('PINECONE_API_KEY')
pinecone_env = os.getenv('PINECONE_ENVIRONMENT')
index_name = os.getenv('PINECONE_INDEX_NAME')

print("Environment variables loaded successfully!")
print(f"OpenAI Key: {openai_key}")
print(f"Pinecone Key: {pinecone_key}")
print(f"Pinecone Environment: {pinecone_env}")
print(f"Index Name: {index_name}")