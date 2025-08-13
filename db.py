import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

def get_database():
    connection_string = os.getenv("MONGODB_URI")
    client = MongoClient(connection_string)
    return client['quiz_app']

def test_connection():
    try:
        db = get_database()
        db.list_collection_names()
        print("Database connected successfully!")
        return True
    except Exception as e:
        print(f"Connection failed: {e}")
        return False