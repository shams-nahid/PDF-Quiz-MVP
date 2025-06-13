from dotenv import load_dotenv
import os
from openai import OpenAI

# Load environment variables
load_dotenv()

def test_openai_connection():
    """Test basic OpenAI API connection"""
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        print("OpenAI client created successfully!")
        
        # Test with a very simple completion
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say 'OpenAI API test successful'"}],
            max_tokens=20
        )
        
        result = response.choices[0].message.content
        print(f"API Response: {result}")
        return True
        
    except Exception as e:
        print(f"OpenAI API Error: {e}")
        return False

if __name__ == "__main__":
    test_openai_connection()