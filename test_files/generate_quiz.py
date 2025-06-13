from dotenv import load_dotenv
import os
from pinecone import Pinecone
from openai import OpenAI

load_dotenv()

def retrieve_relevant_chunks(query_text, top_k=3):
    """Retrieve most relevant chunks for quiz generation"""
    # Generate query embedding
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    query_response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=[query_text]
    )
    query_embedding = query_response.data[0].embedding
    
    # Search Pinecone (without filter for now)
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    index = pc.Index(os.getenv('PINECONE_INDEX_NAME'))
    
    search_response = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    
    # Extract text from matches
    relevant_texts = []
    for match in search_response['matches']:
        if 'sample.pdf' in match['id']:  # Only get our PDF chunks
            text = match['metadata']['text']
            score = match['score']
            relevant_texts.append(f"Content (relevance: {score:.3f}):\n{text}\n")
    
    return relevant_texts

def generate_quiz(content_chunks):
    """Generate quiz questions from content chunks"""
    try:
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Combine chunks into single content
        combined_content = "\n".join(content_chunks)
        
        prompt = f"""You are an expert quiz creator. Generate multiple-choice questions from the following content.

Requirements:
- Create 4-5 questions
- Each question should have 4 options (A, B, C, D)
- Only one correct answer per question
- Focus on key concepts and important facts
- Vary difficulty levels
- Include the correct answer

Content:
{combined_content}

Format your response as:
Question 1: [question text]
A) [option]
B) [option]
C) [option]
D) [option]
Correct Answer: [letter]

[Continue for all questions]"""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
            temperature=0.3
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"Quiz generation error: {e}")
        return None

if __name__ == "__main__":
    print("=== PDF Quiz Generator ===")
    
    # Retrieve relevant content
    print("Retrieving relevant content...")
    relevant_chunks = retrieve_relevant_chunks("important concepts key facts main principles")
    
    if relevant_chunks:
        print(f"Retrieved {len(relevant_chunks)} relevant chunks")
        
        # Generate quiz
        print("Generating quiz...")
        quiz = generate_quiz(relevant_chunks)
        
        if quiz:
            print("\n=== GENERATED QUIZ ===")
            print(quiz)
        else:
            print("Failed to generate quiz")
    else:
        print("No relevant content found")