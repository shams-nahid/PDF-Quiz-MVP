#!/usr/bin/env python3
"""
PDF Quiz Generator MVP
Converts PDF documents into multiple-choice quizzes using AI
"""

from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
from openai import OpenAI
from pinecone import Pinecone

# Load environment variables
load_dotenv()

class PDFQuizGenerator:
    def __init__(self):
        """Initialize the PDF Quiz Generator"""
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        self.index = self.pc.Index(os.getenv('PINECONE_INDEX_NAME'))
        
    def extract_pdf_text(self, pdf_path):
        """Extract text from PDF file"""
        try:
            reader = PdfReader(pdf_path)
            text = ""
            print(f"📄 Processing PDF with {len(reader.pages)} pages...")
            
            for page in reader.pages:
                text += page.extract_text() + "\n"
            
            print(f"✅ Extracted {len(text)} characters")
            return text.strip()
            
        except Exception as e:
            print(f"❌ PDF extraction error: {e}")
            return None
    
    def create_chunks(self, text, chunk_size=250, overlap=50):
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        start = 0
        chunk_id = 0
        
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)
            
            chunks.append({
                "id": f"chunk_{chunk_id:03d}",
                "text": chunk_text
            })
            
            start = end - overlap
            chunk_id += 1
            
            if end >= len(words):
                break
        
        print(f"✅ Created {len(chunks)} text chunks")
        return chunks
    
    def generate_embeddings(self, chunks):
        """Generate embeddings for text chunks"""
        try:
            texts = [chunk["text"] for chunk in chunks]
            
            print(f"🔄 Generating embeddings...")
            response = self.openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=texts
            )
            
            embeddings = [data.embedding for data in response.data]
            
            for chunk, embedding in zip(chunks, embeddings):
                chunk["embedding"] = embedding
            
            print(f"✅ Generated embeddings for {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            print(f"❌ Embedding generation error: {e}")
            return None
    
    def store_in_pinecone(self, chunks, source_name):
        """Store chunks with embeddings in Pinecone"""
        try:
            vectors = []
            for chunk in chunks:
                vector = {
                    "id": f"{source_name}_{chunk['id']}",
                    "values": chunk["embedding"],
                    "metadata": {
                        "text": chunk["text"],
                        "source": source_name
                    }
                }
                vectors.append(vector)
            
            upsert_response = self.index.upsert(vectors=vectors)
            print(f"✅ Stored {upsert_response['upserted_count']} chunks in Pinecone")
            return True
            
        except Exception as e:
            print(f"❌ Pinecone storage error: {e}")
            return False
    
    def retrieve_relevant_content(self, query="important concepts key facts", top_k=3, source_filter=None):
        """Retrieve most relevant chunks for quiz generation"""
        try:
            # Generate query embedding
            query_response = self.openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=[query]
            )
            query_embedding = query_response.data[0].embedding
            
            # Search Pinecone
            search_response = self.index.query(
                vector=query_embedding,
                top_k=top_k * 2,  # Get more to filter
                include_metadata=True
            )
            
            # Filter results by source if specified
            relevant_chunks = []
            for match in search_response['matches']:
                if source_filter and source_filter not in match['id']:
                    continue
                if len(relevant_chunks) >= top_k:
                    break
                relevant_chunks.append(match['metadata']['text'])
            
            print(f"✅ Retrieved {len(relevant_chunks)} relevant chunks")
            return relevant_chunks
            
        except Exception as e:
            print(f"❌ Content retrieval error: {e}")
            return []
    
    def generate_quiz(self, content_chunks, num_questions=5):
        """Generate quiz from content chunks"""
        try:
            combined_content = "\n\n".join(content_chunks)
            
            prompt = f"""You are an expert quiz creator. Generate multiple-choice questions from the following content.

Requirements:
- Create {num_questions} questions
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

            print(f"🔄 Generating {num_questions} quiz questions...")
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.3
            )
            
            print(f"✅ Quiz generated successfully")
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"❌ Quiz generation error: {e}")
            return None
    
    def process_pdf_to_quiz(self, pdf_path, num_questions=5):
        """Complete pipeline: PDF to Quiz"""
        print("=" * 50)
        print("🚀 PDF QUIZ GENERATOR MVP")
        print("=" * 50)
        
        # Extract PDF text
        text = self.extract_pdf_text(pdf_path)
        if not text:
            return None
        
        # Create chunks
        chunks = self.create_chunks(text)
        
        # Generate embeddings
        chunks_with_embeddings = self.generate_embeddings(chunks)
        if not chunks_with_embeddings:
            return None
        
        # Store in Pinecone
        source_name = os.path.splitext(os.path.basename(pdf_path))[0]
        stored = self.store_in_pinecone(chunks_with_embeddings, source_name)
        if not stored:
            return None
        
        # Retrieve relevant content
        relevant_content = self.retrieve_relevant_content(source_filter=source_name)
        if not relevant_content:
            print("❌ No relevant content found")
            return None
        
        # Generate quiz
        quiz = self.generate_quiz(relevant_content, num_questions)
        
        return quiz

def main():
    """Main function"""
    # Initialize generator
    generator = PDFQuizGenerator()
    
    # Process PDF and generate quiz
    pdf_path = "sample.pdf"
    quiz = generator.process_pdf_to_quiz(pdf_path, num_questions=5)
    
    if quiz:
        print("\n" + "=" * 50)
        print("📝 GENERATED QUIZ")
        print("=" * 50)
        print(quiz)
        print("\n" + "=" * 50)
        print("✅ Quiz generation complete!")
    else:
        print("❌ Failed to generate quiz")

if __name__ == "__main__":
    main()