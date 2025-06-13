#!/usr/bin/env python3
"""
PDF Quiz Generator MVP
Converts PDF documents into multiple-choice quizzes using AI
"""

from dotenv import load_dotenv
import os
from openai import OpenAI
from pinecone import Pinecone
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Add this after your imports, before the class definition
LANGUAGE_PROMPTS = {
    "english": """You are an expert quiz creator. Generate multiple-choice questions from the following content.

    Requirements:
    - Create {num_questions} questions
    - Each question should have 4 options (A, B, C, D)
    - Only one correct answer per question
    - Focus on key concepts and important facts
    - Vary difficulty levels
    - Include the correct answer

    Content:
    {content}

    Format your response as:
    Question 1: [question text]
    A) [option]
    B) [option]
    C) [option]
    D) [option]
    Correct Answer: [letter]

    [Continue for all questions]""",

    "japanese": """あなたは専門のクイズ作成者です。以下の英語の内容を理解し、完全に日本語で選択式問題を作成してください。

        重要: 問題文、選択肢、番号、すべてを日本語で作成してください。英語は一切使用しないでください。

        要件:
        - {num_questions}問の問題を作成
        - 各問題に4つの選択肢（ア、イ、ウ、エ）
        - 各問題につき正解は1つのみ
        - 重要な概念と事実に焦点を当てる
        - 難易度を変える
        - 正解を含める
        - 内容を日本語に翻訳して理解しやすい問題を作成

        内容:
        {content}

        以下の形式で回答してください（すべて日本語で）:
        問題１: [日本語の問題文]
        ア) [日本語の選択肢]
        イ) [日本語の選択肢]
        ウ) [日本語の選択肢]
        エ) [日本語の選択肢]
        正解: [ア/イ/ウ/エ]

        [続けて全ての問題]"""
}

# Add this after LANGUAGE_PROMPTS definition
SUPPORTED_LANGUAGES = ["english", "japanese"]

def validate_languages(languages):
    """Validate and filter supported languages"""
    if not languages:
        return ["english"]  # Default fallback
    
    # Filter out invalid languages
    valid_languages = [lang.lower() for lang in languages if lang.lower() in SUPPORTED_LANGUAGES]
    
    # If no valid languages, fallback to default
    if not valid_languages:
        print("⚠️  No valid languages found, defaulting to English")
        return ["english"]
    
    print(f"✅ Valid languages: {valid_languages}")
    return valid_languages

class PDFQuizGenerator:
    def extract_pdf_text(self, pdf_path):
        """Extract text from PDF using LangChain's PyPDFLoader"""
        try:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            print(f"📄 Processing PDF with {len(documents)} pages...")
            
            # Combine all pages into single text string
            full_text = ""
            for doc in documents:
                full_text += doc.page_content + "\n"
            
            print(f"✅ Extracted {len(full_text)} characters")
            return full_text.strip()
            
        except Exception as e:
            print(f"❌ PDF extraction error: {e}")
            return None

    def create_chunks(self, text):
        """Create chunks using LangChain's RecursiveCharacterTextSplitter"""
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=300,
                chunk_overlap=50,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            
            chunks = text_splitter.split_text(text)
            print(f"✅ Created {len(chunks)} chunks")
            
            return chunks
            
        except Exception as e:
            print(f"❌ Chunking error: {e}")
            return []

    def store_in_pinecone(self, chunks, embeddings=None):
        """Store chunks in Pinecone using LangChain's PineconeVectorStore"""
        try:
            # Create embeddings model
            embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002")
            
            # Get index name from environment variable
            index_name = os.getenv('PINECONE_INDEX_NAME')
            
            # Create vector store and add documents
            vector_store = PineconeVectorStore.from_texts(
                texts=chunks,
                embedding=embeddings_model,
                index_name=index_name
            )
            
            print(f"✅ Stored {len(chunks)} chunks in Pinecone")
            return vector_store
            
        except Exception as e:
            print(f"❌ Pinecone storage error: {e}")
            return None

    def retrieve_relevant_content(self, query="important concepts key facts", top_k=3, source_filter=None):
        """Retrieve most relevant chunks for quiz generation using LangChain"""
        try:
            # Create embeddings model
            embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002")
            
            # Get index name from environment variable
            index_name = os.getenv('PINECONE_INDEX_NAME')
            
            # Create vector store
            vector_store = PineconeVectorStore(
                index_name=index_name,
                embedding=embeddings_model
            )
            
            # Create retriever
            retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
            
            # Query for relevant content
            relevant_docs = retriever.invoke(query)
            
            # Extract text content
            relevant_chunks = [doc.page_content for doc in relevant_docs]
            
            print(f"✅ Retrieved {len(relevant_chunks)} relevant chunks")
            return relevant_chunks
            
        except Exception as e:
            print(f"❌ Content retrieval error: {e}")
            return []

    def generate_quiz(self, content_chunks, num_questions=5, languages=["japanese"]):
        """Generate quiz from content chunks using LangChain for multiple languages"""
        try:
            # Validate languages
            languages = validate_languages(languages)
            combined_content = "\n\n".join(content_chunks)
            
            # Create ChatOpenAI instance
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
            
            results = {}
            
            for language in languages:
                print(f"🔄 Generating {language} quiz with {num_questions} questions...")

                # Get the prompt template for this language
                prompt_template = PromptTemplate(
                    input_variables=["content", "num_questions"],
                    template=LANGUAGE_PROMPTS[language]
                )
                
                # Format prompt
                formatted_prompt = prompt_template.format(
                    content=combined_content,
                    num_questions=num_questions
                )
                
                # Generate quiz for this language
                response = llm.invoke(formatted_prompt)
                results[language] = response.content
                
                print(f"✅ {language} quiz generated successfully")
            
            return {"quizzes": results}
            
        except Exception as e:
            print(f"❌ Quiz generation error: {e}")
            return None

    def process_pdf_to_quiz(self, pdf_path, num_questions=5, languages=["japanese"]):
        """Complete pipeline: PDF to Quiz using LangChain"""
        print("=" * 50)
        print("🚀 PDF QUIZ GENERATOR MVP (LangChain)")
        print("=" * 50)
        
        # Extract PDF text
        text = self.extract_pdf_text(pdf_path)
        if not text:
            return None
        
        # Create chunks
        chunks = self.create_chunks(text)
        if not chunks:
            return None
        
        # Store in Pinecone
        vector_store = self.store_in_pinecone(chunks)
        if not vector_store:
            return None
        
        # Retrieve relevant content
        relevant_content = self.retrieve_relevant_content()
        if not relevant_content:
            print("❌ No relevant content found")
            return None
        
        # Generate quiz
        quiz = self.generate_quiz(relevant_content, num_questions, languages=languages)
        
        return quiz

# Test complete pipeline
if __name__ == "__main__":
    generator = PDFQuizGenerator()
    quiz = generator.process_pdf_to_quiz("sample.pdf", num_questions=2, languages=["japanese"])
    
    if quiz:
        print("\n" + "=" * 50)
        print("📝 COMPLETE LANGCHAIN QUIZ")
        print("=" * 50)
        print(quiz)
        print("\n" + "=" * 50)
        print("✅ LangChain integration complete!")
    else:
        print("❌ Failed to generate quiz")