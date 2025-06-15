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


# Add this helper function to generate unique namespaces
def generate_namespace(pdf_path):
    """Generate a unique namespace for each PDF document"""
    import hashlib
    import time
    
    # Get file name without path
    filename = os.path.basename(pdf_path)
    
    # Create hash of filename + timestamp for uniqueness
    timestamp = str(int(time.time()))
    content = f"{filename}_{timestamp}"
    namespace_hash = hashlib.md5(content.encode()).hexdigest()[:12]
    
    # Create readable namespace
    namespace = f"pdf_{namespace_hash}"
    print(f"📁 Using namespace: {namespace} for {filename}")
    return namespace

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

    # Updated store_in_pinecone method
    def store_in_pinecone(self, chunks, embeddings=None, namespace=None):
        """Store chunks in Pinecone using LangChain's PineconeVectorStore with namespace isolation"""
        try:
            # Create embeddings model
            embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002")
            
            # Get index name from environment variable
            index_name = os.getenv('PINECONE_INDEX_NAME')
            
            # Create vector store and add documents with namespace
            vector_store = PineconeVectorStore.from_texts(
                texts=chunks,
                embedding=embeddings_model,
                index_name=index_name,
                namespace=namespace  # 🔑 KEY CHANGE: Add namespace isolation
            )
            
            print(f"✅ Stored {len(chunks)} chunks in Pinecone namespace: {namespace}")
            return vector_store
        
        except Exception as e:
            print(f"❌ Pinecone storage error: {e}")
            return None

    # Update your retrieve_relevant_content method (if you're still using it)
    def retrieve_relevant_content(self, query="important concepts key facts", top_k=3, namespace=None):
        """Retrieve most relevant chunks for quiz generation using LangChain with namespace isolation"""
        try:
            # 🔑 FIX: Use text-embedding-ada-002 to match your Pinecone index
            embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002")
            
            # Get index name from environment variable
            index_name = os.getenv('PINECONE_INDEX_NAME')
            
            # Create vector store with namespace
            vector_store = PineconeVectorStore(
                index_name=index_name,
                embedding=embeddings_model,
                namespace=namespace
            )
            
            # Create retriever
            retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
            
            # Query for relevant content
            relevant_docs = retriever.invoke(query)
            
            # Extract text content
            relevant_chunks = [doc.page_content for doc in relevant_docs]
            
            print(f"✅ Retrieved {len(relevant_chunks)} relevant chunks from namespace: {namespace}")
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

    # Replace your process_pdf_to_quiz method with this debug version
    def process_pdf_to_quiz(self, pdf_path, num_questions=5, languages=["japanese"]):
        """Complete pipeline: PDF to Quiz using LangChain with namespace isolation"""
        print("=" * 50)
        print("🚀 PDF QUIZ GENERATOR MVP (LangChain)")
        print("=" * 50)
        
        # Generate unique namespace for this PDF
        namespace = generate_namespace(pdf_path)
        
        # Extract PDF text
        text = self.extract_pdf_text(pdf_path)
        if not text:
            return None
        
        # Create chunks
        chunks = self.create_chunks(text)
        if not chunks:
            return None
        
        # 🔧 DEBUG: Show what we're storing
        print(f"🔧 DEBUG: First chunk preview: {chunks[0][:100]}...")
        
        # Store in Pinecone with namespace
        vector_store = self.store_in_pinecone(chunks, namespace=namespace)
        if not vector_store:
            return None
        
        # 🔧 DEBUG: Try multiple retrieval approaches
        try:
            print("🔍 Testing different retrieval methods...")
            
            # Method 1: Try similarity search with different queries
            test_queries = [
                "important concepts key facts",
                "main points",
                chunks[0][:30],  # Use actual content
                "information",
                ""  # Empty query
            ]
            
            relevant_content = []
            
            for i, query in enumerate(test_queries):
                print(f"🔧 Test {i+1}: Query '{query[:20]}...'")
                try:
                    # Try direct similarity search
                    results = vector_store.similarity_search(query, k=3)
                    if results:
                        relevant_content = [doc.page_content for doc in results]
                        print(f"   ✅ SUCCESS: Found {len(relevant_content)} chunks")
                        break
                    else:
                        print(f"   ❌ No results")
                except Exception as e:
                    print(f"   ❌ Error: {e}")
            
            # Method 2: If nothing worked, try without query
            if not relevant_content:
                print("🔧 Trying retrieval without specific query...")
                try:
                    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
                    docs = retriever.invoke("")
                    relevant_content = [doc.page_content for doc in docs]
                    print(f"✅ Fallback retrieval found {len(relevant_content)} chunks")
                except Exception as e:
                    print(f"❌ Fallback failed: {e}")
            
            # Method 3: If still nothing, just use first few chunks
            if not relevant_content:
                print("🔧 Using first 3 chunks as fallback...")
                relevant_content = chunks[:3]
                print(f"✅ Using {len(relevant_content)} original chunks")
            
            if not relevant_content:
                print("❌ All retrieval methods failed")
                return None
                
        except Exception as e:
            print(f"❌ Content retrieval error: {e}")
            return None
        
        # 🔧 DEBUG: Show what we're sending to quiz generation
        print(f"🔧 DEBUG: Content for quiz generation:")
        for i, content in enumerate(relevant_content[:2]):
            print(f"   Chunk {i+1}: {content[:100]}...")
        
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