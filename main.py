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
import hashlib
import time

# Load environment variables
load_dotenv()

# Add this after  imports, before the class definition
LANGUAGE_PROMPTS = {
    "english": """You are an expert quiz creator. Generate multiple-choice questions from the following content.

    Requirements:
    - Create {num_questions} questions
    - Each question should have 4 options (A, B, C, D)
    - Only one correct answer per question
    - Focus on key concepts and important facts
    - Vary difficulty levels

    Content:
    {content}

    IMPORTANT: Return your response as valid JSON in this exact format:
    {{
    "questions": [
        {{
        "id": 1,
        "question": "Your question text here?",
        "options": {{
            "A": "Option A text",
            "B": "Option B text", 
            "C": "Option C text",
            "D": "Option D text"
        }},
        "correct_answer": "A"
        }}
    ]
    }}

    Return ONLY the JSON, no other text.""",

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
    
    # Get file name without path
    filename = os.path.basename(pdf_path)
    
    # Create hash of filename + timestamp for uniqueness
    timestamp = str(int(time.time()))
    content = f"{filename}_{timestamp}"
    namespace_hash = hashlib.md5(content.encode()).hexdigest()[:12]
    
    # Create readable namespace
    namespace = f"pdf_{namespace_hash}"
    return namespace

def validate_languages(languages):
    """Validate and filter supported languages"""
    if not languages:
        return ["english"]  # Default fallback
    
    # Filter out invalid languages
    valid_languages = [lang.lower() for lang in languages if lang.lower() in SUPPORTED_LANGUAGES]
    
    # If no valid languages, fallback to default
    if not valid_languages:
        return ["english"]
    
    return valid_languages

class PDFQuizGenerator:
    def extract_pdf_text(self, pdf_path):
        """Extract text from PDF using LangChain's PyPDFLoader"""
        try:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            # Combine all pages into single text string
            full_text = ""
            for doc in documents:
                full_text += doc.page_content + "\n"
            
            return full_text.strip()
            
        except Exception as e:
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
            
            return chunks
            
        except Exception as e:
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
                namespace=namespace  # KEY CHANGE: Add namespace isolation
            )
            
            return vector_store
        
        except Exception as e:
            return None

    # Update retrieve_relevant_content method (if you're still using it)
    def retrieve_relevant_content(self, query="important concepts key facts", top_k=3, namespace=None):
        """Retrieve most relevant chunks for quiz generation using LangChain with namespace isolation"""
        try:
            # FIX: Use text-embedding-ada-002 to match  Pinecone index
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
            
            return relevant_chunks
            
        except Exception as e:
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

                # Parse JSON response for English
                if language == "english":
                    import json
                    try:
                        results[language] = json.loads(response.content)
                    except:
                        results[language] = response.content  # Fallback to original text
                else:
                    results[language] = response.content
            
            return {"quizzes": results}
            
        except Exception as e:
            return None

    def process_pdf_to_quiz(self, pdf_path, num_questions=5, languages=["japanese"]):
        """Complete pipeline: PDF to Quiz using LangChain with namespace isolation"""
        
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
        
        # Store in Pinecone with namespace
        vector_store = self.store_in_pinecone(chunks, namespace=namespace)
        if not vector_store:
            return None
        
        # Try multiple retrieval approaches
        try:
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
                try:
                    # Try direct similarity search
                    results = vector_store.similarity_search(query, k=3)
                    if results:
                        relevant_content = [doc.page_content for doc in results]
                        break
                except Exception as e:
                    continue
            
            # Method 2: If nothing worked, try without query
            if not relevant_content:
                try:
                    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
                    docs = retriever.invoke("")
                    relevant_content = [doc.page_content for doc in docs]
                except Exception as e:
                    pass
            
            # Method 3: If still nothing, just use first few chunks
            if not relevant_content:
                relevant_content = chunks[:3]
            
            if not relevant_content:
                return None
                
        except Exception as e:
            return None
        
        # Generate quiz
        quiz = self.generate_quiz(relevant_content, num_questions, languages=languages)
        
        return quiz

    def analyze_quiz_performance(self, quiz_results, pdf_content):
        """
        Analyze user's quiz performance and provide feedback
        """
        try:
            # Set up OpenAI client
            llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.3,
                api_key=os.getenv("OPENAI_API_KEY")
            )
            
            # Create analysis prompt
            analysis_prompt = f"""You are an educational analyst. Analyze this student's quiz performance and provide feedback.

                PDF Content Context:
                {pdf_content[:2000]}...

                Quiz Results:
                Topic: {quiz_results['topic']}
                Total Questions: {quiz_results['total_questions']}
                Score: {quiz_results['score']}

                Student Answers:
                {quiz_results['user_answers']}

                Please provide analysis in this JSON format:
                {{
                "overall_performance": "Brief assessment of performance",
                "strong_areas": ["list", "of", "strong", "topics"],
                "weak_areas": ["list", "of", "weak", "topics"],
                "recommendations": ["specific", "study", "suggestions"]
                }}

                Focus on identifying knowledge gaps and providing actionable study recommendations."""

            # Get analysis from OpenAI
            response = llm.invoke(analysis_prompt)
            
            # Try to parse JSON response
            import json
            try:
                analysis = json.loads(response.content)
                return analysis
            except:
                # Fallback to text response if JSON parsing fails
                return {"analysis_text": response.content}
                
        except Exception as e:
            raise Exception(f"Analysis failed: {str(e)}")