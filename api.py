from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
import tempfile
import os
from main import PDFQuizGenerator
from typing import List
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from db import test_connection
from passlib.context import CryptContext
from pydantic import BaseModel
from db import get_database

class UserRegistration(BaseModel):
    name: str
    email: str
    password: str

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str):
    return pwd_context.hash(password)


# Create FastAPI app instance
app = FastAPI(title="PDF Quiz Generator API", version="1.0.0")
templates = Jinja2Templates(directory="templates")

# Configuration constants
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MIN_QUESTIONS = 1
MAX_QUESTIONS = 10
ALLOWED_FILE_TYPES = ['.pdf']

# Request model for accepting parameters (keeping for future use)
class QuizRequest(BaseModel):
    num_questions: int = 2  # Default to 2 if not provided

class QuizAnalysisRequest(BaseModel):
    quiz_results: dict
    pdf_content: str

@app.post("/generate-quiz")
async def generate_quiz(
    file: UploadFile = File(..., description="PDF file to generate quiz from"),
    num_questions: int = Form(2, description="Number of questions to generate (1-10)"),
    languages: List[str] = Form(["english", "japanese"], description="Languages for quiz generation (english, japanese)")
):
    """
    Generate quiz from uploaded PDF with robust error handling
    
    Parameters:
    - file: PDF file (max 10MB)
    - num_questions: Number of questions (1-10)
    
    Returns:
    - Generated quiz questions with metadata
    """
    temp_file_path = None
    
    try:
        # Input validation
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Validate file type
        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension not in ALLOWED_FILE_TYPES:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid file type. Only {', '.join(ALLOWED_FILE_TYPES)} files are allowed"
            )
        
        # Validate number of questions
        if not MIN_QUESTIONS <= num_questions <= MAX_QUESTIONS:
            raise HTTPException(
                status_code=400, 
                detail=f"Number of questions must be between {MIN_QUESTIONS} and {MAX_QUESTIONS}"
            )
        
        # Read file content and validate size
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400, 
                detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB"
            )
        
        # Create temporary file to save uploaded PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Initialize PDF quiz generator
        generator = PDFQuizGenerator()
        
        # Generate quiz
        quiz_result = generator.process_pdf_to_quiz(temp_file_path, num_questions=num_questions, languages=languages)
        
        if not quiz_result:
            raise HTTPException(
                status_code=500, 
                detail="Failed to generate quiz. The PDF might not contain sufficient text content."
            )
        
        return {
            "success": True,
            "message": "Quiz generated successfully!",
            "quiz_text": quiz_result,
            "metadata": {
                "source_filename": file.filename,
                "file_size_bytes": len(content),
                "num_questions_requested": num_questions,
                "languages_generated": languages,
                "content_type": file.content_type
            }
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions (validation errors)
        raise
        
    except Exception as e:
        # Handle unexpected errors
        error_msg = str(e)
        if "PDF" in error_msg or "parsing" in error_msg.lower():
            raise HTTPException(
                status_code=400, 
                detail="Failed to process PDF. The file might be corrupted or password-protected."
            )
        elif "embedding" in error_msg.lower() or "openai" in error_msg.lower():
            raise HTTPException(
                status_code=503, 
                detail="AI service temporarily unavailable. Please try again later."
            )
        elif "pinecone" in error_msg.lower():
            raise HTTPException(
                status_code=503, 
                detail="Vector database temporarily unavailable. Please try again later."
            )
        else:
            raise HTTPException(
                status_code=500, 
                detail=f"An unexpected error occurred while processing your request."
            )
    
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except OSError:
                pass  # File cleanup failed, but don't crash the response

@app.post("/generate-quiz-json")
async def generate_quiz_json(
    file: UploadFile = File(..., description="PDF file to generate quiz from"),
    num_questions: int = Form(2, description="Number of questions to generate (1-10)"),
    languages: List[str] = Form(["english", "japanese"], description="Languages for quiz generation (english, japanese)")
):
    """
    Generate quiz from uploaded PDF with robust error handling
    
    Parameters:
    - file: PDF file (max 10MB)
    - num_questions: Number of questions (1-10)
    
    Returns:
    - Generated quiz questions with metadata
    """
    temp_file_path = None
    
    try:
        # Input validation
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Validate file type
        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension not in ALLOWED_FILE_TYPES:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid file type. Only {', '.join(ALLOWED_FILE_TYPES)} files are allowed"
            )
        
        # Validate number of questions
        if not MIN_QUESTIONS <= num_questions <= MAX_QUESTIONS:
            raise HTTPException(
                status_code=400, 
                detail=f"Number of questions must be between {MIN_QUESTIONS} and {MAX_QUESTIONS}"
            )
        
        # Read file content and validate size
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400, 
                detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB"
            )
        
        # Create temporary file to save uploaded PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Initialize PDF quiz generator
        generator = PDFQuizGenerator()
        
        # Generate quiz
        quiz_result = generator.process_pdf_to_quiz(temp_file_path, num_questions=num_questions, languages=["english"])
        
        if not quiz_result:
            raise HTTPException(
                status_code=500, 
                detail="Failed to generate quiz. The PDF might not contain sufficient text content."
            )
        
        if quiz_result and "quizzes" in quiz_result and "english" in quiz_result["quizzes"]:
            return {
                "success": True,
                "questions": quiz_result["quizzes"]["english"]["questions"],
                "metadata": {
                    "source_filename": file.filename,
                    "file_size_bytes": len(content),
                    "num_questions_requested": num_questions,
                    "content_type": file.content_type
                }
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to generate quiz")
        
    except HTTPException:
        # Re-raise HTTP exceptions (validation errors)
        raise
        
    except Exception as e:
        # Handle unexpected errors
        error_msg = str(e)
        if "PDF" in error_msg or "parsing" in error_msg.lower():
            raise HTTPException(
                status_code=400, 
                detail="Failed to process PDF. The file might be corrupted or password-protected."
            )
        elif "embedding" in error_msg.lower() or "openai" in error_msg.lower():
            raise HTTPException(
                status_code=503, 
                detail="AI service temporarily unavailable. Please try again later."
            )
        elif "pinecone" in error_msg.lower():
            raise HTTPException(
                status_code=503, 
                detail="Vector database temporarily unavailable. Please try again later."
            )
        else:
            raise HTTPException(
                status_code=500, 
                detail=f"An unexpected error occurred while processing your request."
            )
    
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except OSError:
                pass  # File cleanup failed, but don't crash the response

@app.post("/analyze-quiz-performance")
async def analyze_quiz_performance(request: QuizAnalysisRequest):
    """
    Analyze user's quiz performance and provide insights
    """
    try:
        # Initialize the PDF quiz generator
        generator = PDFQuizGenerator()
        
        # Get analysis from the generator
        analysis = generator.analyze_quiz_performance(
            quiz_results=request.quiz_results,
            pdf_content=request.pdf_content
        )
        
        # Return the analysis
        return {
            "success": True,
            "analysis": analysis
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "PDF Quiz Generator API is running"}


@app.get("/test-db")
def test_database():
    success = test_connection()
    return {"database_connected": success}

@app.post("/register")
def register_user(user: UserRegistration):
    try:
        db = get_database()
        users_collection = db.users
        
        # Check if email already exists
        if users_collection.find_one({"email": user.email}):
            return {"success": False, "message": "Email already registered"}
        
        # Hash password and create user
        hashed_password = hash_password(user.password)
        
        user_data = {
            "name": user.name,
            "email": user.email,
            "password": hashed_password,
            "created_at": "2025-08-16"  # You can use datetime.now() later
        }
        
        result = users_collection.insert_one(user_data)
        
        return {
            "success": True, 
            "message": "User registered successfully",
            "user_id": str(result.inserted_id)
        }
        
    except Exception as e:
        return {"success": False, "message": f"Registration failed: {str(e)}"}
    
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
