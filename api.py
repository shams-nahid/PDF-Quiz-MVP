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
import shutil
from datetime import datetime
from bson import ObjectId
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import timedelta
from fastapi.middleware.cors import CORSMiddleware
import requests
import base64
from dotenv import load_dotenv

load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPO = os.getenv("GITHUB_REPO")

class UserRegistration(BaseModel):
    name: str
    email: str
    password: str

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str):
    return pwd_context.hash(password)


# Create FastAPI app instance
app = FastAPI(title="PDF Quiz Generator API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
templates = Jinja2Templates(directory="templates")

# Configuration constants
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MIN_QUESTIONS = 1
MAX_QUESTIONS = 10
ALLOWED_FILE_TYPES = ['.pdf']

# JWT Configuration
SECRET_KEY = "your-secret-key-here-make-it-long-and-random"  # Change this!
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def upload_to_github(content, filename, user_id):
    # Create path in repo
    file_path = f"uploads/{user_id}/{filename}"
    
    # GitHub API URL
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{file_path}"
    
    # Encode file content
    content_encoded = base64.b64encode(content).decode()
    
    # Upload
    response = requests.put(url, 
        headers={"Authorization": f"Bearer {GITHUB_TOKEN}"},
        json={
            "message": f"Upload {filename}",
            "content": content_encoded
        }
    )
    
    # Return the raw URL for downloading
    if response.status_code == 201:
        return f"https://raw.githubusercontent.com/{GITHUB_REPO}/main/{file_path}"
    else:
        raise Exception("Upload failed")

class UserLogin(BaseModel):
    email: str
    password: str

# Request model for accepting parameters (keeping for future use)
class QuizRequest(BaseModel):
    num_questions: int = 2  # Default to 2 if not provided

class QuizAnalysisRequest(BaseModel):
    quiz_id: str
    user_id: str
    user_answers: List[dict]

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
    user_id: str = Form(..., description="User ID who is uploading"),
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
        
        # Create permanent file storage
        db = get_database()
        upload_dir = f"uploads/{user_id}"
        os.makedirs(upload_dir, exist_ok=True)
        
        # Upload to GitHub for permanent storage
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        github_url = upload_to_github(content, filename, user_id)

        # Create temporary file for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name

        # Save PDF metadata to database
        pdf_data = {
            "user_id": ObjectId(user_id),
            "filename": filename,
            "original_name": file.filename,
            "file_path": github_url,
            "file_size": len(content),
            "uploaded_at": datetime.now(),
            "status": "processing"
        }

        pdf_result = db.pdfs.insert_one(pdf_data)
        pdf_id = pdf_result.inserted_id
        
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
            # Save quiz to database
            quiz_data = {
                "pdf_id": pdf_id,
                "user_id": ObjectId(user_id),
                "questions": quiz_result["quizzes"]["english"]["questions"],
                "quiz_settings": {
                    "total_questions": num_questions,
                    "language": "english",
                    "generated_at": datetime.now(),
                    "source_filename": file.filename
                }
            }
            
            quiz_save_result = db.quizzes.insert_one(quiz_data)
            quiz_id = quiz_save_result.inserted_id
            
            return {
                "success": True,
                "questions": quiz_result["quizzes"]["english"]["questions"],
                "quiz_id": str(quiz_id),
                "pdf_id": str(pdf_id),
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
        # Update PDF status to ready (don't delete the file)
        if 'pdf_id' in locals():
            try:
                db.pdfs.update_one(
                    {"_id": pdf_id}, 
                    {"$set": {"status": "ready"}}
                )
            except:
                pass

@app.post("/analyze-quiz-performance")
async def analyze_quiz_performance(request: QuizAnalysisRequest):
    """
    Analyze user's quiz performance and provide insights
    """
    try:
        db = get_database()
        
        # Fetch quiz from database
        quiz = db.quizzes.find_one({"_id": ObjectId(request.quiz_id)})
        if not quiz:
            raise HTTPException(status_code=404, detail="Quiz not found")
        
        # Fetch PDF for context
        pdf = db.pdfs.find_one({"_id": quiz["pdf_id"]})
        if not pdf:
            raise HTTPException(status_code=404, detail="PDF not found")
        
        # Get the GitHub URL from the database
        github_url = pdf.get("file_path")
        if not github_url:
            raise HTTPException(status_code=404, detail="PDF file path not found")
        
        # Download the PDF file from GitHub
        try:
            response = requests.get(github_url, timeout=30)
            response.raise_for_status()
            pdf_content = response.content
        except requests.RequestException as e:
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to download PDF from storage: {str(e)}"
            )
        
        # Compare answers and calculate score
        quiz_questions = quiz["questions"]
        correct_count = 0
        user_results = []
        
        for user_answer in request.user_answers:
            question_num = user_answer["question_number"]
            user_ans = user_answer["user_answer"]
            
            # Find the corresponding question
            question = next((q for q in quiz_questions if q.get("id") == question_num), None)
            if question:
                correct_ans = question.get("correct_answer")
                is_correct = user_ans == correct_ans and user_ans != "skipped"
                
                if is_correct:
                    correct_count += 1
                
                user_results.append({
                    "question_number": question_num,
                    "question": question.get("question"),
                    "correct_answer": correct_ans,
                    "user_answer": user_ans,
                    "status": "correct" if is_correct else ("skipped" if user_ans == "skipped" else "incorrect")
                })
        
        # Calculate score
        total_questions = len(quiz_questions)
        score_percentage = round((correct_count / total_questions) * 100) if total_questions > 0 else 0
        
        # Generate AI analysis (using existing generator)
        generator = PDFQuizGenerator()
        
        # Create quiz_results format for the AI analysis
        ai_quiz_results = {
            "topic": pdf["original_name"],
            "total_questions": total_questions,
            "user_answers": user_results,
            "score": f"{score_percentage}%"
        }
        
        analysis = generator.analyze_quiz_performance(
            quiz_results=ai_quiz_results,
            pdf_content=pdf_content
        )
        
        # Save analysis to database
        analysis_data = {
            "user_id": ObjectId(request.user_id),
            "quiz_id": ObjectId(request.quiz_id),
            "pdf_id": quiz["pdf_id"],
            "user_answers": user_results,
            "score": f"{score_percentage}%",
            "analysis": analysis,
            "completed_at": datetime.now()
        }
        
        analysis_result = db.analysis.insert_one(analysis_data)
        
        return {
            "success": True,
            "analysis": analysis,
            "score": f"{score_percentage}%",
            "analysis_id": str(analysis_result.inserted_id)
        }
        
    except HTTPException:
        raise
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

@app.post("/login")
def login_user(user: UserLogin):
    try:
        db = get_database()
        users_collection = db.users
        
        # Find user by email
        db_user = users_collection.find_one({"email": user.email})
        if not db_user:
            raise HTTPException(
                status_code=401, 
                detail="Invalid email or password"
            )
        
        # Verify password
        if not verify_password(user.password, db_user["password"]):
            raise HTTPException(
                status_code=401, 
                detail="Invalid email or password"
            )
        
        # Create access token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": str(db_user["_id"]), "email": db_user["email"]},
            expires_delta=access_token_expires
        )
        
        return {
            "success": True,
            "message": "Login successful",
            "access_token": access_token,
            "token_type": "bearer",
            "user": {
                "user_id": str(db_user["_id"]),
                "name": db_user["name"],
                "email": db_user["email"]
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Login failed: {str(e)}"
        )

@app.get("/user-pdfs/{user_id}")
def get_user_pdfs(user_id: str):
    """
    Get all PDFs for a specific user
    """
    try:
        db = get_database()
        
        # Validate user_id format
        try:
            user_object_id = ObjectId(user_id)
        except:
            raise HTTPException(
                status_code=400, 
                detail="Invalid user ID format"
            )
        
        # Check if user exists
        user = db.users.find_one({"_id": user_object_id})
        if not user:
            raise HTTPException(
                status_code=404, 
                detail="User not found"
            )
        
        # Get all PDFs for the user
        pdfs = list(db.pdfs.find(
            {"user_id": user_object_id},
            {"file_path": 0}  # Exclude file_path for security
        ).sort("uploaded_at", -1))  # Most recent first
        
        # Format the response
        pdf_list = []
        for pdf in pdfs:
            pdf_list.append({
                "pdf_id": str(pdf["_id"]),
                "original_name": pdf.get("original_name", "Unknown"),
                "file_size": pdf.get("file_size", 0),
                "uploaded_at": pdf.get("uploaded_at").isoformat() if pdf.get("uploaded_at") else None,
                "status": pdf.get("status", "unknown")
            })
        
        return {
            "success": True,
            "pdfs": pdf_list,
            "total": len(pdf_list)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to retrieve PDFs: {str(e)}"
        )

@app.get("/pdf-quiz-history/{pdf_id}")
def get_pdf_quiz_history(pdf_id: str):
    """
    Get quiz questions, user answers, and analysis for a specific PDF
    """
    try:
        db = get_database()
        
        # Validate pdf_id format
        try:
            pdf_object_id = ObjectId(pdf_id)
        except:
            raise HTTPException(
                status_code=400, 
                detail="Invalid PDF ID format"
            )
        
        # Get PDF info
        pdf = db.pdfs.find_one({"_id": pdf_object_id})
        if not pdf:
            raise HTTPException(
                status_code=404, 
                detail="PDF not found"
            )
        
        # Get quiz for this PDF
        quiz = db.quizzes.find_one({"pdf_id": pdf_object_id})
        if not quiz:
            raise HTTPException(
                status_code=404, 
                detail="No quiz found for this PDF"
            )
        
        # Get analysis/user answers for this PDF
        analysis = db.analysis.find_one(
            {"pdf_id": pdf_object_id},
            sort=[("completed_at", -1)]  # Get most recent attempt
        )
        
        if not analysis:
            # Quiz exists but not taken yet
            return {
                "success": True,
                "pdf_info": {
                    "pdf_id": str(pdf["_id"]),
                    "original_name": pdf.get("original_name"),
                    "uploaded_at": pdf.get("uploaded_at").isoformat() if pdf.get("uploaded_at") else None
                },
                "quiz_taken": False,
                "questions": [
                    {
                        "id": q.get("id"),
                        "question": q.get("question"),
                        "options": q.get("options"),
                        "correct_answer": q.get("correct_answer")
                    }
                    for q in quiz.get("questions", [])
                ],
                "message": "Quiz generated but not taken yet"
            }
        
        # Format the complete quiz history
        quiz_history = []
        quiz_questions = quiz.get("questions", [])
        user_answers_dict = {ans["question_number"]: ans for ans in analysis.get("user_answers", [])}
        
        for question in quiz_questions:
            q_id = question.get("id")
            user_answer_data = user_answers_dict.get(q_id, {})
            
            quiz_history.append({
                "question_number": q_id,
                "question": question.get("question"),
                "options": question.get("options"),
                "correct_answer": question.get("correct_answer"),
                "user_answer": user_answer_data.get("user_answer"),
                "status": user_answer_data.get("status"),
                "is_correct": user_answer_data.get("status") == "correct"
            })
        
        return {
            "success": True,
            "pdf_info": {
                "pdf_id": str(pdf["_id"]),
                "original_name": pdf.get("original_name"),
                "uploaded_at": pdf.get("uploaded_at").isoformat() if pdf.get("uploaded_at") else None
            },
            "quiz_taken": True,
            "quiz_history": quiz_history,
            "analysis": {
                "score": analysis.get("score"),
                "overall_performance": analysis.get("analysis", {}).get("overall_performance"),
                "strong_areas": analysis.get("analysis", {}).get("strong_areas", []),
                "weak_areas": analysis.get("analysis", {}).get("weak_areas", []),
                "recommendations": analysis.get("analysis", {}).get("recommendations", []),
                "completed_at": analysis.get("completed_at").isoformat() if analysis.get("completed_at") else None
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to retrieve quiz history: {str(e)}"
        )

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
