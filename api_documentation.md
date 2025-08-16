# PDF Quiz Generator API Documentation

## Overview

A complete MVP system that allows users to upload PDF documents, generate AI-powered quizzes, and track performance analytics. Built with FastAPI, MongoDB, and OpenAI integration.

## Base URL

```
http://localhost:8000
```

## Database Schema

### Collections

#### 1. Users Collection

```json
{
  "_id": ObjectId,
  "name": "string",
  "email": "string (unique)",
  "password": "string (hashed)",
  "created_at": "date"
}
```

#### 2. PDFs Collection

```json
{
  "_id": ObjectId,
  "user_id": ObjectId,
  "filename": "string",
  "original_name": "string",
  "file_path": "string",
  "file_size": "number",
  "uploaded_at": "date",
  "status": "processing|ready|failed"
}
```

#### 3. Quizzes Collection

```json
{
  "_id": ObjectId,
  "pdf_id": ObjectId,
  "user_id": ObjectId,
  "questions": [
    {
      "id": "number",
      "question": "string",
      "options": {
        "A": "string",
        "B": "string",
        "C": "string",
        "D": "string"
      },
      "correct_answer": "string"
    }
  ],
  "quiz_settings": {
    "total_questions": "number",
    "language": "string",
    "generated_at": "date",
    "source_filename": "string"
  }
}
```

#### 4. Analysis Collection

```json
{
  "_id": ObjectId,
  "user_id": ObjectId,
  "quiz_id": ObjectId,
  "pdf_id": ObjectId,
  "user_answers": [
    {
      "question_number": "number",
      "question": "string",
      "correct_answer": "string",
      "user_answer": "string",
      "status": "correct|incorrect|skipped"
    }
  ],
  "score": "string (percentage)",
  "analysis": {
    "overall_performance": "string",
    "strong_areas": ["string"],
    "weak_areas": ["string"],
    "recommendations": ["string"]
  },
  "completed_at": "date"
}
```

## Authentication

### JWT Token Format

- **Algorithm**: HS256
- **Expiration**: 30 minutes
- **Payload**: `{"sub": "user_id", "email": "user_email", "exp": timestamp}`

### Using JWT Tokens

Include in request headers:

```
Authorization: Bearer <your_jwt_token>
```

## API Endpoints

### 1. User Registration

**Endpoint**: `POST /register`

**Request Body**:

```json
{
  "name": "Test User",
  "email": "test@example.com",
  "password": "password123"
}
```

**Response**:

```json
{
  "success": true,
  "message": "User registered successfully",
  "user_id": "68a062f7d657d68d23eeeec3"
}
```

**Errors**:

- `400`: Email already registered
- `500`: Registration failed

---

### 2. User Login

**Endpoint**: `POST /login`

**Request Body**:

```json
{
  "email": "test@example.com",
  "password": "password123"
}
```

**Response**:

```json
{
  "success": true,
  "message": "Login successful",
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "user": {
    "user_id": "68a062f7d657d68d23eeeec3",
    "name": "Test User",
    "email": "test@example.com"
  }
}
```

**Errors**:

- `401`: Invalid email or password
- `500`: Login failed

---

### 3. Generate Quiz from PDF

**Endpoint**: `POST /generate-quiz-json`

**Content-Type**: `multipart/form-data`

**Request Body**:

```
file: <PDF_FILE>
user_id: "68a062f7d657d68d23eeeec3"
num_questions: "3"
languages: "english"
```

**Response**:

```json
{
  "success": true,
  "questions": [
    {
      "id": 1,
      "question": "Which technology does Shams Nahid have expertise in?",
      "options": {
        "A": "Java",
        "B": "Python",
        "C": "Node.js",
        "D": "C#"
      },
      "correct_answer": "C"
    }
  ],
  "quiz_id": "68a06a27329c72c696edc2bc",
  "pdf_id": "68a06a1c329c72c696edc2bb",
  "metadata": {
    "source_filename": "document.pdf",
    "file_size_bytes": 141554,
    "num_questions_requested": 3,
    "content_type": "application/pdf"
  }
}
```

**Validation**:

- File size: Max 10MB
- File type: PDF only
- Questions: 1-10 range
- User ID: Valid MongoDB ObjectId

**Errors**:

- `400`: Invalid file type/size, invalid parameters
- `500`: Processing failed, AI service unavailable

---

### 4. Analyze Quiz Performance

**Endpoint**: `POST /analyze-quiz-performance`

**Request Body**:

```json
{
  "quiz_id": "68a06a27329c72c696edc2bc",
  "user_id": "68a062f7d657d68d23eeeec3",
  "user_answers": [
    { "question_number": 1, "user_answer": "C" },
    { "question_number": 2, "user_answer": "B" },
    { "question_number": 3, "user_answer": "skipped" }
  ]
}
```

**Response**:

```json
{
  "success": true,
  "analysis": {
    "overall_performance": "The student demonstrated a basic understanding...",
    "strong_areas": [
      "Technology expertise identification",
      "Framework knowledge"
    ],
    "weak_areas": [
      "Experience timeline",
      "International collaboration details"
    ],
    "recommendations": [
      "Review the specific years of experience mentioned",
      "Study the international collaboration projects in detail"
    ]
  },
  "score": "67%",
  "analysis_id": "68a06b12329c72c696edc2bd"
}
```

**Answer Format**:

- Valid answers: "A", "B", "C", "D", "skipped"
- Question numbers must match generated quiz

**Errors**:

- `404`: Quiz not found, PDF not found
- `500`: Analysis failed

---

### 5. Database Health Check

**Endpoint**: `GET /test-db`

**Response**:

```json
{
  "database_connected": true
}
```

## File Storage

### PDF Files

- **Location**: `uploads/{user_id}/{timestamp}_{filename}.pdf`
- **Example**: `uploads/68a062f7d657d68d23eeeec3/20250816_143022_document.pdf`

### Directory Structure

```
project/
├── uploads/
│   ├── {user_id_1}/
│   │   ├── 20250816_143022_document1.pdf
│   │   └── 20250816_150315_document2.pdf
│   └── {user_id_2}/
│       └── 20250816_144455_report.pdf
```

## Data Relationships

```
User (1) ──→ (N) PDF
PDF (1) ──→ (N) Quiz
Quiz (1) ──→ (N) Analysis
User (1) ──→ (N) Analysis
```

## Frontend Implementation Guide

### Authentication Flow

1. **Register/Login** → Get JWT token
2. **Store token** in localStorage/sessionStorage
3. **Include token** in all API requests
4. **Handle token expiry** → Redirect to login

### Quiz Generation Flow

1. **Upload PDF** with user_id
2. **Show loading** while processing
3. **Display questions** from response
4. **Store quiz_id** for analysis

### Quiz Taking Flow

1. **Present questions** from quiz data
2. **Collect user answers**
3. **Submit to analysis** endpoint
4. **Show results** and performance insights

### State Management Suggestions

```javascript
// User state
const userState = {
  isAuthenticated: boolean,
  token: string,
  user: { id, name, email }
};

// Quiz state
const quizState = {
  currentQuiz: { id, questions, metadata },
  userAnswers: [{ question_number, user_answer }],
  isSubmitting: boolean
};

// Analysis state
const analysisState = {
  results: { score, analysis, analysis_id },
  history: [previous_analyses]
};
```

### Error Handling

- **401 Unauthorized**: Redirect to login
- **400 Bad Request**: Show validation errors
- **500 Server Error**: Show retry option
- **Network errors**: Show offline message

### API Integration Examples

#### Axios Setup

```javascript
const api = axios.create({
  baseURL: "http://localhost:8000",
  headers: {
    Authorization: `Bearer ${localStorage.getItem("token")}`
  }
});
```

#### File Upload

```javascript
const formData = new FormData();
formData.append("file", pdfFile);
formData.append("user_id", userId);
formData.append("num_questions", "5");
formData.append("languages", "english");

const response = await api.post("/generate-quiz-json", formData);
```

#### Quiz Submission

```javascript
const analysisData = {
  quiz_id: currentQuizId,
  user_id: userId,
  user_answers: collectedAnswers
};

const response = await api.post("/analyze-quiz-performance", analysisData);
```

## Testing Data

### Sample User

```json
{
  "email": "test@example.com",
  "password": "password123",
  "name": "Test User"
}
```

### Sample Quiz Response

- Questions with A/B/C/D options
- Each question has correct_answer field
- Quiz linked to PDF and user

### Sample Analysis Request

- Use actual quiz_id from generation
- Match question numbers exactly
- Use valid answer options or "skipped"

## Environment Variables

```bash
MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/quiz_app
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=pc-...
PINECONE_ENVIRONMENT=us-east-1
PINECONE_INDEX_NAME=docs
```

## Development Notes

- **File uploads** require multipart/form-data
- **JSON endpoints** require application/json
- **MongoDB ObjectIds** are 24-character hex strings
- **JWT tokens** expire in 30 minutes
- **PDF files** are permanently stored, not deleted
- **Quiz questions** use "id" field (not "question_number")
- **Analysis comparison** maps "id" to "question_number"
