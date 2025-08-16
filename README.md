# PDF Quiz Generator MVP

A web application that converts PDF documents into multiple-choice quizzes using AI, supporting both English and Japanese.

## Features

- 📄 **PDF Processing**: Extract text from uploaded PDF files
- 🧠 **AI-Powered**: Generate intelligent quiz questions using OpenAI
- 🌏 **Multi-language**: Support for English and Japanese quizzes
- 🔍 **Vector Search**: Use Pinecone for semantic content retrieval
- 🌐 **Web Interface**: FastAPI-based REST API with HTML interface
- ⚡ **Real-time**: Hot reload during development

## Tech Stack

- **Backend**: FastAPI, Python 3.11
- **AI/ML**: LangChain, OpenAI GPT-3.5-turbo, OpenAI Embeddings
- **Vector DB**: Pinecone
- **PDF Processing**: PyPDF via LangChain
- **Environment**: Pipenv

## Prerequisites

- Python 3.11+
- Pipenv installed (`pip install pipenv`)
- OpenAI API key
- Pinecone API key and index

## Environment Setup

### 1. Clone and Navigate

```bash
cd pdf_quiz_mvp
```

### 2. Environment Variables

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment
PINECONE_INDEX_NAME=pdf-quiz-index
```

### 3. Virtual Environment Setup

If you're in another virtual environment (like conda/pipenv from another project):

```bash
# Set this environment variable to let pipenv create its own environment
export PIPENV_IGNORE_VIRTUALENVS=1
```

### 4. Install Dependencies

```bash
pipenv install
```

## Running the Application

### Development Server (Recommended)

```bash
# Set environment variable (if needed)
export PIPENV_IGNORE_VIRTUALENVS=1

# Start the FastAPI development server with hot reload
pipenv run uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

### Access the Application

- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## API Usage

### Upload PDF and Generate Quiz

```bash
curl -X POST "http://localhost:8000/generate-quiz" \
  -F "file=@your_document.pdf" \
  -F "num_questions=3" \
  -F "languages=english" \
  -F "languages=japanese"
```

### Response Format

```json
{
  "success": true,
  "message": "Quiz generated successfully!",
  "quiz_text": {
    "quizzes": {
      "english": "Question 1: What is...\nA) Option 1\n...",
      "japanese": "問題１: これは何ですか...\nア) 選択肢１\n..."
    }
  },
  "metadata": {
    "source_filename": "document.pdf",
    "file_size_bytes": 12345,
    "num_questions_requested": 3,
    "languages_generated": ["english", "japanese"]
  }
}
```

## Project Structure

```
pdf_quiz_mvp/
├── main.py              # Core quiz generation logic
├── api.py               # FastAPI web application
├── templates/           # HTML templates
│   └── index.html
├── Pipfile              # Dependencies
├── Pipfile.lock         # Locked dependency versions
├── .env                 # Environment variables (create this)
└── README.md            # This file
```

## Troubleshooting

### Virtual Environment Issues

If you see "Pipenv found itself running within a virtual environment":

```bash
export PIPENV_IGNORE_VIRTUALENVS=1
pipenv install
```

### Import Errors

If you get OpenAI import errors:

```bash
pipenv --rm                    # Remove old environment
pipenv install                # Reinstall fresh
```

### PDF Processing Errors

- Ensure PDF files are not password-protected
- Check file size (max 10MB)
- Verify PDF contains extractable text (not just images)

## Development

### Testing the Core Components

```bash
# Test OpenAI import
pipenv run python -c "from openai import OpenAI; print('OpenAI OK')"

# Test all imports
pipenv run python -c "
from openai import OpenAI
from pinecone import Pinecone
from langchain_community.document_loaders import PyPDFLoader
print('All imports successful!')
"
```

### Code Style

- Follow small step development approach
- Test each change before proceeding
- Verify functionality at each step

## Configuration

### Supported Languages

- `english`: English quiz generation
- `japanese`: Japanese quiz generation (translates content and creates Japanese questions)

### Quiz Parameters

- **Questions**: 1-10 per request
- **File Size**: Max 10MB
- **File Types**: PDF only
- **Chunk Size**: 300 characters with 50 character overlap

## License

MIT License - see project documentation for details.

## Troubleshoot

Install package like

```
pipenv install "pymongo[srv]==3.11"
```

Sync pipfile and requirements file before deploy to render platform

```
pipenv requirements > requirements.txt
```
