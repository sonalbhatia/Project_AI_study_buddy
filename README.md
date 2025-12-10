# AI-Powered Study Buddy for MISM Students

An intelligent study assistant that helps MISM students at CMU learn from their course materials through AI-powered summaries, practice questions, and interactive Q&A using Retrieval-Augmented Generation (RAG).

## Features

### Core Functionality
- **Document Processing**: Upload and process PDFs, PowerPoint slides, Word documents, and text files
- **RAG-Powered Q&A**: Ask questions and get accurate answers grounded in your course materials
- **Smart Summaries**: Generate concise summaries of topics or entire documents
- **Practice Questions**: Auto-generate diverse question types:
  - Multiple Choice Questions (MCQs)
  - True/False
  - Fill-in-the-Blanks
  - Short Answer
  - Match-the-Following
  - Long Answer/Essay Questions
- **Evaluation Framework**: Built-in RAGAS and DeepEval metrics for quality assessment
- **User Feedback**: Collect and analyze user feedback for continuous improvement

## Architecture

```
AI-study-buddy/
├── backend/              # FastAPI backend server
│   ├── main.py          # API endpoints
│   ├── config.py        # Configuration management
│   └── requirements.txt
├── frontend/            # Streamlit UI
│   ├── app.py          # Main application
│   └── requirements.txt
├── utils/              # Core utilities
│   ├── document_processor.py  # Document parsing and chunking
│   ├── vector_store.py        # Vector database and RAG pipeline
│   └── content_generator.py  # LLM-powered content generation
├── evaluation/         # Evaluation framework
│   └── evaluator.py   # RAGAS and DeepEval integration
├── data/              # Data storage
│   ├── raw/          # Uploaded documents
│   ├── processed/    # Processed documents
│   └── chromadb/     # Vector database
├── models/           # Model configurations
└── tests/           # Test suite for debugging
```

## Quick Start

### Prerequisites
- Python 3.9 or higher
- **OpenAI API key** (required)
- Git

### Setup

1. **Get your OpenAI API key**: Visit [OpenAI API Keys](https://platform.openai.com/api-keys)

### Installation

1. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

2. **Install backend dependencies**
```bash
cd backend
pip install -r requirements.txt
```

3. **Install frontend dependencies**
```bash
cd ../frontend
pip install -r requirements.txt
```

### Running the Application

1. **Start the backend server** (Terminal 1)
```bash
cd backend
python main.py
```
The API will be available at `http://localhost:8000`

2. **Start the frontend** (Terminal 2)
```bash
cd frontend
streamlit run app.py
```
The UI will open in your browser at `http://localhost:8501`

## Usage Guide

### 1. Upload Documents
- Navigate to the "Upload Documents" page
- Select a PDF, PPTX, DOCX, or TXT file
- Click "Process Document"
- The system will extract text, create chunks, and generate embeddings

### 2. Ask Questions
- Go to the "Ask Questions" page
- Type your question about the course materials
- Get AI-generated answers with supporting context
- Rate the answer to help improve the system

### 3. Generate Summaries
- Visit the "Generate Summary" page
- Choose topic-based or custom query summary
- Receive a concise summary of the content
- Download the summary for later reference

### 4. Practice Questions
- Select "Practice Questions"
- Choose question type (MCQ, True/False, etc.)
- Specify number of questions
- Review questions and answers for self-assessment

## Configuration

Edit `.env` file to customize:

```bash
# OpenAI Settings
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4-turbo-preview
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# RAG Settings
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_RETRIEVAL=5
TEMPERATURE=0.4

# Vector Database
VECTOR_DB_TYPE=chromadb
CHROMADB_PATH=./data/chromadb
```

## RAG Pipeline

The system implements a sophisticated RAG pipeline:

1. **Document Processing**
   - Extract text from various file formats
   - Split into manageable chunks with overlap
   - Preserve metadata and context

2. **Embedding Generation**
   - Generate vector embeddings using OpenAI
   - Store in ChromaDB for efficient retrieval
   - Support batch processing for large documents

3. **Retrieval**
   - Convert queries to embeddings
   - Perform similarity search
   - Retrieve top k most relevant chunks

4. **Generation**
   - Provide context to LLM
   - Generate accurate, grounded responses
   - Include source references

## Evaluation

The system includes comprehensive evaluation:

### RAGAS Metrics
- **Faithfulness**: How factually accurate are the answers?
- **Answer Relevancy**: How relevant is the answer to the query?
- **Context Precision**: How precise is the retrieved context?
- **Context Recall**: How complete is the retrieved context?

### DeepEval Metrics
- **Answer Relevancy**: Semantic relevance of responses
- **Faithfulness**: Consistency with source material
- **Coherence**: Logical flow and clarity

### User Feedback
- Star ratings (1-5)
- Qualitative comments
- Usage analytics

## Security & Privacy

- Documents are stored locally
- No data sharing with third parties
- API keys stored securely in environment variables
- User sessions isolated
- Uploaded files can be deleted anytime

## API Documentation

Once the backend is running, visit:
- API Docs: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Key Endpoints

- **POST /upload** - Upload and process a document
- **POST /query** - Ask questions about materials
- **POST /summary** - Generate summaries
- **POST /questions** - Generate practice questions
- **GET /stats** - Get knowledge base statistics

## Tips for Best Results

1. **Upload Quality Materials**: Clear, well-formatted documents work best
2. **Specific Questions**: More specific queries yield better answers
3. **Chunk Size**: Adjust based on your document structure
4. **Regular Updates**: Keep adding new materials for better coverage
5. **Provide Feedback**: Help improve the system through ratings

---

Built with ❤️ for MISM students at Carnegie Mellon University
