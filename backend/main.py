"""
FastAPI backend for AI Study Buddy application.
"""
import os

os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['USE_TF'] = '0'

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
import uuid
import shutil
import json
import time

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from utils.document_processor import DocumentProcessor, TextChunker
from utils.vector_store import (
    ChromaDBStore, 
    EmbeddingGenerator,
    RAGPipeline
)
from utils.content_generator import ContentGenerator
from evaluation.evaluator import RAGEvaluator

# Paths for course ingestion state
COURSE_INDEX_FILE = Path(settings.processed_dir) / "courses_index.json"

Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
Path(settings.processed_dir).mkdir(parents=True, exist_ok=True)
Path(settings.courses_dir).mkdir(parents=True, exist_ok=True)

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG for more verbose output
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Ensure logs go to console
    ]
)
# Add file logging
log_dir = Path(settings.processed_dir).parent / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
file_handler = logging.FileHandler(log_dir / "app.log")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(file_handler)
logger = logging.getLogger(__name__)

# Set log levels for different modules
logging.getLogger('utils.document_processor').setLevel(logging.DEBUG)
logging.getLogger('utils.vector_store').setLevel(logging.INFO)
logging.getLogger('utils.content_generator').setLevel(logging.INFO)
logging.getLogger('chromadb').setLevel(logging.WARNING)  # Reduce ChromaDB noise
logging.getLogger('openai').setLevel(logging.WARNING)  # Reduce OpenAI noise
logging.getLogger('httpx').setLevel(logging.WARNING)  # Reduce HTTP noise

# Initialize FastAPI app

logger.info("Starting AI Study Buddy Backend Server")


# Ensure downstream libraries (e.g., DeepEval) see the OpenAI key
if settings.openai_api_key:
    os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key)
else:
    logger.warning("OPENAI_API_KEY is not configured; LLM features will fail until it is set.")

logger.info(f"OpenAI Model: {settings.openai_model}")
logger.info(f"Temperature: {settings.temperature}")
logger.info(f"Chunk Size: {settings.chunk_size}")
logger.info(f"Chunk Overlap: {settings.chunk_overlap}")
logger.info(f"Top-K Retrieval: {settings.top_k_retrieval}")
logger.info(f"Upload Directory: {settings.upload_dir}")
logger.info(f"ChromaDB Path: {settings.chromadb_path}")
logger.info(f"Courses Directory: {settings.courses_dir}")


app = FastAPI(
    title="AI Study Buddy API",
    description="API for AI-powered study assistance for MISM students",
    version="1.0.0"
)
logger.info(" FastAPI app initialized")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger.info("CORS middleware configured")

# Initialize components
logger.info("Initializing document processor...")
document_processor = DocumentProcessor()
logger.info(" Document processor initialized")

logger.info("Initializing text chunker...")
text_chunker = TextChunker(
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap
)
logger.info(f"Text chunker initialized (size={settings.chunk_size}, overlap={settings.chunk_overlap})")

# Initialize vector store and RAG pipeline
logger.info("Initializing vector store and RAG pipeline...")
try:
    logger.info(f"Creating ChromaDB store at: {settings.chromadb_path}")
    vector_store = ChromaDBStore(
        persist_directory=settings.chromadb_path,
        collection_name="study_materials"
    )
    logger.info(" ChromaDB store created")
    
    # Initialize OpenAI embedding generator
    logger.info("Initializing OpenAI embedding generator...")
    embedding_generator = EmbeddingGenerator(
        api_key=settings.openai_api_key,
        model=settings.openai_embedding_model
    )
    logger.info(f" Using OpenAI embeddings: {settings.openai_embedding_model}")
    
    logger.info("Creating RAG pipeline...")
    rag_pipeline = RAGPipeline(vector_store, embedding_generator)
    logger.info(" RAG pipeline initialized successfully")
except Exception as e:
    logger.error(f"Error initializing RAG pipeline: {str(e)}")
    import traceback
    logger.error(traceback.format_exc())
    rag_pipeline = None

# Initialize content generator
logger.info("Initializing OpenAI content generator...")
try:
    content_generator = ContentGenerator(
        api_key=settings.openai_api_key,
        model=settings.openai_model,
        temperature=settings.temperature
    )
    logger.info(f" Using OpenAI content generator: {settings.openai_model}")
    
    logger.info(" Content generator initialized successfully")
except Exception as e:
    logger.error(f"Error initializing content generator: {str(e)}")
    import traceback
    logger.error(traceback.format_exc())
    content_generator = None

# Initialize evaluator (optional dependency)
evaluator = RAGEvaluator(use_ragas=True, use_deepeval=True, llm_model=settings.openai_model)
logger.info("All components initialized!")

# Course ingestion utilities
SUPPORTED_EXTENSIONS = {'.pdf', '.pptx', '.docx', '.txt'}

def _load_course_index() -> Dict[str, float]:
    if COURSE_INDEX_FILE.exists():
        try:
            return json.load(open(COURSE_INDEX_FILE, "r"))
        except Exception:
            logger.warning("Could not read course index file, starting fresh")
    return {}

def _save_course_index(index: Dict[str, float]) -> None:
    COURSE_INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(COURSE_INDEX_FILE, "w") as f:
        json.dump(index, f, indent=2)

def get_course_subjects() -> List[str]:
    base = Path(settings.courses_dir)
    if not base.exists():
        return []
    return sorted([item.name for item in base.iterdir() if item.is_dir()])

def get_course_files(subject: str) -> List[Dict]:
    """List files for a given subject with metadata."""
    subject_dir = Path(settings.courses_dir) / subject
    if not subject_dir.exists():
        return []
    files = []
    for file_path in subject_dir.iterdir():
        if file_path.suffix.lower() in SUPPORTED_EXTENSIONS and file_path.is_file():
            stat = file_path.stat()
            files.append({
                "file_name": file_path.name,
                "file_path": str(file_path.resolve()),
                "subject": subject,
                "size": stat.st_size,
                "modified_time": stat.st_mtime
            })
    return sorted(files, key=lambda f: f["file_name"])

def count_uploaded_documents() -> int:
    """Count uploaded documents in the upload directory."""
    upload_dir = Path(settings.upload_dir)
    if not upload_dir.exists():
        return 0
    return len([f for f in upload_dir.iterdir() if f.is_file()])

def count_course_documents() -> int:
    """Count course documents across subjects."""
    total = 0
    for subject in get_course_subjects():
        total += len(get_course_files(subject))
    return total

def get_document_counts() -> Dict[str, int]:
    """Return document counts by source and total."""
    uploads = count_uploaded_documents()
    courses = count_course_documents()
    return {
        "uploads": uploads,
        "courses": courses,
        "total": uploads + courses
    }

def build_filter_dict(subject: Optional[str], file_names: Optional[List[str]]) -> Optional[Dict]:
    """Build a Chroma filter dict from subject and file names."""
    clauses = []
    if subject:
        clauses.append({"subject": subject})
    if file_names:
        cleaned = [f for f in file_names if f]
        if cleaned:
            clauses.append({"file_name": {"$in": cleaned}})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}

def index_course_materials(force: bool = False) -> Dict:
    """Index course materials under courses/<subject>."""
    if not rag_pipeline:
        logger.warning("RAG pipeline not initialized, skipping course indexing")
        return {"indexed": 0, "skipped": 0, "deleted": 0, "subjects": []}
    
    index_state = _load_course_index()
    seen_paths = set()
    indexed = 0
    skipped = 0
    deleted = 0
    subjects = get_course_subjects()
    
    for subject in subjects:
        for file_meta in get_course_files(subject):
            file_path = Path(file_meta["file_path"])
            seen_paths.add(file_meta["file_path"])
            mtime = file_meta["modified_time"]
            
            if not force and index_state.get(file_meta["file_path"]) == mtime:
                skipped += 1
                continue
            
            try:
                logger.info(f"Indexing course file: {file_meta['file_name']} (subject={subject})")
                processed = document_processor.process_file(str(file_path))
                chunks = text_chunker.chunk_text(
                    processed["text"],
                    metadata={
                        **processed["metadata"],
                        "source": "course",
                        "subject": subject,
                        "file_name": file_path.name,
                        "file_path": str(file_path.resolve()),
                        "document_id": f"course::{subject}::{file_path.name}"
                    }
                )
                
                # Remove old entries before adding
                rag_pipeline.delete_by_metadata({"file_path": str(file_path.resolve())})
                rag_pipeline.add_documents(chunks)
                
                index_state[file_meta["file_path"]] = mtime
                indexed += 1
            except Exception as exc:
                logger.error(f"Failed to index {file_path}: {exc}")
    
    # Remove entries for deleted files
    for stored_path in list(index_state.keys()):
        if stored_path not in seen_paths:
            rag_pipeline.delete_by_metadata({"file_path": stored_path})
            index_state.pop(stored_path, None)
            deleted += 1
    
    _save_course_index(index_state)
    return {"indexed": indexed, "skipped": skipped, "deleted": deleted, "subjects": subjects}

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Log startup information."""
    
    logger.info("AI Study Buddy Backend Started Successfully!")
    logger.info("Backend is ready to accept requests!")
    # Index course materials at startup (non blocking on failure)
    try:
        result = index_course_materials(force=False)
        logger.info(f"Course indexing completed at startup: {result}")
    except Exception as exc:
        logger.error(f"Error during startup course indexing: {exc}")

@app.on_event("shutdown")
async def shutdown_event():
    """Log shutdown information."""
    
    logger.info("Shutting down AI Study Buddy Backend...")
    

# Pydantic models for API
class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    subject: Optional[str] = None
    file_names: Optional[List[str]] = None  # Only for course files
    document_id: Optional[str] = None       # Uploaded document filter

class SummaryRequest(BaseModel):
    topic: Optional[str] = None
    document_id: Optional[str] = None
    subject: Optional[str] = None
    file_names: Optional[List[str]] = None

class QuestionRequest(BaseModel):
    question_type: str  # mcq, true_false, fill_blank, short_answer, match_following
    num_questions: Optional[int] = 5
    document_id: Optional[str] = None
    context: Optional[str] = None
    subject: Optional[str] = None
    file_names: Optional[List[str]] = None

class NotesRequest(BaseModel):
    subject: Optional[str] = None
    file_names: Optional[List[str]] = None
    document_id: Optional[str] = None
    topic: Optional[str] = None

class EvaluateRequest(BaseModel):
    query: str
    answer: str
    ground_truth: Optional[str] = None
    contexts: Optional[List[str]] = None  # list of context strings (preferred)
    context: Optional[str] = None  # single concatenated context string (fallback)
    subject: Optional[str] = None
    file_names: Optional[List[str]] = None
    document_id: Optional[str] = None

class FeedbackRequest(BaseModel):
    session_id: str
    question_id: str
    rating: int  # 1-5
    comment: Optional[str] = None

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Welcome to AI Study Buddy API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    logger.debug("Health check requested")
    health_status = {
        "status": "healthy",
        "rag_pipeline": "initialized" if rag_pipeline else "not initialized",
        "content_generator": "initialized" if content_generator else "not initialized",
        "llm_provider": "openai",
        "evaluator": "initialized"
    }
    logger.debug(f"Health status: {health_status}")
    return health_status

@app.get("/courses/subjects")
async def list_subjects():
    """List available course subjects."""
    subjects = get_course_subjects()
    return {"status": "success", "subjects": subjects}

@app.get("/courses/{subject}/files")
async def list_subject_files(subject: str):
    """List files for a specific subject."""
    files = get_course_files(subject)
    if not files:
        return {"status": "success", "files": [], "subject": subject}
    return {"status": "success", "files": files, "subject": subject}

@app.post("/courses/index")
async def trigger_course_indexing(force: bool = False):
    """Trigger indexing of course materials."""
    result = index_course_materials(force=force)
    return {"status": "success", "result": result}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process a document.
    Args:
        file: Uploaded file (PDF, PPTX, DOCX, TXT)
        
    Returns:
        Processing results and document ID
    """
    
    logger.info(f"Document upload request received")
    logger.info(f"Filename: {file.filename}")
    logger.info(f"Content Type: {file.content_type}")
    
    if not rag_pipeline:
        logger.error("RAG pipeline not initialized")
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    # Validate file type
    file_ext = Path(file.filename).suffix.lower()
    logger.info(f"File extension: {file_ext}")
    if file_ext not in document_processor.SUPPORTED_FORMATS:
        logger.error(f"Unsupported file format: {file_ext}")
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Supported formats: {document_processor.SUPPORTED_FORMATS}"
        )
    logger.info(" File type validated")
    
    # Generate unique document ID
    doc_id = str(uuid.uuid4())
    logger.info(f"Generated document ID: {doc_id}")
    
    # Create upload directory
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Upload directory ready: {upload_dir}")
    
    # Save uploaded file
    file_path = upload_dir / f"{doc_id}_{file.filename}"
    
    try:
        logger.info("Saving uploaded file...")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f" File saved: {file_path}")
        # Check file size limit (10MB max to prevent crashes)
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        logger.info(f"File size: {file_size_mb:.2f} MB")
        
        if file_size_mb > 10:
            logger.warning(f"File exceeds 10MB limit ({file_size_mb:.2f} MB)")
            file_path.unlink()  # Delete file
            raise HTTPException(
                status_code=400,
                detail=f"File too large ({file_size_mb:.1f} MB). Maximum file size is 10 MB. Please split your document into smaller files."
            )
        # Process document
        
        logger.info("STEP 1/4: Extracting text from document...")
        
        processed_doc = document_processor.process_file(str(file_path))
        text_length = len(processed_doc['text'])
        
        logger.info(f"STEP 1 COMPLETE: {text_length} characters extracted")
        
        # Warn if document is very large
        if text_length > 100000:
            logger.warning(f"Large document detected ({text_length} chars). Processing may take 1-2 minutes...")
        # Chunk text
        logger.info("STEP 2/4: Chunking text into smaller segments...")
        chunks = text_chunker.chunk_text(
            processed_doc['text'],
            metadata={
                'document_id': doc_id,
                'file_name': file.filename,
                'file_type': file_ext,
                **processed_doc['metadata']
            }
        )
        
        logger.info(f" STEP 2 COMPLETE: Created {len(chunks)} chunks from document")
        
        # Warn if too many chunks
        if len(chunks) > 200:
            logger.warning(f"Large number of chunks ({len(chunks)}). Embedding generation may take several minutes...")
        
        # Add to RAG pipeline
        estimated_time_min = len(chunks) * 0.5
        estimated_time_max = len(chunks) * 2
        
        logger.info(f"STEP 3/4: Generating embeddings for {len(chunks)} chunks...")
        logger.info(f"  Estimated time: {estimated_time_min:.0f}-{estimated_time_max:.0f} seconds")
        logger.info("  Please wait, this may take a while for large documents...")
        
        chunk_ids = rag_pipeline.add_documents(chunks)
        
        logger.info(f" STEP 3 COMPLETE: Generated {len(chunk_ids)} embeddings and stored in vector database")
        
        logger.info(" Document upload completed successfully!")
        
        return {
            "status": "success",
            "document_id": doc_id,
            "file_name": file.filename,
            "file_type": file_ext,
            "chunks_created": len(chunks),
            "metadata": processed_doc['metadata'],
            "message": "Document processed and added to knowledge base successfully"
        }
    
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        # Clean up file if processing failed
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@app.post("/query")
async def query_documents(request: QueryRequest):
    """
    Query the knowledge base using RAG.
    Args:
        request: Query request with query text and optional top_k    
    Returns:
        Retrieved context and generated answer
    """
    if not rag_pipeline or not content_generator:
        raise HTTPException(status_code=503, detail="Services not initialized")
    try:
        # Retrieve relevant context
        filter_dict = build_filter_dict(request.subject, request.file_names)
        if request.document_id:
            filter_dict = filter_dict or {}
            filter_dict["document_id"] = request.document_id
        context = rag_pipeline.retrieve_context(
            query=request.query,
            top_k=request.top_k,
            filter_dict=filter_dict
        )
        # Generate answer
        answer = content_generator.generate_answer_with_context(
            query=request.query,
            context=context
        )
        return {
            "status": "success",
            "query": request.query,
            "answer": answer,
            "context": context
        }
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/summary")
async def generate_summary(request: SummaryRequest, query: Optional[str] = None):
    """
    Generate a summary of course materials.
    Args:
        request: Summary request with optional topic and document_id
        query: Optional specific content to summarize    
    Returns:
        Generated summary
    """
    if not rag_pipeline or not content_generator:
        raise HTTPException(status_code=503, detail="Services not initialized")
    try:
        # If no specific query provided, use topic or generic query
        if not query:
            if request.topic:
                query = f"Explain the topic: {request.topic}"
            else:
                query = "Provide an overview of the course materials"
        
        # Build filter for document-specific retrieval or course selections
        filter_dict = build_filter_dict(request.subject, request.file_names)
        if request.document_id:
            filter_dict = filter_dict or {}
            filter_dict["document_id"] = request.document_id
            logger.info(f"Filtering summary by document_id: {request.document_id}")
        if filter_dict:
            logger.info(f"Using filter for summary: {filter_dict}")
        
        # Retrieve context (filtered by document if specified)
        context = rag_pipeline.retrieve_context(
            query=query, 
            top_k=settings.top_k_retrieval,
            filter_dict=filter_dict
        )
        
        logger.info(f"Retrieved context length: {len(context)} characters")
        
        # Check if we got any context
        if context == "No relevant context found." or not context or len(context.strip()) < 50:
            logger.warning(f"No context found for document_id: {request.document_id}")
            raise HTTPException(
                status_code=404,
                detail="No content found for this document. The document may not have been properly processed. Please try re-uploading the document."
            )
        
        # Generate summary
        summary = content_generator.generate_summary(context, topic=request.topic)
        
        return {
            "status": "success",
            "topic": request.topic,
            "document_id": request.document_id,
            "summary": summary,
            "context": context  # Include context so users can verify source
        }
    
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating summary: {str(e)}")


@app.post("/notes")
async def generate_notes(request: NotesRequest):
    """
    Generate detailed study notes for selected course materials.
    """
    if not rag_pipeline or not content_generator:
        raise HTTPException(status_code=503, detail="Services not initialized")
    
    try:
        filter_dict = build_filter_dict(request.subject, request.file_names)
        if request.document_id:
            filter_dict = filter_dict or {}
            filter_dict["document_id"] = request.document_id
        
        query = request.topic or "Generate detailed study notes covering all main ideas"
        context = rag_pipeline.retrieve_context(
            query=query,
            top_k=settings.top_k_retrieval,
            filter_dict=filter_dict
        )
        if not context or context == "No relevant context found.":
            raise HTTPException(
                status_code=404,
                detail="No content found for the selected files/subject. Please check your selection."
            )
        
        notes = content_generator.generate_notes(context, topic=request.topic)
        return {
            "status": "success",
            "notes": notes,
            "subject": request.subject,
            "file_names": request.file_names
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating notes: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating notes: {str(e)}")

@app.post("/questions")
async def generate_questions(request: QuestionRequest):
    """
    Generate practice questions from course materials.
    
    Args:
        request: Question generation request
        
    Returns:
        Generated questions
    """
    if not rag_pipeline or not content_generator:
        raise HTTPException(status_code=503, detail="Services not initialized")
    
    try:
        # Get context if not provided
        if not request.context:
            # Use a generic query to get relevant context
            filter_dict = build_filter_dict(request.subject, request.file_names)
            if request.document_id:
                filter_dict = filter_dict or {}
                filter_dict["document_id"] = request.document_id
            context = rag_pipeline.retrieve_context(
                query="Generate practice questions covering main concepts",
                top_k=settings.top_k_retrieval,
                filter_dict=filter_dict
            )
        else:
            context = request.context
        
        # Generate questions based on type
        if request.question_type == "mcq":
            questions = content_generator.generate_mcq_questions(context, request.num_questions)
        elif request.question_type == "true_false":
            questions = content_generator.generate_true_false_questions(context, request.num_questions)
        elif request.question_type == "fill_blank":
            questions = content_generator.generate_fill_blank_questions(context, request.num_questions)
        elif request.question_type == "short_answer":
            questions = content_generator.generate_short_answer_questions(context, request.num_questions)
        elif request.question_type == "match_following":
            questions = content_generator.generate_match_following_questions(context, request.num_questions)
        else:
            raise HTTPException(status_code=400, detail=f"Invalid question type: {request.question_type}")
        
        return {
            "status": "success",
            "question_type": request.question_type,
            "questions": questions
        }
    
    except Exception as e:
        logger.error(f"Error generating questions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating questions: {str(e)}")


def _normalize_contexts(contexts: Optional[List[str]], context_str: Optional[str]) -> List[str]:
    """
    Normalize contexts input to a list of strings.
    """
    if contexts and isinstance(contexts, list):
        return [c for c in contexts if c]
    if context_str:
        parts = []
        marker = "[Context"
        if marker in context_str:
            for block in context_str.split(marker):
                block = block.strip()
                if not block:
                    continue
                block = block.split("]", 1)[-1].strip() if "]" in block else block
                if block:
                    parts.append(block)
        else:
            parts = [context_str]
        return parts
    return []


@app.post("/evaluate")
async def evaluate_rag(request: EvaluateRequest):
    """
    Evaluate a RAG response using available metrics (RAGAS/DeepEval).
    If contexts are not provided, retrieves them using the subject/file filters or document_id.
    """
    if not evaluator:
        raise HTTPException(status_code=503, detail="Evaluator not initialized")
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    try:
        contexts_list = _normalize_contexts(request.contexts, request.context)
        
        # Retrieve context if not provided
        if not contexts_list:
            filter_dict = build_filter_dict(request.subject, request.file_names)
            if request.document_id:
                filter_dict = filter_dict or {}
                filter_dict["document_id"] = request.document_id
            retrieved = rag_pipeline.retrieve(
                query=request.query,
                top_k=settings.top_k_retrieval,
                filter_dict=filter_dict
            )
            contexts_list = [item["text"] for item in retrieved]
        
        if not contexts_list:
            raise HTTPException(status_code=404, detail="No context available to evaluate.")
        
        eval_result = evaluator.evaluate_rag_response(
            query=request.query,
            answer=request.answer,
            contexts=contexts_list,
            ground_truth=request.ground_truth
        )
        
        return {"status": "success", "evaluation": eval_result}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during evaluation: {str(e)}")

@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """
    Submit user feedback for evaluation.
    
    Args:
        request: Feedback request
        
    Returns:
        Confirmation
    """
    logger.info(f"Feedback received: {request.dict()}")
    
    return {
        "status": "success",
        "message": "Feedback received successfully"
    }

@app.get("/stats")
async def get_stats():
    """
    Get statistics about the knowledge base.
    
    Returns:
        Statistics
    """
    if not vector_store:
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    
    try:
        doc_count = vector_store.get_collection_count()
        collections = vector_store.list_collections()
        doc_counts = get_document_counts()
        
        return {
            "status": "success",
            "total_documents": doc_count,
            "collections": collections,
            "document_counts": doc_counts
        }
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")

@app.get("/documents")
async def list_documents():
    """
    List all uploaded documents.
    
    Returns:
        List of uploaded documents with metadata
    """
    try:
        upload_dir = Path(settings.upload_dir)
        
        if not upload_dir.exists():
            return {
                "status": "success",
                "documents": []
            }
        
        documents = []
        for file_path in upload_dir.glob("*"):
            if file_path.is_file():
                # Parsing filename to extract doc_id and original name
                filename = file_path.name
                parts = filename.split("_", 1)
                
                if len(parts) == 2:
                    doc_id = parts[0]
                    original_name = parts[1]
                else:
                    doc_id = "unknown"
                    original_name = filename
                
                # Get file stats
                stat = file_path.stat()
                
                documents.append({
                    "document_id": doc_id,
                    "file_name": original_name,
                    "file_path": str(file_path),
                    "file_size": stat.st_size,
                    "upload_time": stat.st_mtime,
                    "file_type": file_path.suffix.lower()
                })
        
        # Sort by upload time (newest first)
        documents.sort(key=lambda x: x["upload_time"], reverse=True)
        
        return {
            "status": "success",
            "documents": documents,
            "count": len(documents)
        }
    
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")


@app.post("/extract-topics")
async def extract_topics(document_id: str = Form(...)):
    """
    Extract topics from a specific document using AI.
    
    Args:
        document_id: The document ID to extract topics from
        
    Returns:
        List of extracted topics
    """
    if not content_generator:
        raise HTTPException(status_code=503, detail="Content generator not initialized")
    
    try:
        # Find the document file
        upload_dir = Path(settings.upload_dir)
        doc_file = None
        
        for file_path in upload_dir.glob(f"{document_id}_*"):
            doc_file = file_path
            break
        
        if not doc_file or not doc_file.exists():
            raise HTTPException(status_code=404, detail="Document not found")
        
        logger.info(f"Extracting topics from document: {doc_file.name}")
        
        # Process document to get text
        processed_doc = document_processor.process_file(str(doc_file))
        text = processed_doc['text']
        
        # Limit text length for topic extraction (first 5000 characters)
        text_sample = text[:5000] if len(text) > 5000 else text
        
        # Use AI to extract topics
        prompt = f"""Analyze the following document excerpt and extract 5-10 main topics or themes.
        
Document excerpt:
{text_sample}

Please provide a JSON response with a list of topics. Each topic should be:
- Concise (2-6 words)
- Representative of key concepts in the document
- Suitable as a summary topic

Format:
{{
    "topics": [
        "Topic 1",
        "Topic 2",
        ...
    ]
}}
"""
        
        # Generate topics using the content generator
        if hasattr(content_generator, 'client'):
            # OpenAI
            response = content_generator.client.chat.completions.create(
                model=content_generator.model,
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing documents and extracting key topics. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            result_text = response.choices[0].message.content
        else:
            # Hugging Face
            result_text = content_generator._call_model(
                "You are an expert at analyzing documents and extracting key topics. Always respond with valid JSON.",
                prompt
            )
        
        # Parse the JSON response
        try:
            result = json.loads(result_text)
            topics = result.get("topics", [])
        except json.JSONDecodeError:
            # Fallback: extract topics manually from text
            logger.warning("Failed to parse AI response, using fallback topic extraction")
            topics = [
                "Overview and Introduction",
                "Key Concepts",
                "Main Arguments",
                "Methodology",
                "Conclusions"
            ]
        
        logger.info(f"Extracted {len(topics)} topics from document")
        
        return {
            "status": "success",
            "document_id": document_id,
            "topics": topics,
            "document_name": doc_file.name.split("_", 1)[1] if "_" in doc_file.name else doc_file.name
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error extracting topics: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error extracting topics: {str(e)}")


@app.delete("/delete-all-data")
async def delete_all_data():
    """
    Delete all data from the system including:
    - All uploaded documents (raw files)
    - All processed documents
    - ChromaDB vector store
    """
    try:
        
        logger.info("Starting complete data deletion...")
        
        
        deleted_items = {
            "chromadb": 0,
            "raw_files": 0,
            "processed_files": 0
        }
        
        # Delete ChromaDB data
        chromadb_path = Path(settings.chromadb_path)
        if chromadb_path.exists():
            logger.info(f"Deleting ChromaDB data from: {chromadb_path}")
            for item in chromadb_path.iterdir():
                if item.is_file():
                    item.unlink()
                    deleted_items["chromadb"] += 1
                    logger.debug(f"   Deleted: {item.name}")
                elif item.is_dir():
                    shutil.rmtree(item)
                    deleted_items["chromadb"] += 1
                    logger.debug(f"   Deleted directory: {item.name}")
            logger.info(f" Deleted {deleted_items['chromadb']} ChromaDB items")
        
        # Delete raw uploaded files
        raw_path = Path(settings.upload_dir)
        if raw_path.exists():
            logger.info(f"Deleting raw files from: {raw_path}")
            for item in raw_path.iterdir():
                if item.is_file():
                    item.unlink()
                    deleted_items["raw_files"] += 1
                    logger.debug(f"   Deleted: {item.name}")
            logger.info(f" Deleted {deleted_items['raw_files']} raw files")
        
        # Delete processed files
        processed_path = Path(settings.processed_dir)
        if processed_path.exists():
            logger.info(f"Deleting processed files from: {processed_path}")
            for item in processed_path.iterdir():
                if item.is_file():
                    item.unlink()
                    deleted_items["processed_files"] += 1
                    logger.debug(f"   Deleted: {item.name}")
            logger.info(f" Deleted {deleted_items['processed_files']} processed files")
        
        total_deleted = sum(deleted_items.values())
        
        
        logger.info(f" Data deletion completed successfully!")
        logger.info(f"  - ChromaDB items: {deleted_items['chromadb']}")
        logger.info(f"  - Raw files: {deleted_items['raw_files']}")
        logger.info(f"  - Processed files: {deleted_items['processed_files']}")
        logger.info(f"  - Total items deleted: {total_deleted}")
        
        
        return {
            "status": "success",
            "message": "All data deleted successfully",
            "deleted_items": deleted_items,
            "total_deleted": total_deleted
        }
    
    except Exception as e:
        logger.error(f"Error deleting data: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error deleting data: {str(e)}")


if __name__ == "__main__":
    # Create necessary directories
    os.makedirs(settings.upload_dir, exist_ok=True)
    os.makedirs(settings.processed_dir, exist_ok=True)
    os.makedirs(settings.chromadb_path, exist_ok=True)
    
    # Run the application
    uvicorn.run(
        "main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=True
    )
