"""
Streamlit frontend for AI Study Buddy application.
"""
import streamlit as st
import requests
import json
from typing import Optional
from pathlib import Path

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Page configuration
st.set_page_config(
    page_title="AI Study Buddy",
    page_icon="✎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #e8edf2;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .question-box {
        background-color: #1b1f2a;
        color: #e8edf2;
        border: 1px solid #2c3444;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .success-message {
        color: #28a745;
        font-weight: bold;
    }
    .error-message {
        color: #dc3545;
        font-weight: bold;
    }
    /* Darken text areas/inputs to avoid white boxes on dark background */
    .stTextArea textarea, .stTextInput input, .stNumberInput input {
        background-color: #1b1f2a !important;
        color: #e8edf2 !important;
        border: 1px solid #2c3444 !important;
    }
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: #1b1f2a;
        color: #e8edf2;
    }
</style>
""", unsafe_allow_html=True)


def check_api_health():
    """Check if the API is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def upload_document(file):
    """Upload a document to the API with timeout."""
    try:
        files = {"file": (file.name, file, file.type)}
        # Set a longer timeout for large files (5 minutes)
        response = requests.post(f"{API_BASE_URL}/upload", files=files, timeout=300)
        return response.json()
    except requests.exceptions.Timeout:
        return {"error": "Upload timed out. The document might be too large or the server is busy. Please try again."}
    except requests.exceptions.ConnectionError:
        return {"error": "Connection error. Please ensure the backend server is running."}
    except Exception as e:
        return {"error": str(e)}


def query_documents(query: str, top_k: int = 5, subject: Optional[str] = None, file_names: Optional[list] = None, document_id: Optional[str] = None):
    """Query the knowledge base."""
    try:
        payload = {"query": query, "top_k": top_k, "subject": subject, "file_names": file_names, "document_id": document_id}
        response = requests.post(f"{API_BASE_URL}/query", json=payload)
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def generate_summary(topic: Optional[str] = None, query: Optional[str] = None, document_id: Optional[str] = None, subject: Optional[str] = None, file_names: Optional[list] = None):
    """Generate a summary, optionally filtered by document_id."""
    try:
        payload = {"topic": topic, "document_id": document_id, "subject": subject, "file_names": file_names}
        params = {"query": query} if query else {}
        response = requests.post(f"{API_BASE_URL}/summary", json=payload, params=params)
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def generate_questions(question_type: str, num_questions: int = 5, context: Optional[str] = None, subject: Optional[str] = None, file_names: Optional[list] = None, document_id: Optional[str] = None):
    """Generate practice questions."""
    try:
        payload = {
            "question_type": question_type,
            "num_questions": num_questions,
            "context": context,
            "subject": subject,
            "file_names": file_names,
            "document_id": document_id
        }
        response = requests.post(f"{API_BASE_URL}/questions", json=payload)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def evaluate_rag(query: str, answer: str, ground_truth: Optional[str] = None, subject: Optional[str] = None, file_names: Optional[list] = None, document_id: Optional[str] = None):
    """Call evaluation endpoint."""
    try:
        payload = {
            "query": query,
            "answer": answer,
            "ground_truth": ground_truth,
            "subject": subject,
            "file_names": file_names,
            "document_id": document_id
        }
        response = requests.post(f"{API_BASE_URL}/evaluate", json=payload)
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def get_stats():
    """Get knowledge base statistics."""
    try:
        response = requests.get(f"{API_BASE_URL}/stats")
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def list_documents():
    """Get list of uploaded documents."""
    try:
        response = requests.get(f"{API_BASE_URL}/documents")
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def extract_topics(document_id: str):
    """Extract topics from a specific document."""
    try:
        payload = {"document_id": document_id}
        response = requests.post(f"{API_BASE_URL}/extract-topics", data=payload)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def generate_notes(subject: Optional[str] = None, file_names: Optional[list] = None, document_id: Optional[str] = None, topic: Optional[str] = None):
    """Generate detailed notes for selected materials."""
    try:
        payload = {
            "subject": subject,
            "file_names": file_names,
            "document_id": document_id,
            "topic": topic
        }
        response = requests.post(f"{API_BASE_URL}/notes", json=payload)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def get_subjects():
    """List available subjects."""
    try:
        response = requests.get(f"{API_BASE_URL}/courses/subjects")
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def get_subject_files(subject: str):
    """List files for a subject."""
    try:
        response = requests.get(f"{API_BASE_URL}/courses/{subject}/files")
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def reindex_courses(force: bool = False):
    """Trigger course indexing."""
    try:
        response = requests.post(f"{API_BASE_URL}/courses/index", params={"force": force})
        return response.json()
    except Exception as e:
        return {"error": str(e)}

@st.cache_data(show_spinner=False)
def cached_subjects():
    """Cached subjects list."""
    return get_subjects()

@st.cache_data(show_spinner=False)
def cached_subject_files(subject: str):
    """Cached files for a subject."""
    return get_subject_files(subject)

def render_subject_file_selector(key_prefix: str):
    """Render subject + file selectors and return selections."""
    subjects_response = cached_subjects()
    subjects = subjects_response.get("subjects", []) if isinstance(subjects_response, dict) else []
    subject = st.selectbox(
        "Select a subject (optional)",
        options=["-- None --"] + subjects,
        index=0,
        key=f"{key_prefix}_subject"
    )
    
    selected_subject = subject if subject != "-- None --" else None
    selected_files = []
    available_files = []
    
    if selected_subject:
        files_response = cached_subject_files(selected_subject)
        available_files = files_response.get("files", []) if isinstance(files_response, dict) else []
        file_names = [f["file_name"] for f in available_files]
        
        col_a, col_b = st.columns([1, 2])
        with col_a:
            select_all = st.checkbox("Select all files", key=f"{key_prefix}_select_all")
        with col_b:
            selected_files = st.multiselect(
                "Choose files within the subject",
                options=file_names,
                default=file_names if select_all else [],
                key=f"{key_prefix}_files"
            )
            if select_all:
                selected_files = file_names
    
    return selected_subject, selected_files, available_files

@st.cache_data(show_spinner=False, ttl=60)
def cached_uploaded_documents():
    """Cached list of uploaded documents."""
    docs_result = list_documents()
    if "error" in docs_result:
        return []
    return docs_result.get("documents", [])


def delete_all_data():
    """Delete all data from the system."""
    try:
        response = requests.delete(f"{API_BASE_URL}/delete-all-data")
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def show_home_page():
    """Home page with overview and disclaimer."""
    
    st.markdown('<h2 style="color:#e8edf2;">Your AI-Powered Learning Companion at CMU</h2>', unsafe_allow_html=True)
    st.markdown("""
<div style="
    background: #f6f8fb;
    color: #1c2833;
    padding: 16px 18px;
    border-radius: 10px;
    border: 1px solid #d0d7e2;
    font-size: 16px;
    line-height: 1.5;
">
<strong style="color:#0d47a1;">AI Study Buddy</strong> is your on-demand assistant for course materials ask questions, get summaries, craft practice questions, and generate detailed notes. Use preloaded course folders or your own uploads, then pick one source at a time on each page to stay focused.
</div>
""", unsafe_allow_html=True)
    
    st.markdown("### What you can do")
    st.markdown("""
- Upload PDFs/PPTX/DOCX/TXT and query them with AI-grounded answers
- Summarize course materials or generate detailed study notes
- Create practice questions (MCQ, True/False, Fill-in-the-blank, Short Answer, Match)
- Work with preloaded course folders or your own uploads
    """)
    
    st.markdown("### How to use")
    st.markdown("""
1) **Use course folders** (already preloaded): pick a subject in the data source switcher on each page  
2) **Or upload** your own documents (Upload tab)  
3) **Choose a single data source** on each page (Uploaded document **or** Course subject files)  
4) Run summaries, notes, questions, or chat from the respective tabs  
5) If you just uploaded, use the **Refresh uploaded documents** button on Query/Practice to see it
    """)
    
    st.markdown("### Disclaimer")
    st.info("AI-generated content may be imperfect. Always cross-check with your source materials.")


def main():
    """Main application."""
    
    # Header
    st.markdown('<h1 class="main-header">✎ AI Study Buddy for MISM Students</h1>', unsafe_allow_html=True)
    
    
    # Check API status
    if not check_api_health():
        st.error(" Backend API is not running. Please start the backend server first.")
        st.code("cd backend && python main.py", language="bash")
        return
    
    # Sidebar
    with st.sidebar:
        logo_path = Path(__file__).parent / "assets" / "cmu_logo.jpg"
        if logo_path.exists():
            st.image(str(logo_path), width=160)
        else:
            st.markdown(
                "Add your CMU logo at `frontend/assets/cmu_logo.jpg` to show it here.",
                help="Place a PNG logo in the assets folder so the sidebar can load it locally."
            )
        st.markdown("## Navigation")

        nav_items = [
            ("home", "⌂ Home"),
            ("upload", "⤴ Upload Documents"),
            ("ask", "✎ Ask Questions"),
            ("summary", "⎘ Generate Summary"),
            ("practice", "❔ Practice Questions"),
            ("notes", "✍︎ Notes"),
            ("stats", "▦ Statistics"),
        ]
        nav_labels = {key: label for key, label in nav_items}
        page = st.radio(
            "Select a feature:",
            [key for key, _ in nav_items],
            format_func=lambda key: nav_labels.get(key, key),
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.info("AI Study Buddy helps MISM students learn through interactive summaries and practice questions powered by RAG and LLMs.")
        
        # Show stats in sidebar
        stats = get_stats()
        if "error" not in stats:
            doc_counts = stats.get("document_counts", {})
            st.metric("Documents", doc_counts.get("total", stats.get("total_documents", 0)))
    
    # Main content area
    if page == "home":
        show_home_page()
    elif page == "upload":
        show_upload_page()
    elif page == "ask":
        show_query_page()
    elif page == "summary":
        show_summary_page()
    elif page == "practice":
        show_practice_page()
    elif page == "notes":
        show_notes_page()
    elif page == "stats":
        show_stats_page()


def show_upload_page():
    """Document upload page."""
    st.markdown('<div class="section-header">⤴ Upload Course Materials</div>', unsafe_allow_html=True)
    st.markdown("Upload your PDFs, slides, or notes to build your personalized knowledge base.")
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "pptx", "docx", "txt"],
        help="Supported formats: PDF, PPTX, DOCX, TXT"
    )
    
    if uploaded_file:
        file_size_mb = uploaded_file.size / (1024 * 1024)
        st.info(f"File selected: **{uploaded_file.name}** ({file_size_mb:.2f} MB)")
        
        # Warn about large files
        if file_size_mb > 0.5:
            st.warning(f" Large file detected ({file_size_mb:.1f} MB). Processing may take 2-5 minutes. Please be patient!")
        
        # Show estimated processing time based on file size
        if file_size_mb < 0.1:
            estimated_time = "5-15 seconds"
        elif file_size_mb < 0.5:
            estimated_time = "30-60 seconds"
        elif file_size_mb < 2:
            estimated_time = "2-5 minutes"
        else:
            estimated_time = "5-10 minutes"
        
        st.caption(f"Estimated processing time: {estimated_time}")
        
        # Warn if file is too large
        if file_size_mb > 10:
            st.error("File is too large! Maximum file size is 10 MB.")
            st.info("Please split your document into smaller files or reduce the file size.")
            return
        
        if st.button("▶ Process Document", type="primary"):
            # Create a progress bar and status text
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("⇧ Uploading document to server...")
            progress_bar.progress(20)
            
            status_text.text("🗈 Extracting text from document...")
            progress_bar.progress(40)
            
            status_text.text("✂ Chunking text for processing...")
            progress_bar.progress(60)
            
            status_text.text("⌁ Generating embeddings (this may take a moment)...")
            progress_bar.progress(80)
            
            status_text.text("⌂ Storing in knowledge base...")
            
            result = upload_document(uploaded_file)
            
            progress_bar.progress(100)
            
            if "error" in result:
                status_text.empty()
                progress_bar.empty()
                st.error(f"Error: {result['error']}")
                st.info(" **Troubleshooting Tips:**")
                st.markdown("""
                - Make sure the backend server is running (`cd backend && python main.py`)
                - Check the backend terminal for detailed logs
                - For large PDFs (50+ pages), the process may take 30-60 seconds
                - If using OpenAI, verify your API key is set correctly
                """)
            else:
                status_text.empty()
                progress_bar.empty()
                st.success("✅ Document processed successfully!")
                st.balloons()
                
                # Show details in columns
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Document ID", result.get("document_id", "N/A")[:8] + "...")
                with col2:
                    st.metric("Chunks Created", result.get("chunks_created", 0))
                with col3:
                    st.metric("File Type", result.get("file_type", "N/A"))
                
                # Show metadata
                with st.expander("▦ View Document Metadata"):
                    st.json(result.get("metadata", {}))


def show_query_page():
    """Question answering page."""
    st.markdown('<div class="section-header">✎ Ask Questions</div>', unsafe_allow_html=True)
    st.markdown("Ask questions about your course materials and get AI-powered answers with context.")
    
    # Choose data source: uploads OR course files
    source_choice = st.radio(
        "Choose data source",
        ["Uploaded document", "Course subject files"],
        index=0,
        key="query_source_choice"
    )
    
    selected_doc_id = None
    subject = None
    selected_files = None
    
    if source_choice == "Uploaded document":
        refresh_uploads = st.button("↻ Refresh uploaded documents", key="query_refresh_uploads")
        if refresh_uploads:
            cached_uploaded_documents.clear()
            st.rerun()
        uploaded_docs = cached_uploaded_documents()
        if not uploaded_docs:
            st.info("No uploaded documents yet. Upload a document first.")
        doc_options = {"-- Select an uploaded document --": None}
        for doc in uploaded_docs:
            doc_options[f"{doc.get('file_name')}"] = doc.get("document_id")
        selected_doc_label = st.selectbox(
            "Restrict to an uploaded document",
            options=list(doc_options.keys()),
            key="query_doc_filter"
        )
        selected_doc_id = doc_options[selected_doc_label]
    else:
        subject, selected_files, _ = render_subject_file_selector("query")
    
    # Initialize session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Query input
    query = st.text_input(
        "Your Question:",
        placeholder="e.g., What is machine learning?",
        key="query_input"
    )
    
    col1, col2 = st.columns([3, 1])
    with col1:
        search_button = st.button("🔍 Search", type="primary")
    with col2:
        top_k = st.number_input("Top Results", min_value=1, max_value=10, value=5)
    
    if search_button and query:
        with st.spinner("Searching knowledge base..."):
            result = query_documents(
                query,
                top_k,
                subject=subject,
                file_names=selected_files or None,
                document_id=selected_doc_id
            )
            
            if "error" in result:
                st.error(f"Error: {result['error']}")
            else:
                # Add to chat history
                st.session_state.chat_history.append({
                    "query": query,
                    "answer": result.get("answer", "No answer generated")
                })
                
                # Display answer
                st.markdown("### Answer")
                st.markdown(result.get("answer", "No answer generated"))
                
                # Show context
                with st.expander("▦ View Retrieved Context"):
                    st.text_area(
                        "Context used to generate the answer:",
                        result.get("context", "No context available"),
                        height=300
                    )
                
                # Feedback
                st.markdown("---")
                st.markdown("### ▦ Was this answer helpful?")
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.button("⭐")
                with col2:
                    st.button("⭐⭐")
                with col3:
                    st.button("⭐⭐⭐")
                with col4:
                    st.button("⭐⭐⭐⭐")
                with col5:
                    st.button("⭐⭐⭐⭐⭐")
    
    # Show chat history
    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("### Recent Questions")
        for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):
            with st.expander(f"Q: {chat['query'][:50]}..."):
                st.markdown(f"**Question:** {chat['query']}")
                st.markdown(f"**Answer:** {chat['answer']}")


def show_summary_page():
    """Summary page - Generate a comprehensive summary of the entire document."""
    st.markdown('<div class="section-header">⎘ Generate Document Summary</div>', unsafe_allow_html=True)
    st.markdown("Use course folders or uploaded documents to create summaries.")
    
    source_choice = st.radio(
        "Choose data source",
        ["Uploaded document", "Course subject files"],
        index=0,
        key="summary_source_choice"
    )
    topic = st.text_input("Topic focus (optional)", key="summary_topic_input")
    
    selected_doc_id = None
    subject = None
    files_for_request = None
    
    if source_choice == "Uploaded document":
        refresh_uploads = st.button("↻ Refresh uploaded documents", key="summary_refresh_uploads")
        if refresh_uploads:
            cached_uploaded_documents.clear()
            st.rerun()
        documents = cached_uploaded_documents()
        if not documents:
            st.info("No uploaded documents yet. Go to the **Upload Documents** page to get started!")
            return
        st.markdown("### Select an Uploaded Document")
        doc_options = {"-- Select an uploaded document --": None}
        for doc in documents:
            doc_options[f"{doc.get('file_name')}"] = doc.get("document_id")
        selected_label = st.selectbox(
            "Choose a document to summarize:",
            options=list(doc_options.keys()),
            key="summary_doc_selector"
        )
        selected_doc_id = doc_options[selected_label]
        if not selected_doc_id:
            return
        selected_doc = next(doc for doc in documents if doc['document_id'] == selected_doc_id)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("File Type", selected_doc['file_type'].upper())
        with col2:
            st.metric("File Size", f"{selected_doc['file_size'] / 1024:.1f} KB")
        with col3:
            upload_time = datetime.fromtimestamp(selected_doc['upload_time'])
            st.metric("Uploaded", upload_time.strftime("%b %d, %Y"))
        
        if 'current_doc_id' not in st.session_state or st.session_state.current_doc_id != selected_doc_id:
            st.session_state.current_doc_id = selected_doc_id
            st.session_state.document_summary = None
            st.session_state.current_doc_label = selected_doc['file_name']
        
        if st.button("⎘ Generate Summary", type="primary"):
            with st.spinner(f"Summarizing {selected_doc['file_name']}..."):
                query = "Provide a comprehensive summary of the entire document covering all main topics, key points, and important information"
                result = generate_summary(
                    topic=topic or "Complete Document Overview",
                    query=query,
                    document_id=selected_doc_id
                )
                if "error" in result:
                    st.error(f"Error: {result['error']}")
                    st.warning(" This document may not have been properly processed.")
                    st.info("Try re-uploading this document in the 'Upload Documents' page.")
                    return
                st.session_state.document_summary = {
                    "summary": result.get("summary", "No summary available"),
                    "context": result.get("context", "")
                }
                st.session_state.current_doc_label = selected_doc['file_name']
                st.success("Summary generated successfully!")
    else:
        subject, selected_files, available_files = render_subject_file_selector("summary")
        files_for_request = selected_files or ([f["file_name"] for f in available_files] if subject else None)
        if subject:
            st.markdown(f"**Subject selected:** {subject}")
            if available_files and not files_for_request:
                st.warning("Select at least one file or use 'Select all files' to continue.")
                return
            if st.button("⎘ Generate Summary from Course Files", type="primary"):
                with st.spinner("Generating summary from selected course files..."):
                    query = "Provide a comprehensive summary of the selected materials."
                    result = generate_summary(
                        topic=topic or "Course Materials Summary",
                        query=query,
                        subject=subject,
                        file_names=files_for_request
                    )
                    if "error" in result:
                        st.error(f"Error: {result['error']}")
                        return
                    st.session_state.document_summary = {
                        "summary": result.get("summary", "No summary available"),
                        "context": result.get("context", "")
                    }
                    selection_label = f"{subject} ({', '.join(files_for_request) if files_for_request else 'all files'})"
                    st.session_state.current_doc_label = selection_label
                    st.success("Summary generated successfully!")
    
    if st.session_state.get("document_summary"):
        st.markdown("---")
        st.markdown("### Document Summary")
        st.markdown(f"**Source:** {st.session_state.get('current_doc_label', 'Selection')}")
        
        st.markdown('<div class="question-box">', unsafe_allow_html=True)
        st.markdown(st.session_state.document_summary['summary'])
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            st.download_button(
                "↓ Download Summary",
                st.session_state.document_summary['summary'],
                file_name="summary.txt",
                mime="text/plain",
                use_container_width=True
            )
        with col2:
            show_source = st.button("View Source Content", use_container_width=True)
        
        if show_source:
            st.session_state.show_source_toggle = not st.session_state.get("show_source_toggle", False)
        
        if st.session_state.get("show_source_toggle", False):
            st.markdown("---")
            st.caption("Source content used to generate this summary:")
            st.text_area(
                "Source Content",
                st.session_state.document_summary.get('context', 'No source available'),
                height=300,
                disabled=True,
                label_visibility="collapsed"
            )


def show_practice_page():
    """Practice questions page."""
    st.markdown('<div class="section-header">❔ Practice Questions</div>', unsafe_allow_html=True)
    st.markdown("Generate practice questions to test your understanding.")
    
    source_choice = st.radio(
        "Choose data source",
        ["Uploaded document", "Course subject files"],
        index=0,
        key="practice_source_choice"
    )
    
    selected_doc_id = None
    subject = None
    selected_files = None
    
    if source_choice == "Uploaded document":
        refresh_uploads = st.button("↻ Refresh uploaded documents", key="practice_refresh_uploads")
        if refresh_uploads:
            cached_uploaded_documents.clear()
            st.rerun()
        uploaded_docs = cached_uploaded_documents()
        if not uploaded_docs:
            st.info("No uploaded documents yet. Upload a document first.")
        doc_options = {"-- Select an uploaded document --": None}
        for doc in uploaded_docs:
            doc_options[f"{doc.get('file_name')}"] = doc.get("document_id")
        selected_doc_label = st.selectbox(
            "Restrict to an uploaded document",
            options=list(doc_options.keys()),
            key="practice_doc_filter"
        )
        selected_doc_id = doc_options[selected_doc_label]
    else:
        subject, selected_files, _ = render_subject_file_selector("practice")
    
    # Question type selection
    col1, col2 = st.columns([2, 1])
    with col1:
        question_type = st.selectbox(
            "Question Type:",
            ["mcq", "true_false", "fill_blank", "short_answer", "match_following"],
            format_func=lambda x: {
                "mcq": "Multiple Choice Questions (MCQ)",
                "true_false": "True/False",
                "fill_blank": "Fill in the Blanks",
                "short_answer": "Short Answer",
                "match_following": "Match the Following"
            }[x]
        )
    with col2:
        num_questions = st.number_input(
            "Number of Questions:",
            min_value=1,
            max_value=10,
            value=5
        )
    
    if st.button("▶ Generate Questions", type="primary"):
        with st.spinner("Generating questions..."):
            result = generate_questions(
                question_type,
                num_questions,
                subject=subject,
                file_names=selected_files or None,
                document_id=selected_doc_id
            )
            
            if "error" in result:
                st.error(f"Error: {result['error']}")
            else:
                questions = result.get("questions", [])
                
                if not questions:
                    st.warning("No questions generated. Try uploading more documents first.")
                else:
                    st.success(f"Generated {len(questions)} questions!")
                    
                    # Display questions based on type
                    for i, q in enumerate(questions, 1):
                        with st.container():
                            st.markdown(f'<div class="question-box">', unsafe_allow_html=True)
                            
                            if question_type == "mcq":
                                st.markdown(f"**Question {i}:** {q.get('question', '')}")
                                for option, text in q.get('options', {}).items():
                                    st.markdown(f"- **{option}:** {text}")
                                with st.expander("Show Answer"):
                                    st.success(f"Correct Answer: **{q.get('correct_answer', '')}**")
                                    st.info(q.get('explanation', ''))
                            
                            elif question_type == "true_false":
                                st.markdown(f"**Statement {i}:** {q.get('statement', '')}")
                                with st.expander("Show Answer"):
                                    st.success(f"Answer: **{'True' if q.get('answer') else 'False'}**")
                                    st.info(q.get('explanation', ''))
                            
                            elif question_type == "fill_blank":
                                st.markdown(f"**Question {i}:** {q.get('question', '')}")
                                with st.expander("Show Answer"):
                                    st.success(f"Answer: **{q.get('answer', '')}**")
                                    if q.get('hint'):
                                        st.info(f"Hint: {q.get('hint')}")
                            
                            elif question_type == "short_answer":
                                st.markdown(f"**Question {i}:** {q.get('question', '')}")
                                with st.expander("Show Sample Answer"):
                                    st.success(f"Sample Answer: {q.get('sample_answer', '')}")
                                    st.info("Key Points: " + ", ".join(q.get('key_points', [])))
                            
                            elif question_type == "match_following":
                                st.markdown(f"**{q.get('instruction', 'Match the Following')}**")
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown("**Column A:**")
                                    for item in q.get('column_a', []):
                                        st.markdown(f"- {item.get('id')}: {item.get('text')}")
                                with col2:
                                    st.markdown("**Column B:**")
                                    for item in q.get('column_b', []):
                                        st.markdown(f"- {item.get('id')}: {item.get('text')}")
                                with st.expander("Show Answer"):
                                    st.success("Correct Matches:")
                                    for a, b in q.get('correct_matches', {}).items():
                                        st.markdown(f"- {a} → {b}")
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                            st.markdown("")


def show_notes_page():
    """Detailed study notes page."""
    st.markdown('<div class="section-header">✍︎ Notes</div>', unsafe_allow_html=True)
    st.markdown("Generate detailed, section-by-section study notes.")
    
    source_choice = st.radio(
        "Choose data source",
        ["Uploaded document", "Course subject files"],
        index=0,
        key="notes_source_choice"
    )
    topic = st.text_input("Topic focus (optional)", key="notes_topic_input")
    
    notes_result = None
    
    if source_choice == "Uploaded document":
        with st.spinner("Loading your uploaded documents..."):
            docs_result = list_documents()
        if "error" in docs_result:
            st.error(f"❌ Error loading documents: {docs_result['error']}")
            st.info("Upload documents first or select a subject.")
            return
        documents = docs_result.get("documents", [])
        if not documents:
            st.info("No uploaded documents yet.")
            return
        doc_options = {doc['file_name']: doc['document_id'] for doc in documents}
        chosen_name = st.selectbox("Choose an uploaded document for notes", options=list(doc_options.keys()))
        doc_id = doc_options.get(chosen_name)
        if st.button("✍︎ Generate Notes", type="primary"):
            with st.spinner("Creating detailed notes..."):
                result = generate_notes(document_id=doc_id, topic=topic)
                if "error" in result:
                    st.error(f"❌ Error: {result['error']}")
                    return
                notes_result = result
                st.session_state.notes_content = result.get("notes", "")
                st.session_state.notes_source = chosen_name
    else:
        subject, selected_files, available_files = render_subject_file_selector("notes")
        files_for_request = selected_files or ([f["file_name"] for f in available_files] if subject else None)
        if subject:
            if available_files and not files_for_request:
                st.warning("Select at least one file or use 'Select all files' to continue.")
                return
            if st.button("✍︎ Generate Notes from Course Files", type="primary"):
                with st.spinner("Creating detailed notes from course materials..."):
                    result = generate_notes(
                        subject=subject,
                        file_names=files_for_request,
                        topic=topic
                    )
                    if "error" in result:
                        st.error(f"❌ Error: {result['error']}")
                        return
                    notes_result = result
                    st.session_state.notes_content = result.get("notes", "")
                    st.session_state.notes_source = f"{subject} ({', '.join(files_for_request) if files_for_request else 'all files'})"
    
    notes_to_show = st.session_state.get("notes_content")
    if notes_to_show:
        st.markdown("---")
        st.markdown(f"### ✍︎ Notes for {st.session_state.get('notes_source', '')}")
        st.markdown('<div class="question-box">', unsafe_allow_html=True)
        st.markdown(notes_to_show)
        st.markdown('</div>', unsafe_allow_html=True)
        st.download_button(
            "↓ Download Notes",
            notes_to_show,
            file_name="notes.txt",
            mime="text/plain",
            use_container_width=True
        )


def show_stats_page():
    """Statistics page."""
    st.markdown('<div class="section-header">▦ Knowledge Base Statistics</div>', unsafe_allow_html=True)
    
    stats = get_stats()
    
    if "error" in stats:
        st.error(f"❌ Error: {stats['error']}")
    else:
        doc_counts = stats.get("document_counts", {})
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Document files",
                doc_counts.get("total", 0),
                help="Total number of course + uploaded files"
            )
        with col2:
            st.metric(
                "Uploads",
                doc_counts.get("uploads", 0),
                help="User-uploaded files"
            )
        with col3:
            st.metric(
                "Course files",
                doc_counts.get("courses", 0),
                help="Preloaded course files"
            )
        with col4:
            st.metric(
                "Vector chunks",
                stats.get("total_documents", 0),
                help="Number of chunks stored in the vector database"
            )
        
        st.markdown("### Collections")
        if stats.get("collections"):
            for collection in stats.get("collections", []):
                st.markdown(f"- {collection}")
        else:
            st.info("No collections found. Upload documents to create a collection.")
        
        st.markdown("### 🧪 Evaluate a response")
        eval_query = st.text_area("Query", key="eval_query", height=80)
        eval_answer = st.text_area("Answer", key="eval_answer", height=80)
        eval_ground = st.text_area("Ground truth (optional)", key="eval_ground", height=80)
        
        source_choice = st.radio(
            "Choose data source for context retrieval (optional)",
            ["Uploaded document", "Course subject files", "None (use provided contexts)"],
            index=2,
            key="eval_source_choice"
        )
        
        eval_subject = None
        eval_files = None
        eval_doc_id = None
        
        if source_choice == "Uploaded document":
            refresh_uploads = st.button("↻ Refresh uploaded documents", key="eval_refresh_uploads")
            if refresh_uploads:
                cached_uploaded_documents.clear()
                st.rerun()
            uploaded_docs = cached_uploaded_documents()
            doc_options = {"-- Select an uploaded document --": None}
            for doc in uploaded_docs:
                doc_options[f"{doc.get('file_name')}"] = doc.get("document_id")
            selected_label = st.selectbox(
                "Restrict to an uploaded document",
                options=list(doc_options.keys()),
                key="eval_doc_selector"
            )
            eval_doc_id = doc_options[selected_label]
        elif source_choice == "Course subject files":
            eval_subject, eval_files, _ = render_subject_file_selector("eval")
        
        if st.button("Run evaluation"):
            if not eval_query or not eval_answer:
                st.error("Please provide both query and answer.")
            else:
                with st.spinner("Running evaluation..."):
                    eval_result = evaluate_rag(
                        query=eval_query,
                        answer=eval_answer,
                        ground_truth=eval_ground or None,
                        subject=eval_subject,
                        file_names=eval_files or None,
                        document_id=eval_doc_id
                    )
                if "error" in eval_result:
                    st.error(f"❌ Error: {eval_result['error']}")
                else:
                    data = eval_result.get("evaluation", {})
                    st.success("✅ Evaluation complete")
                    st.json(data)


if __name__ == "__main__":
    main()
