import os
import streamlit as st
from preprocessor import FilePreprocessor
from rag_pipeline import RAGPipeline

# Initialize session state
if 'rag_pipeline' not in st.session_state:
    st.session_state.rag_pipeline = None
if 'document_processed' not in st.session_state:
    st.session_state.document_processed = False

# Set page config
st.set_page_config(
    page_title="RAG Pipeline",
    layout="wide",
    page_icon="📄"
)

# Dark theme CSS
st.markdown("""
    <style>
        .stApp {
            background-color: #2e2e2e;
            color: white;
        }
        .stTextInput>div>div>input {
            color: white !important;
            background-color: #4a4a4a;
        }
        .css-1d391kg, .st-bb, .st-at, .st-ae, .st-af, .st-ag, .st-ah, .st-ai, .st-aj, .st-ak {
            color: white !important;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize components
preprocessor = FilePreprocessor()
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Main UI
st.title("📄 Document Question Answering with RAG")

# File upload in sidebar
with st.sidebar:
    st.header("📤 Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a file (PDF, DOCX, TXT, CSV, etc.)",
        type=["pdf", "docx", "txt", "csv", "db", "sqlite", "jpg", "jpeg", "png"],
        key="file_uploader"
    )
    
    if uploaded_file is not None:
        file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        try:
            with st.spinner("Processing document..."):
                documents = preprocessor.process_file(file_path)
                st.session_state.rag_pipeline = RAGPipeline()
                st.session_state.rag_pipeline.initialize_from_documents(documents)
                st.session_state.document_processed = True
                st.success("✅ Document processed successfully!")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.session_state.document_processed = False

# Question answering section
st.header("❓ Ask a Question")
question = st.text_input(
    "Enter your question about the document:",
    key="question_input"
)

if st.button("Get Answer", type="primary"):
    if not question:
        st.warning("⚠️ Please enter a question")
    elif not st.session_state.document_processed:
        st.warning("⚠️ Please upload and process a document first")
    else:
        with st.spinner("🔍 Searching for answer..."):
            try:
                answer = st.session_state.rag_pipeline.query(question)
                st.subheader("📝 Answer:")
                st.markdown(f"<div style='background-color: #4a4a4a; padding: 15px; border-radius: 5px;'>{answer}</div>", 
                            unsafe_allow_html=True)
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# Status indicator
st.sidebar.markdown("---")
if st.session_state.document_processed:
    st.sidebar.success("✅ Document ready for queries")
else:
    st.sidebar.warning("⚠️ No document loaded yet")