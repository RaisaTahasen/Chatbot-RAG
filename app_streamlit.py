import os
import streamlit as st
from preprocessor import FilePreprocessor
from rag_pipeline import RAGPipeline
from PIL import Image
import io

# Initialize session state
if 'rag_pipeline' not in st.session_state:
    st.session_state.rag_pipeline = None
if 'document_processed' not in st.session_state:
    st.session_state.document_processed = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'show_history' not in st.session_state:
    st.session_state.show_history = False
if 'image_question' not in st.session_state:
    st.session_state.image_question = None

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
        .chat-message {
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
        }
        .user-message {
            background-color: #4a4a4a;
            margin-left: 20%;
            margin-right: 5%;
        }
        .bot-message {
            background-color: #3a3a3a;
            margin-left: 5%;
            margin-right: 20%;
        }
        .history-container {
            max-height: 500px;
            overflow-y: auto;
            padding: 10px;
            border-radius: 5px;
            background-color: #3a3a3a;
            margin-bottom: 20px;
        }       
    </style>
""", unsafe_allow_html=True)

preprocessor = FilePreprocessor()
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Main UI
col1, col2 = st.columns([1.2, 0.2])
with col1:
    st.title("🧰 Welcome to Chatbot API with RAG")
with col2:
    if st.button("Session History", key="toggle_history"):
        st.session_state.show_history = not st.session_state.show_history

# Show chat history if toggled
if st.session_state.show_history:
    st.header("Chat History")
    if not st.session_state.chat_history:
        st.markdown("<div style='text-align: center; color: #888;'>No chat history yet</div>", unsafe_allow_html=True)
    else:
        with st.container():
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    st.markdown(
                        f"<div class='chat-message user-message'>👤 <strong>You:</strong> {message['content']}</div>", 
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"<div class='chat-message bot-message'>🤖 <strong>Bot:</strong> {message['content']}</div>", 
                        unsafe_allow_html=True
                    )
# File upload in sidebar

with st.sidebar:
    st.header("📤 Upload Document or paste an url")

    input_type = st.radio("Select input type:", ("File Upload", "URL"))
    if input_type == "File Upload":
        uploaded_file = st.file_uploader(
            "Choose a file (PDF, DOCX, TXT, CSV, etc.)",
            type=["pdf", "docx", "txt", "csv", "sqlite.db", "jpg", "jpeg", "png"],
            key="file_uploader"
        )
    
        if uploaded_file is not None:
            file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            try:
                with st.spinner("Processing document..."):
                    #documents = preprocessor.process_file(file_path)
                    new_documents = preprocessor.process_file(file_path)

                    if st.session_state.rag_pipeline is None:
                        st.session_state.rag_pipeline = RAGPipeline()
                        st.session_state.rag_pipeline.initialize_from_documents(new_documents)
                    else:
                        st.session_state.rag_pipeline.add_documents(new_documents)
                        
                    st.session_state.document_processed = True
                    st.success("✅ Document processed successfully!")
                    
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.session_state.document_processed = False
    else:  # URL input
        url = st.text_input("Enter URL of document:", key="url_input")
        if st.button("Process URL"):
            if not url:
                st.warning("⚠️ Please enter a URL")
            else:
                try:
                    with st.spinner("Processing URL content..."):
                        new_documents = preprocessor.process_file(url, is_url=True)
                        if st.session_state.rag_pipeline is None:
                            st.session_state.rag_pipeline = RAGPipeline()
                            st.session_state.rag_pipeline.initialize_from_documents(new_documents)
                        else:
                            st.session_state.rag_pipeline.add_documents(new_documents)
                            
                        st.session_state.document_processed = True
                        st.success("✅ Document processed successfully!")
                        
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.session_state.document_processed = False



# Add a clear button in sidebar
# with st.sidebar:
#     if st.button("❌ Clear All Documents"):
#         if st.session_state.rag_pipeline is not None:
#             st.session_state.rag_pipeline.cleanup()
#             st.session_state.rag_pipeline = None
#         st.session_state.file_uploader = None
#         st.session_state.document_processed = False
#         st.rerun()            


st.header("❓ Ask a Question")
question = st.text_input(
    "",
    key="question_input",
    label_visibility="collapsed",
    placeholder="Enter your question about the uploaded document"
)

if st.button("Get Answer", type="primary"):
    if not question:
        st.warning("⚠️ Please enter a question")
    elif not st.session_state.document_processed:
        st.warning("⚠️ Please upload and process a document first")
    else:
        with st.spinner("🔍 Searching for answer..."):
            try:
                st.session_state.chat_history.append({"role": "user", "content": question})
                result = st.session_state.rag_pipeline.query(question)
                st.session_state.chat_history.append({"role": "bot", "content": result['answer']})
                st.subheader("📝 Answer:")
                st.markdown(f"<div style='background-color: #4a4a4a; padding: 15px; border-radius: 5px;'>{result['answer']}</div>", 
                            unsafe_allow_html=True)
                st.subheader("🔍 Context Used:")
                st.markdown(f"<div class='context-box'>{result['context']}</div>", unsafe_allow_html=True)
                
                # Rerun to update the chat history display
                #st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

            
            

# Status indicator
st.sidebar.markdown("---")
if st.session_state.document_processed:
    st.sidebar.success("✅ Document ready for queries")
else:
    st.sidebar.warning("⚠️ No document loaded yet")
