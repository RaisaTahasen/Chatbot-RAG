# 📚 RAG Pipeline Chatbot – Documentation  
Overview  
This project implements a Retrieval-Augmented Generation (RAG) pipeline chatbot that enables users to upload documents or provide URLs, then ask natural-language questions about the content. The system processes multiple file formats, generates vector embeddings, and retrieves contextually relevant answers from the uploaded data.  

#### For the detailed setup guide please go through [Setup_Guide.pdf](https://github.com/RaisaTahasen/Chatbot-RAG/blob/main/Setup_Guide_Chatbot.pdf)

### ✨ Key Features  
📂 Document Processing  


Upload and process .csv, .txt, .docx, .sqlite / .db, .pdf, .jpeg, .jpg, .png files via the upload feature.  


Extract and process content from:  


Local file uploads  


URLs (web pages, PDFs, images)  


OCR support for scanned documents and images  


Automatic text chunking for optimal retrieval  


### 💬 Chat Interface  
Interactive Streamlit web interface  


Dark theme with responsive design  

 
Session history tracking  


Context-aware answers with source references  


Document status indicators (processing / ready)  


### 🔍 RAG Pipeline  
Multilingual sentence embeddings  


Local LLM integration via Ollama  


Vector similarity search for precise retrieval  


Context-aware prompting  


Full document source tracking  



### 🛠️ Technologies Used  
Core Libraries  
Streamlit – Web app framework  


LangChain – RAG pipeline orchestration  

 
Ollama – Local LLM inference  


HuggingFace – Sentence embeddings  

 
ChromaDB – Vector storage and retrieval  


### Document Processing   
PyMuPDF – PDF extraction  


python-docx – Word document handling  


pandas – CSV file processing  


sqlite3 – Database schema extraction  


pytesseract – OCR for images and scanned PDFs  
