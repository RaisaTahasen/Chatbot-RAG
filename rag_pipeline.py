from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough 
from langchain_core.output_parsers import StrOutputParser 
from langchain_core.documents import Document
from typing import List
import os
import torch
from dotenv import load_dotenv

load_dotenv()

class RAGPipeline:
    def __init__(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            model_kwargs={"device": device },
            encode_kwargs={"normalize_embeddings": True}
        )
        
        self.llm = OllamaLLM(
            model="phi3:mini",
            temperature=0.3,
            num_gpu=1 if device == 'cuda' else 0
        )
        
        self.prompt = ChatPromptTemplate.from_template(""" 
            Answer the question based only on the following context:
            {context}
            
            Question: {question}
            
            If you don't know the answer, just say you don't know.
            Provide a concise and accurate response.
        """)
        
        self.vector_store = None
        self.retriever = None
        self.chain = None

    

    def initialize_from_documents(self, documents: List[Document]):
        """Initialize the RAG pipeline with documents."""
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )
        
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        
        self.chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )



    def query(self, question: str) -> str:
        """Query the RAG pipeline."""
        if not self.chain:
            raise ValueError("Pipeline not initialized. Load documents first.")
        
        try:
            return self.chain.invoke(question)
        except Exception as e:
            return f"Error processing your query: {str(e)}"