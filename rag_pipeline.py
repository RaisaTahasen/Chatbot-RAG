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
from pathlib import Path
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

    def add_documents(self, new_documents: List[Document]):
        """Add new documents to existing vector store."""
        if self.vector_store is None:
            self.initialize_from_documents(new_documents)
        else:
            # Get existing collection name
            collection = self.vector_store._collection.name
            
            # Add new documents to existing Chroma collection
            Chroma.from_documents(
                documents=new_documents,
                embedding=self.embeddings,
                collection_name=collection,
                persist_directory="./chroma_db"
            )
            
            # Refresh retriever
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            )

    def cleanup(self):
        """Release resources and clean up."""
        if self.vector_store is not None:
            self.vector_store.delete_collection()
            self.vector_store = None
        self.retriever = None
        self.chain = None
        torch.cuda.empty_cache()

    def query(self, question: str) -> str:
        """Query the RAG pipeline."""
        if not self.chain:
            raise ValueError("Pipeline not initialized. Load documents first.")
        
      
        try:
        # First retrieve documents
            docs = self.retriever.invoke(question)
            unique_docs = []
            seen_content = set() 
            for doc in docs: 
                if doc.page_content not in seen_content:
                    seen_content.add(doc.page_content)
                    unique_docs.append(doc)

            # Format context with page info
            context_parts = []
            source_info = set()
            for doc in unique_docs:
                source = doc.metadata.get("source", "unknown")
                source_name = Path(source).name if not source.startswith('http') else source
                page = doc.metadata.get("page", "N/A")
                context_parts.append(f"From {source} (page {page}):\n{doc.page_content}")
                source_info.add(f"{source_name}, page {page}")
            
            # Get the answer
            answer = self.chain.invoke(question)
            
            return {
                "answer": answer,
                "context": "\n\n---\n\n".join(context_parts),
                "sources": list(source_info)
            }
            
        except Exception as e:
            return {
                "answer": f"Error: {str(e)}",
                "context": "",
                "sources": []
            }
