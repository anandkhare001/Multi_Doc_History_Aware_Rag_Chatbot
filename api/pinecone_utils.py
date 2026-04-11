"""
Module: pinecone_utils

This module provides utility functions to load, split, index, and delete documents and their embeddings using Pinecone vector database.

Functionalities:
- Load documents from PDF and DOCX files
- Split documents into manageable chunks for embedding
- Index document chunks to Pinecone
- Delete indexed documents by filename from Pinecone

Requirements:
- Pinecone API key and OpenAI API key should be set in environment variables
- Supported document formats: PDF, DOCX

"""

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from dotenv import load_dotenv
load_dotenv()

def split_documents(documents):
     text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
        )
     splits = text_splitter.split_documents(documents)
     return splits

def load_documents(filepath: str) -> List[Document]:
    path = Path(filepath)
    filename = path.name
    if filepath.endswith('.pdf'):
        loader = PyPDFLoader(filepath)
    elif filepath.endswith('.docx'):
        loader = Docx2txtLoader(filepath)
    else:
        print(f"Unsḍupported file type: {filepath}")
    documents = loader.load()
    for document in documents:
        document.metadata['file_name'] = filename
    return documents

def index_documents_to_pinecone(index_name, documents_splits, file_name):
    try:
        ## Vector Search DB In Pinecone
        # Import the Pinecone library
        from pinecone import Pinecone, ServerlessSpec
        from langchain.embeddings.openai import OpenAIEmbeddings    

        # Initialize a Pinecone client with your API key
        pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
        embeddings=OpenAIEmbeddings(api_key=os.environ['OPENAI_API_KEY'])

        # Create a dense index with integrated embedding    
        if not pc.has_index(index_name):
            pc.create_index(
                name=index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )

        index = pc.Index(index_name)

        vectors = []

        for chunk in documents_splits:
            
            text = chunk.page_content
            embedding = embeddings.embed_query(text)      
            #path = Path(chunk.metadata["source"])
            #filename = path.name

            vectors.append({
                "id": str(uuid.uuid4()),
                "values": embedding,
                "metadata": {
                    "text": text,
                    'file_name': file_name,
                    "chunk":  + 1,
                    "source": chunk.metadata.get("source"),
                    "page": chunk.metadata.get("page")
                }
            })

        # -----------------------------
        # Upsert to Pinecone
        # -----------------------------
        batch_size = 50

        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i+batch_size]
            index.upsert(vectors=batch)

        print("PDF indexed successfully!")
        return True
    except Exception as e:
        print(f"Error indexing document with file_id {file_name}: {e}")
        return False

def delete_document_index_from_pinecone(index_name, file_name=None):
    try:
        from pinecone import Pinecone

        pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
        index = pc.Index(index_name)
        if file_name is not None:
            index.delete(
                filter={
                    "file_name": file_name
                }
            )
        print(f"Deleted all documents with file_id {file_name}")
        return True
    except Exception as e:
        print(f"Error deleting document with file_id {file_name} from Pinecone: {str(e)}")
        return False

if __name__ == "__main__":
    folder_path = os.path.join(os.getcwd(),  "content", "docs")
    file_list = os.listdir(folder_path)
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        documents = load_documents(file_path) 
        documents_splits = split_documents(documents)
        index_documents_to_pinecone("multi-doc-rag", documents_splits, filename)
    # delete_document_index_from_pinecone("multi-doc-rag", "Welcome to our extensive MLOps Bootcamp.pdf")


#file_path = 'C:\\Users\\Admin\\Documents\\ML_Projects\\GenAI\\RAG_Langchain\\content\\docs\\Welcome to our extensive MLOps Bootcamp.pdf'       
#documents = load_documents(file_path, 2) 
#documents_splits = split_documents(documents)
#index_documents_to_pinecone("multi-doc-rag", documents_splits, 2)
#delete_document_index_from_pinecone("multi-doc-rag", 2)
#delete_document_index_from_pinecone("multi-doc-rag", "Welcome to our extensive MLOps Bootcamp.pdf")