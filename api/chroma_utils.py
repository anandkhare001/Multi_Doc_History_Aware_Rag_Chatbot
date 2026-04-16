"""
Module: chroma_utils

This module provides utility functions to load, split, index, and delete documents
using ChromaDB vector database.

Functionalities:
- Load documents from PDF and DOCX files
- Split documents into manageable chunks for embedding
- Index document chunks to ChromaDB
- Delete indexed documents by filename from ChromaDB

Requirements:
- OpenAI API key should be set in environment variables
- Supported document formats: PDF, DOCX
"""

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(api_key=os.environ['OPENAI_API_KEY'])

persist_directory = os.path.join(os.getcwd(), "chroma_db")

# Initialize Chroma
vectordb = Chroma(
            collection_name="multi-doc-rag",
            embedding_function=embeddings,
            persist_directory=persist_directory
        )

# -----------------------------
# Split Documents
# -----------------------------
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_documents(documents)


# -----------------------------
# Load Documents
# -----------------------------
def load_documents(filepath: str) -> List[Document]:
    path = Path(filepath)
    filename = path.name

    if filepath.endswith('.pdf'):
        loader = PyPDFLoader(filepath)
    elif filepath.endswith('.docx'):
        loader = Docx2txtLoader(filepath)
    else:
        raise ValueError(f"Unsupported file type: {filepath}")

    documents = loader.load()

    for doc in documents:
        doc.metadata['file_name'] = filename

    return documents


# -----------------------------
# Index to ChromaDB
# -----------------------------
def index_documents_to_chroma(collection_name, documents_splits, file_name):
    try:
        # Add metadata properly
        for i, chunk in enumerate(documents_splits):
            chunk.metadata.update({
                "file_name": file_name,
                "chunk": i + 1
            })

        vectordb.add_documents(documents_splits)
        print(f"{file_name} indexed successfully!")
        return True

    except Exception as e:
        print(f"Error indexing document {file_name}: {e}")
        return False


# -----------------------------
# Delete from ChromaDB
# -----------------------------
def delete_document_index_from_chroma(collection_name, file_name=None):
    try:
        if file_name:
            vectordb._collection.delete(
                where={"file_name": file_name}
            )
            print(f"Deleted documents with file_name: {file_name}")
        else:
            print("No file_name provided. Nothing deleted.")

        return True

    except Exception as e:
        print(f"Error deleting document {file_name}: {e}")
        return False


# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    #folder_path = os.path.join(os.getcwd(), "content", "docs")

    #for filename in os.listdir(folder_path):
    #    file_path = os.path.join(folder_path, filename)

    #    documents = load_documents(file_path)
    #    documents_splits = split_documents(documents)

    #    index_documents_to_chroma("multi-doc-rag", documents_splits, filename)

    # Example delete
    delete_document_index_from_chroma("multi-doc-rag", "speech.pdf")