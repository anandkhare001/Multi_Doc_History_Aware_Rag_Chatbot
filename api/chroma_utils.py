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
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Ollama base URL — overridden by Docker env var when running in containers
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Chroma persistence directory — overridden by Docker env var
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)

# ── Embedding Factory ──────────────────────────────────────────────────────────

def get_embedding_function(embedding_model: str = "openai"):
    """
    Return the appropriate embedding function based on the selected model.

    Args:
        embedding_model (str): "openai"           → OpenAI text-embedding-ada-002
                               "nomic-embed-text" → Ollama local embedding

    Returns:
        An embeddings object compatible with LangChain vectorstores.
    """
    if embedding_model == "nomic-embed-text":
        return OllamaEmbeddings(
            model="nomic-embed-text",
            base_url=OLLAMA_BASE_URL
        )
    # Default: OpenAI
    return OpenAIEmbeddings()

def get_vectorstore(embedding_model: str = "openai") -> Chroma:
    """
    Return a Chroma vectorstore initialised with the chosen embedding function.
    Each embedding model gets its own subdirectory so embeddings don't mix.

    Args:
        embedding_model (str): "openai" or "nomic-embed-text"

    Returns:
        Chroma vectorstore instance.
    """
    # Separate persist dirs prevent dimension mismatch errors when switching models
    subdir = "openai" if embedding_model == "openai" else "nomic"
    persist_dir = os.path.join(CHROMA_PERSIST_DIR, subdir)

    return Chroma(
        persist_directory=persist_dir,
        embedding_function=get_embedding_function(embedding_model)
    )


# Default vectorstore (OpenAI) — used by langchain_utils for backwards compatibility
vectorstore = get_vectorstore("openai")


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


# ── Indexing ───────────────────────────────────────────────────────────────────

def index_document_to_chroma(file_path: str, file_id: int,
                              embedding_model: str = "openai") -> bool:
    """
    Index a document into the Chroma vectorstore using the chosen embedding model.

    Args:
        file_path (str):       Path to the document.
        file_id (int):         Unique file identifier stored as metadata.
        embedding_model (str): "openai" or "nomic-embed-text"

    Returns:
        bool: True on success, False on failure.
    """
    try:
        document = load_documents(file_path)
        splits = split_documents(document)
        for split in splits:
            split.metadata['file_id'] = file_id

        vs = get_vectorstore(embedding_model)
        vs.add_documents(splits)
        return True
    except Exception as e:
        print(f"Error indexing document: {e}")
        return False


# ── Deletion ───────────────────────────────────────────────────────────────────

def delete_document_index_from_chroma(file_id: int, embedding_model: str = "openai") -> bool:
    """
    Delete all chunks for a given file_id from the appropriate vectorstore.

    Args:
        file_id (int):         The file identifier to delete.
        embedding_model (str): "openai" or "nomic-embed-text"

    Returns:
        bool: True on success, False on failure.
    """
    try:
        vs = get_vectorstore(embedding_model)
        vs.delete(where={"file_id": file_id})
        print(f"Deleted all documents with file_id {file_id} from {embedding_model} store")
        return True
    except Exception as e:
        print(f"Error deleting document with file_id {file_id}: {str(e)}")
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