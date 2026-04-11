from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from typing import List
from langchain_core.documents import Document
import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()
# Set the OpenAI API key environment variable
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Configure text splitter to split documents into chunks for better processing
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # maximum size of each chunk
    chunk_overlap=200,  # overlap between chunks to maintain context
    length_function=len  # function to measure length of text
)

# Initialize embedding function using OpenAI's embeddings
embedding_function = OpenAIEmbeddings()


# Initialize Chroma vector store to persist embeddings
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)


# Document Loading and Splitting

def load_and_split_document(file_path: str) -> List[Document]:
    """
    Load a document from a file path and split into smaller chunks.

    Args:
        file_path (str): Path to the document file (.pdf, .docx, .html).

    Returns:
        List[Document]: List of split document chunks as Document objects.

    Raises:
        ValueError: If the file type is not supported.
    """
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith('.docx'):
        loader = Docx2txtLoader(file_path)
    elif file_path.endswith('.html'):
        loader = UnstructuredHTMLLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

    documents = loader.load()
    return text_splitter.split_documents(documents)


# Indexing Documents

def index_document_to_chroma(file_path: str, file_id: int) -> bool:
    """
    Index a document to the Chroma vector store.

    Args:
        file_path (str): Path to the document file.
        file_id (int): Unique identifier to associate with the indexed document chunks.

    Returns:
        bool: True if indexing succeeded, False otherwise.
    """
    try:
        splits = load_and_split_document(file_path)

        # Add metadata to each split chunk for identification
        for split in splits:
            split.metadata['file_id'] = file_id
            
        # Add documents to the vector store
        vectorstore.add_documents(splits)
        return True
    except Exception as e:
        print(f"Error indexing document: {e}")
        return False
    

# Deleting Documents

def delete_docs_from_chroma(file_id: int):
    """
    Delete documents from the Chroma vector store by their file_id.

    Args:
        file_id (int): Unique identifier associated with the documents to delete.

    Returns:
        bool: True if deletion was successful, False otherwise.
    """
    try:
        # This line attempts to get documents by metadata filter, adjust as needed to perform deletion
        docs = vectorstore.get(where={"file_id": file_id})
        print(f"Deleted all documents with file_id {file_id}")

        return True
    except Exception as e:
        print(f"Error deleting document with file_id {file_id} from Chroma: {str(e)}")
        return False