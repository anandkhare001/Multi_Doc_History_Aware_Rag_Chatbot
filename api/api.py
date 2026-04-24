from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from pydantic_models import QueryInput, QueryResponse, DocumentInfo, DeleteFileRequest
from langchain_utils import get_rag_chain
from db_utils import (insert_application_logs, get_chat_history, get_all_documents,
                      insert_document_record, delete_document_record, get_document_details,
                      get_all_sessions, get_session_messages)
#from pinecone_utils import *
from chroma_utils import index_document_to_chroma, delete_document_index_from_chroma, load_documents, split_documents
import os
import uuid
import logging
import shutil


# Set up logging
logging.basicConfig(filename='app.log', level=logging.INFO)

# Initialize FastAPI app
app = FastAPI(
    title="Multi Doc RAG",
    version="1.0",
    description="Rag API Server"
)

@app.get("/")
def read_root():
    return {"message": "Welcome to Multi Doc RAG API!"}


# Chat Endpoint
@app.post("/chat", response_model=QueryResponse)
def chat(query_input: QueryInput):
    """
    Handle a chat query.

    Routing:
      - model = "gemma:2b-instruct"  → Ollama (local)
      - model = "gpt-*"                 → OpenAI (API)

      - embedding_model = "nomic-embed-text" → Ollama nomic-embed-text (local)
      - embedding_model = "openai"           → OpenAI text-embedding-ada-002
    """
    session_id = query_input.session_id or str(uuid.uuid4())
    logging.info(
        f"Session: {session_id} | Model: {query_input.model.value} "
        f"| Embedding: {query_input.embedding_model.value} "
        f"| Query: {query_input.question}"
    )

    chat_history = get_chat_history(session_id)
    rag_chain = get_rag_chain(
        model=query_input.model.value,
        embedding_model=query_input.embedding_model.value
    )
    answer = rag_chain.invoke({
        "input": query_input.question,
        "chat_history": chat_history
    })['answer']

    insert_application_logs(session_id, query_input.question, answer, query_input.model.value)
    logging.info(f"Session: {session_id} | Response: {answer}")
    return QueryResponse(answer=answer, session_id=session_id, model=query_input.model)


# Document Upload Endpoint
@app.post("/upload-doc")
def upload_and_index_document(
    file: UploadFile = File(...),
    embedding_model: str = Form(default="openai")   # "openai" or "nomic-embed-text"
):
    """
    Upload a document and index it using the chosen embedding model.

    The embedding_model used here MUST match the one used at query time
    so the retriever searches the correct vectorstore.
    """
    allowed_extensions = ['.pdf', '.docx', '.html']
    file_extension = os.path.splitext(file.filename)[1].lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )

    # Validate embedding_model value
    valid_embeddings = {"openai", "nomic-embed-text"}
    if embedding_model not in valid_embeddings:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid embedding_model. Choose from: {valid_embeddings}"
        )

    temp_file_path = f"temp_{file.filename}"
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        file_id = insert_document_record(file.filename)
        success = index_document_to_chroma(temp_file_path, file_id, embedding_model)

        if success:
            return {
                "message": f"File '{file.filename}' uploaded and indexed successfully.",
                "file_id": file_id,
                "embedding_model": embedding_model
            }
        else:
            delete_document_record(file_id)
            raise HTTPException(status_code=500, detail=f"Failed to index '{file.filename}'.")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


# List Documents Endpoint
@app.get("/list-docs", response_model=list[DocumentInfo])
def list_documents():
    return get_all_documents()


# Delete Document Endpoint
@app.post("/delete-doc")
def delete_document(request: DeleteFileRequest):
    """Delete a document record and remove its vectors from both vectorstores."""
    delete_document_record(request.file_name)
    # Remove from both stores in case it was re-indexed with a different model
    delete_document_index_from_chroma(request.file_name, "openai")
    delete_document_index_from_chroma(request.file_name, "nomic-embed-text")
    return {"message": f"Document '{request.file_name}' deleted successfully."}


# --- NEW: List all chat sessions ---
# Returns a list of all unique chat sessions with their first message as a preview label.
# Usage: GET /sessions
@app.get("/sessions")
def list_sessions():
    """
    Retrieve all distinct chat sessions.
    Returns a list of dicts with:
      - session_id
      - preview: first user message of the session (used as label in the UI)
      - created_at: timestamp of the first message in the session
    """
    return get_all_sessions()


# --- NEW: Get full message history for a session ---
# Returns all messages (user + AI) for a given session_id.
# Usage: GET /sessions/{session_id}/messages
@app.get("/sessions/{session_id}/messages")
def get_session_history(session_id: str):
    """
    Retrieve full chat history for a specific session.
    Returns a list of message dicts with role ('user' or 'assistant') and content.
    """
    messages = get_session_messages(session_id)
    if messages is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return messages


# Uvicorn entrypoint
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
   