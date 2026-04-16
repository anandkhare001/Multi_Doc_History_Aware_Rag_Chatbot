from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic_models import QueryInput, QueryResponse, DocumentInfo, DeleteFileRequest
from langchain_utils import get_rag_chain
from db_utils import (insert_application_logs, get_chat_history, get_all_documents,
                      insert_document_record, delete_document_record, get_document_details,
                      get_all_sessions, get_session_messages)
#from pinecone_utils import *
from chroma_utils import *
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
    session_id = query_input.session_id or str(uuid.uuid4())
    logging.info(f"Session ID: {session_id}, User Query: {query_input.question}, Model: {query_input.model.value}")

    chat_history = get_chat_history(session_id)
    rag_chain = get_rag_chain(query_input.model.value)
    answer = rag_chain.invoke({
        "input": query_input.question,
        "chat_history": chat_history
    })['answer']

    insert_application_logs(session_id, query_input.question, answer, query_input.model.value)
    logging.info(f"Session ID: {session_id}, AI Response: {answer}")
    return QueryResponse(answer=answer, session_id=session_id, model=query_input.model)


# Document Upload Endpoint
@app.post("/upload-doc")
def upload_and_index_document(file: UploadFile = File(...)):
    allowed_extensions = ['.pdf', '.docs', '.html']
    file_extension = os.path.splitext(file.filename)[1].lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed types are: {', '.join(allowed_extensions)}")

    temp_file_path = f"temp_{file.filename}"

    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        file_id = insert_document_record(file.filename)
        documents = load_documents(temp_file_path)
        documents_splits = split_documents(documents)
        success = index_documents_to_chroma("multi-doc-rag", documents_splits, file.filename)

        if success:
            return {"message": f"File {file.filename} has been successfully uploaded and indexed.", "file_id": file_id}
        else:
            delete_document_record(file_id)
            raise HTTPException(status_code=500, detail=f"Failed to index {file.filename}.")
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
    delete_document_record(request.file_name)
    delete_document_index_from_chroma("multi-doc-rag", request.file_name)
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
   