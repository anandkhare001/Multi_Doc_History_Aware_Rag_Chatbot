from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic_models import QueryInput, QueryResponse, DocumentInfo, DeleteFileRequest
from langchain_utils import get_rag_chain
from db_utils import insert_application_logs, get_chat_history, get_all_documents, insert_document_record, delete_document_record, get_document_details
# from chroma_utils import index_document_to_chroma, delete_docs_from_chroma
from pinecone_utils import *
import os
import uuid
import logging
import shutil


# Set up logging
logging.basicConfig(filename = 'app.log', level=logging.INFO)

# Initialize FastAPI app
app = FastAPI(
    title = "Multi Doc RAG",
    version = "1.0",
    description = "Rag API Server"
)

@app.get("/")
# Root endpoint to verify the API service is running
# Returns a welcome message
# Usage: GET /
def read_root():
    return {"message": "Welcome to Multi Doc RAG API!"}

# Chat Endpoint
# Handles chat queries for document-based question answering
# Params:
# - query_input: QueryInput Pydantic model containing question, session_id and model selection
# Workflow:
# 1. Use existing or generate new session ID
# 2. Retrieve chat history for session
# 3. Invoke retrieval-augmented generation (RAG) model chain with query and chat history
# 4. Log user query and AI response
# 5. Return AI's answer and session information in QueryResponse model
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
# Handles uploading and indexing of documents
# Params:
# - file: Uploaded file from client
# Workflow:
# 1. Validate file extension for allowed types (pdf, docs, html)
# 2. Temporarily save the uploaded file locally
# 3. Insert document record in database and get file_id
# 4. Load, split, and index the document content to Pinecone vector DB
# 5. Remove temporary local file regardless of success or failure
# Returns success message and file_id or raises HTTPException on failure
@app.post("/upload-doc")
def upload_and_index_document(file: UploadFile = File(...)):
    allowed_extensions = ['.pdf', '.docs', '.html']
    file_extension = os.path.splitext(file.filename)[1].lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed types are: {', '.join(allowed_extensions)}")
    
    temp_file_path = f"temp_{file.filename}"

    try:
        # Save the uploaded file to a temporary file
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        file_id = insert_document_record(file.filename)
        documents = load_documents(temp_file_path) 
        documents_splits = split_documents(documents)
        success = index_documents_to_pinecone("multi-doc-rag", documents_splits, file.filename)

        if success:
            return {"message": f"File {file.filename} has been successfully uploaded and indexed.", "file_id": file_id}
        else:
            delete_document_record(file_id)
            raise HTTPException(status_code=500, detail=f"Failed to index {file.filename}.")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)



# List Documents Endpoint:
# Returns a list of metadata for all documents in the system
# Returns list of DocumentInfo Pydantic models
# Usage: GET /list-docs
@app.get("/list-docs", response_model = list[DocumentInfo])
def list_documents():
    return get_all_documents()






# Uvicorn entrypoint for `python main.py` (optional)
# This allows running the FastAPI app directly via `python api.py`
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
