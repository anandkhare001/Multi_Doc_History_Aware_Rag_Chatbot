import requests
import streamlit as st
import os

BASE_URL = "http://localhost:8000"


# ----------------------------------
# Chat
# ----------------------------------
def get_api_response(question, session_id, model, embedding_model="openai"):
    """
    Send a question to the chat endpoint and return the JSON response.

    Args:
        question (str):        User's question.
        session_id (str):      Current session ID (None for new session).
        model (str):           Chat model — "gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini",
                               or "gemma:2b-instruct-q4" (local via Ollama).
        embedding_model (str): "openai" or "nomic-embed-text" (local via Ollama).
                               Must match the model used when the document was uploaded.
    """
    headers = {'accept': 'application/json', 'Content-Type': 'application/json'}
    data = {
        "question": question,
        "model": model,
        "embedding_model": embedding_model
    }
    if session_id:
        data["session_id"] = session_id

    try:
        response = requests.post(f"{BASE_URL}/chat", headers=headers, json=data)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API request failed with status code {response.status_code}: {response.text}")
            return None
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None


# ----------------------------------
# Documents
# ----------------------------------
def upload_document(file, embedding_model="openai"):
    """
    Upload a file to the backend and index it with the chosen embedding model.

    Args:
        file:                  Streamlit UploadedFile object.
        embedding_model (str): "openai" or "nomic-embed-text".
                               Determines which vectorstore the chunks go into.
    """
    try:
        files = {"file": (file.name, file, file.type)}
        # embedding_model sent as form field alongside the file
        data = {"embedding_model": embedding_model}
        response = requests.post(f"{BASE_URL}/upload-doc", files=files, data=data)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to upload file. Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"An error occurred while uploading the file: {str(e)}")
        return None


def list_documents():
    """Fetch the list of all uploaded documents."""
    try:
        response = requests.get(f"{BASE_URL}/list-docs")
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to fetch document list. Error: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        st.error(f"An error occurred while fetching the document list: {str(e)}")
        return []


def delete_document(file_name):
    """Delete a document by filename via the backend API."""
    headers = {'accept': 'application/json', 'Content-Type': 'application/json'}
    data = {"file_name": file_name}
    try:
        response = requests.post(f"{BASE_URL}/delete-doc", headers=headers, json=data)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to delete document. Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"An error occurred while deleting the document: {str(e)}")
        return None


# ----------------------------------
# Chat Sessions  (NEW)
# ----------------------------------
def list_sessions():
    """
    Fetch all past chat sessions from the backend.

    Returns:
        List of dicts with keys: session_id, preview, created_at.
        Returns an empty list on failure.
    """
    try:
        response = requests.get(f"{BASE_URL}/sessions")
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to fetch sessions. Error: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        st.error(f"An error occurred while fetching sessions: {str(e)}")
        return []


def get_session_messages(session_id):
    """
    Fetch the full message history for a given session_id.

    Returns:
        List of dicts with 'role' and 'content' keys, or None on failure.
    """
    try:
        response = requests.get(f"{BASE_URL}/sessions/{session_id}/messages")
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            st.warning("Session not found.")
            return None
        else:
            st.error(f"Failed to fetch session messages. Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"An error occurred while fetching session messages: {str(e)}")
        return None
