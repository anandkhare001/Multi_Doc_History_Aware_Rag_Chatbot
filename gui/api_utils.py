import requests
import streamlit as st


# ----------------------------------
# Functions to interact with the backend API
# ----------------------------------

# get_api_response: sends a question along with session id and model name to the backend API to get a chat response
# Parameters:
#  - question: the user's question string
#  - session_id: session identifier string to maintain conversation context
#  - model: the model name to use for generating response
# Returns:
#  - JSON response from API if successful, None otherwise

def get_api_response(question, session_id, model):
    headers = {'accept': 'application/json', 'Content-Type': 'application/json'}
    data = {"question": question, "model": model}
    if session_id:
        data["session_id"] = session_id
    
    try:
        response = requests.post("http://localhost:8000/chat", headers=headers, json=data)
        if response.status_code ==  200:
            return response.json()
        else:
            st.error(f"API request failed with status code {response.status_code}: {response.text}")
            return None
    except Exception as e:
        st.error(f"An error occured: {str(e)}")
        return None
    

# upload_document: uploads a file to the backend API
# Parameters:
#  - file: file object to upload
# Returns:
#  - JSON response from API if upload is successful, None otherwise

def upload_document(file):
    try:
        files = {"file": (file.name, file, file.type)}
        response = requests.post("http://localhost:8000/upload-doc", files=files)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to upload file. Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"An error occurred while uploading the file: {str(e)}")
        return None
    

# list_documents: fetches the list of uploaded documents from backend API
# Returns:
#  - list of documents if successful, empty list otherwise

def list_documents():
    try:
        response = requests.get("http://localhost:8000/list-docs")
        if response.status_code ==200:
            return response.json()
        else:
            st.error(f"Failed to fetch document list. Error: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        st.error(f"An error occurred while fetching the document list: {str(e)}")
        return []
    

# delete_document: deletes a document by filename via backend API
# Parameters:
#  - file_name: name of the file to delete
# Returns:
#  - JSON response from API if successful, None otherwise

def delete_document(file_name):
    headers = {'accept': 'application/json', 'Content-Type': 'application/json'}
    data = {"file_name": file_name}

    try: 
        response = requests.post("http://localhost:8000/delete-doc", headers=headers, json=data)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to delete document. Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"An error occurred while deleting the document: {str(e)}")
        return None