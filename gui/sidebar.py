import streamlit as st
from api_utils import upload_document, list_documents, delete_document


def display_sidebar():
    """
    Display the sidebar UI for model selection, document upload,
    document listing, and document deletion functionalities.
    """
    # Model selection
    # Allow users to select the desired GPT model from a list of options.
    model_options = ["gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini"]
    st.sidebar.selectbox("select Model", options=model_options, key="model")

    # Document upload
    # Provide a file uploader widget for users to upload documents in PDF, DOCX, or HTML format.
    # When a file is uploaded and the Upload button is clicked, upload the document using the API,
    # then update the document list in the session state.
    uploaded_file = st.sidebar.file_uploader("Choose a file", type=["pdf", "docx", "html"])
    if uploaded_file and st.sidebar.button("Upload"):
        with st.spinner("Uploading..."):
            upload_response = upload_document(uploaded_file)
            if upload_response:
                st.sidebar.success(f"File uploaded successfully with ID {upload_response['file_id']}.")
                st.session_state.documents = list_documents()


    # List and delete documents
    # Display a header and a button to refresh the list of uploaded documents.
    st.sidebar.header("Uploaded Documents")
    if st.sidebar.button("Refresh Document List"):
        st.session_state.documents = list_documents()

    # Display document list and delete functionality
    # Show all uploaded documents with their filenames and IDs.
    # Provide a select box to choose a document to delete, along with a button to delete the selected document.
    # On successful deletion, update the document list in the session state.
    if "documents" in st.session_state and st.session_state.documents:
        for doc in st.session_state.documents:
            st.sidebar.text(f"{doc['filename']} (ID: {doc['id']})")

        selected_file_name = st.sidebar.selectbox("Select a document to delete",
                                            options=[doc['filename'] for doc in st.session_state.documents])

        if st.sidebar.button("Delete Selected Document"):
            delete_response = delete_document(selected_file_name)
            if delete_response:
                st.sidebar.success(f"Document deleted successfully.")
                st.session_state.documents = list_documents()