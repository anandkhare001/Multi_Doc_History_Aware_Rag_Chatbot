import streamlit as st
from api_utils import upload_document, list_documents, delete_document, list_sessions, get_session_messages


def display_sidebar():
    """
    Display the sidebar UI with:
      1. New Chat button
      2. Past chat sessions (clickable to restore)
      3. Model selection
      4. Document upload, listing, and deletion
    """

    # ------------------------------------------------------------------
    # 1. New Chat Button
    # ------------------------------------------------------------------
    if st.sidebar.button("➕ New Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = None
        st.rerun()

    st.sidebar.divider()

    # ------------------------------------------------------------------
    # 2. Past Chat Sessions
    # ------------------------------------------------------------------
    st.sidebar.header("💬 Chat History")

    # Load sessions on first render or after a new chat is saved
    if "sessions" not in st.session_state:
        st.session_state.sessions = list_sessions()

    if st.sidebar.button("🔄 Refresh Chats", use_container_width=True):
        st.session_state.sessions = list_sessions()

    sessions = st.session_state.get("sessions", [])

    if sessions:
        for session in sessions:
            session_id = session["session_id"]
            preview = session["preview"]
            created_at = session.get("created_at", "")

            # Highlight the currently active session
            is_active = st.session_state.get("session_id") == session_id
            label = f"{'▶ ' if is_active else ''}{preview}"

            # Each session is a button; clicking it loads that session's messages
            if st.sidebar.button(label, key=f"session_{session_id}", use_container_width=True):
                messages = get_session_messages(session_id)
                if messages is not None:
                    st.session_state.messages = messages
                    st.session_state.session_id = session_id
                    st.rerun()
    else:
        st.sidebar.caption("No past chats yet. Start a conversation!")

    st.sidebar.divider()

    # ------------------------------------------------------------------
    # 3. Model Selection
    # ------------------------------------------------------------------
    st.sidebar.header("⚙️ Settings")
    model_options = ["gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini"]
    st.sidebar.selectbox("Select Model", options=model_options, key="model")

    st.sidebar.divider()

    # ------------------------------------------------------------------
    # 4. Document Management
    # ------------------------------------------------------------------
    st.sidebar.header("📂 Documents")

    # Upload
    uploaded_file = st.sidebar.file_uploader("Upload a document", type=["pdf", "docx", "html"])
    if uploaded_file and st.sidebar.button("⬆️ Upload", use_container_width=True):
        with st.spinner("Uploading..."):
            upload_response = upload_document(uploaded_file)
            if upload_response:
                st.sidebar.success(f"Uploaded successfully! ID: {upload_response['file_id']}")
                st.session_state.documents = list_documents()

    # List documents
    if st.sidebar.button("🔄 Refresh Document List", use_container_width=True):
        st.session_state.documents = list_documents()

    if "documents" not in st.session_state:
        st.session_state.documents = list_documents()

    if st.session_state.documents:
        for doc in st.session_state.documents:
            st.sidebar.text(f"📄 {doc['filename']} (ID: {doc['id']})")

        selected_file_name = st.sidebar.selectbox(
            "Select document to delete",
            options=[doc['filename'] for doc in st.session_state.documents]
        )

        if st.sidebar.button("🗑️ Delete Selected", use_container_width=True):
            delete_response = delete_document(selected_file_name)
            if delete_response:
                st.sidebar.success("Document deleted successfully.")
                st.session_state.documents = list_documents()
    else:
        st.sidebar.caption("No documents uploaded yet.")
