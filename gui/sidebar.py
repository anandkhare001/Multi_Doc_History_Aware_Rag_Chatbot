import streamlit as st
from api_utils import upload_document, list_documents, delete_document, list_sessions, get_session_messages


# Model options grouped for display
CHAT_MODELS = {
    "🌐 API Models": ["gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini"],
    "💻 Local Models (Ollama)": ["gemma:2b-instruct"],
}

EMBEDDING_MODELS = {
    "🌐 API Embedding": ["openai"],
    "💻 Local Embedding (Ollama)": ["nomic-embed-text"],
}

# Flat lists for selectbox options
ALL_CHAT_MODELS = [m for group in CHAT_MODELS.values() for m in group]
ALL_EMBEDDING_MODELS = [m for group in EMBEDDING_MODELS.values() for m in group]


def display_sidebar():
    """
    Display the sidebar UI with:
      1. New Chat button
      2. Past chat sessions (clickable to restore)
      3. Model settings — chat model + embedding model (API vs Local)
      4. Document upload, listing, and deletion
    """

    # ── 1. New Chat ────────────────────────────────────────────────────────────
    if st.sidebar.button("➕ New Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = None
        st.rerun()

    st.sidebar.divider()

    # ── 2. Chat History ────────────────────────────────────────────────────────
    st.sidebar.header("💬 Chat History")

    if "sessions" not in st.session_state:
        st.session_state.sessions = list_sessions()

    if st.sidebar.button("🔄 Refresh Chats", use_container_width=True):
        st.session_state.sessions = list_sessions()

    sessions = st.session_state.get("sessions", [])
    if sessions:
        for session in sessions:
            session_id = session["session_id"]
            preview = session["preview"]
            is_active = st.session_state.get("session_id") == session_id
            label = f"{'▶ ' if is_active else ''}{preview}"
            if st.sidebar.button(label, key=f"session_{session_id}", use_container_width=True):
                messages = get_session_messages(session_id)
                if messages is not None:
                    st.session_state.messages = messages
                    st.session_state.session_id = session_id
                    st.rerun()
    else:
        st.sidebar.caption("No past chats yet. Start a conversation!")

    st.sidebar.divider()

    # ── 3. Model Settings ──────────────────────────────────────────────────────
    st.sidebar.header("⚙️ Model Settings")

    # Chat model selector with grouped labels as disabled separators
    st.sidebar.markdown("**Chat Model**")
    chat_model_display = []
    chat_model_map = {}     # display label → actual value
    for group_label, models in CHAT_MODELS.items():
        chat_model_display.append(f"── {group_label} ──")   # group header (non-selectable look)
        for m in models:
            chat_model_display.append(m)
            chat_model_map[m] = m

    selected_chat = st.sidebar.selectbox(
        "Chat Model",
        options=ALL_CHAT_MODELS,
        format_func=lambda x: (
            f"🌐 {x}" if x in CHAT_MODELS["🌐 API Models"]
            else f"💻 {x} (local)"
        ),
        key="model",
        label_visibility="collapsed"
    )

    # Embedding model selector
    st.sidebar.markdown("**Embedding Model**")
    selected_embedding = st.sidebar.selectbox(
        "Embedding Model",
        options=ALL_EMBEDDING_MODELS,
        format_func=lambda x: (
            f"🌐 {x} (OpenAI)" if x == "openai"
            else f"💻 {x} (local)"
        ),
        key="embedding_model",
        label_visibility="collapsed"
    )

    # Warn user if chat and embedding backends are mixed (not an error, just advisory)
    chat_is_local = selected_chat in CHAT_MODELS["💻 Local Models (Ollama)"]
    embed_is_local = selected_embedding in EMBEDDING_MODELS["💻 Local Embedding (Ollama)"]
    if chat_is_local != embed_is_local:
        st.sidebar.warning(
            "⚠️ Mixed mode: one model is local and the other uses an API. "
            "This works but requires both Ollama and an API key to be available."
        )

    st.sidebar.divider()

    # ── 4. Document Management ─────────────────────────────────────────────────
    st.sidebar.header("📂 Documents")

    st.sidebar.caption(
        f"Documents will be indexed using: "
        f"**{'💻 ' + selected_embedding + ' (local)' if embed_is_local else '🌐 OpenAI'}**"
    )

    uploaded_file = st.sidebar.file_uploader("Upload a document", type=["pdf", "docx", "html"])
    if uploaded_file and st.sidebar.button("⬆️ Upload", use_container_width=True):
        with st.spinner("Uploading and indexing..."):
            upload_response = upload_document(uploaded_file, embedding_model=selected_embedding)
            if upload_response:
                st.sidebar.success(
                    f"Uploaded! ID: {upload_response['file_id']} | "
                    f"Indexed with: {upload_response.get('embedding_model', selected_embedding)}"
                )
                st.session_state.documents = list_documents()

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
