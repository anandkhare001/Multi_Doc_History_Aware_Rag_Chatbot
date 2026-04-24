import streamlit as st
from api_utils import get_api_response, list_sessions


def display_chat_interface():
    """
    Display the main chat interface.
    Passes both chat model and embedding model to the API on every request.
    """
    # Active session info
    if st.session_state.get("session_id"):
        st.caption(f"Session: `{st.session_state.session_id}`")
    else:
        st.caption("New conversation — session will be created on your first message.")

    # Show active model selections as info badges
    chat_model = st.session_state.get("model", "gpt-3.5-turbo")
    embedding_model = st.session_state.get("embedding_model", "openai")
    is_local_chat = chat_model == "gemma:2b-instruct-q4"
    is_local_embed = embedding_model == "nomic-embed-text"

    col1, col2 = st.columns(2)
    col1.info(f"{'💻 Local' if is_local_chat else '🌐 API'} Chat: `{chat_model}`")
    col2.info(f"{'💻 Local' if is_local_embed else '🌐 API'} Embed: `{embedding_model}`")

    # Render existing messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle new user input
    if prompt := st.chat_input("Ask anything about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner(
            "Thinking locally... 💻" if is_local_chat else "Calling API... 🌐"
        ):
            response = get_api_response(
                question=prompt,
                session_id=st.session_state.session_id,
                model=chat_model,
                embedding_model=embedding_model
            )

            if response:
                st.session_state.session_id = response.get("session_id")
                answer = response["answer"]
                st.session_state.messages.append({"role": "assistant", "content": answer})

                with st.chat_message("assistant"):
                    st.markdown(answer)

                with st.expander("Details"):
                    st.subheader("Answer")
                    st.code(answer)
                    st.subheader("Chat Model")
                    st.code(response["model"])
                    st.subheader("Embedding Model")
                    st.code(embedding_model)
                    st.subheader("Session ID")
                    st.code(response["session_id"])

                # Refresh sidebar session list
                st.session_state.sessions = list_sessions()
            else:
                st.error("Failed to get a response from the API. Please try again.")
